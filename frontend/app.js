// Field Shift frontend. Every number shown comes from the Field Shift API.
(() => {
  const API = window.API_BASE || "";
  const MONTHS = ["J","F","M","A","M","J","J","A","S","O","N","D"];
  const state = { lat: null, lon: null, water: "dryland", soil: "medium", year: "normal" };
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v)) ? null : Number(v).toFixed(d));

  async function api(path, opts = {}) {
    const res = await fetch(API + path, { headers: { "Content-Type": "application/json" }, ...opts });
    if (!res.ok) {
      let detail = "";
      try { const j = await res.json(); detail = j.detail || j.message || ""; } catch (_) {}
      throw new Error(detail || `Request failed (${res.status})`);
    }
    return res.json();
  }

  // API status
  api("/health").then(() => { $("apiStatus").textContent = "Data service online"; $("apiStatus").className = "status ok"; })
    .catch(() => { $("apiStatus").textContent = "Data service offline"; $("apiStatus").className = "status down"; });

  // Map. Starts on a real sample field so the page is never empty on arrival:
  // point 283, a textbook wheat-then-fallow history near the middle of the county.
  const DEFAULT_FIELD = { lat: 37.97648, lon: -100.75804 };

  const map = L.map("map", { scrollWheelZoom: false, zoomControl: false })
    .setView([DEFAULT_FIELD.lat, DEFAULT_FIELD.lon], 11);
  L.control.zoom({ position: "topright" }).addTo(map);

  const streets = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 16, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
  const satellite = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    { maxZoom: 17, attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics" });
  L.control.layers({ "Map": streets, "Satellite": satellite }, null, { position: "topright" }).addTo(map);

  // The measured fields. Showing them stops people clicking where we have no data.
  const fieldDots = L.layerGroup().addTo(map);
  api("/sample-points").then((d) => {
    (d.points || []).forEach(([la, lo]) => {
      L.circleMarker([la, lo], {
        radius: 2.6, fillColor: "#F0C97A", fillOpacity: .95, interactive: false,
        color: "#5A3E12", weight: 1, opacity: .7,   // outlined so it reads on map *and* satellite
      }).addTo(fieldDots);
    });
    $("mapNote").textContent = `${d.count} measured fields · tap within ${d.max_distance_km} km of one`;
  }).catch(() => { $("mapNote").textContent = "Tap anywhere on Finney County farmland"; });

  let marker = null;
  const pin = L.divIcon({
    className: "fieldPin", html: '<span></span>', iconSize: [22, 22], iconAnchor: [11, 11],
  });

  async function selectField(lat, lon) {
    state.lat = +(+lat).toFixed(5); state.lon = +(+lon).toFixed(5);
    const ll = [state.lat, state.lon];
    if (marker) marker.setLatLng(ll); else marker = L.marker(ll, { icon: pin }).addTo(map);
    $("fieldInfo").textContent = `Field at ${state.lat}, ${state.lon}. Looking up soil and crop history…`;
    $("go").disabled = false;
    try {
      const f = await api("/field-info", { method: "POST", body: JSON.stringify({ lat: state.lat, lon: state.lon }) });
      const soil = f.soil || {};
      const tex = soil.texture_class || f.texture_class;
      if (tex && ["light", "medium", "heavy"].includes(tex)) setSeg("soil", tex);
      const guess = f.current_rotation_guess;
      if (guess && [...$("current").options].some((o) => o.value === guess)) $("current").value = guess;
      const bits = [];
      if (f.distance_km !== undefined) bits.push(`nearest sample field ${num(f.distance_km, 1)} km away`);
      const om = soil.organic_matter_pct ?? soil.om_pct ?? f.om_pct;
      if (om !== undefined && om !== null) bits.push(`organic matter ${num(om, 1)}%`);
      if (tex) bits.push(`${tex} soil`);
      $("fieldInfo").textContent = bits.length ? `Field found: ${bits.join(", ")}.` : "Field found.";
    } catch (err) {
      $("fieldInfo").textContent = `${err.message}. Try a point inside Finney County cropland.`;
    }
    loadTrends();
  }

  map.on("click", (e) => selectField(e.latlng.lat, e.latlng.lng));

  // Land on a real field rather than an empty map.
  selectField(DEFAULT_FIELD.lat, DEFAULT_FIELD.lon);

  // Segmented controls
  function setSeg(name, v) {
    const g = document.querySelector(`.seg[data-name="${name}"]`);
    g.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === v));
    state[name] = v;
  }
  document.querySelectorAll(".seg").forEach((g) => g.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return; setSeg(g.dataset.name, b.dataset.v);
  }));

  // Priorities, max 3
  $("prios").addEventListener("change", (e) => {
    const on = [...$("prios").querySelectorAll("input:checked")];
    if (on.length > 3) { e.target.checked = false; $("prioErr").textContent = "Pick at most 3. Untick one first."; }
    else $("prioErr").textContent = "";
  });

  // Compare
  $("go").addEventListener("click", async () => {
    if (state.lat === null) { $("fieldInfo").textContent = "Click your field on the map first."; return; }
    const priorities = [...$("prios").querySelectorAll("input:checked")].map((i) => i.value);
    const body = {
      lat: state.lat, lon: state.lon, current_rotation: $("current").value, water_source: state.water,
      soil_texture: state.soil, priorities, year_type: state.year, soil_om_pct: null,
      machinery: { sorghum_planter: $("mSorghum").checked, forage_equipment: $("mForage").checked },
    };
    $("empty").hidden = true; $("error").hidden = true; $("summary").hidden = true;
    $("cards").innerHTML = ""; $("loading").hidden = false; $("go").disabled = true;
    try {
      const r = await api("/compare", { method: "POST", body: JSON.stringify(body) });
      render(r, body);
    } catch (err) {
      $("error").innerHTML = `<b>We could not run the comparison.</b><br>${esc(err.message)}`;
      $("error").hidden = false;
    } finally { $("loading").hidden = true; $("go").disabled = false; }
  });

  function render(r, req) {
    const rots = (r.rotations || []).slice().sort((a, b) => String(a.id).localeCompare(String(b.id)));
    const best = r.best_match;
    const sc = r.scenario || {};
    const yearLabel = { normal: "a normal year", dry: "a dry year", hot: "a hot and dry year" }[req.year_type] || req.year_type;
    const bestRot = rots.find((x) => x.id === best);
    $("summary").innerHTML = `<div><span class="hint">Planning for</span><br><b>${esc(yearLabel)}</b>${sc.rain_change_pct ? ` <span class="hint">(rain ${sc.rain_change_pct}%${sc.temp_change_c ? `, +${sc.temp_change_c} °C` : ""})</span>` : ""}</div>
      <div><span class="hint">Best match for your priorities</span><br><b>${esc(bestRot ? bestRot.name : best || "None")}</b></div>
      <div><span class="hint">Your priorities</span><br><b>${esc(req.priorities.join(", ") || "balanced")}</b></div>`;
    $("summary").hidden = false;
    $("cards").innerHTML = rots.map((x) => card(x, x.id === best)).join("");
  }

  function card(x, isBest) {
    const h = x.crop_health || {};
    const lo = Number(h.p10), mid = Number(h.p50), hi = Number(h.p90);
    const pos = (v) => Math.max(0, Math.min(100, ((v - 0.3) / 0.6) * 100));
    const conf = (x.confidence || "").toLowerCase();
    const s = x.scores || {};
    const bare = new Set(x.bare_months || []);
    const row = (k, v) => (v === null || v === undefined || v === "" ? "" : `<span>${k}</span><b>${esc(v)}</b>`);
    const list = (cls, title, items) => (items && items.length ? `<div class="list ${cls}"><h4>${title}</h4><ul>${items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul></div>` : "");
    return `<article class="card ${isBest ? "best" : ""}">
      <div class="top"><div><div class="opt">Option ${esc(x.id)}</div><h3>${esc(x.name)}</h3></div>
        <div class="badges">${isBest ? '<span class="badge b-best">Best match</span>' : ""}${x.is_current ? '<span class="badge b-cur">Your current</span>' : ""}${conf ? `<span class="badge b-${esc(conf)}">${esc(conf[0].toUpperCase() + conf.slice(1))} confidence</span>` : ""}</div></div>
      ${s.total !== undefined && s.total !== null ? `<div class="score"><b>${num(s.total, 1)}</b><span>/ 10 for your priorities</span></div>` : ""}
      ${Number.isFinite(mid) ? `<div><div class="hint">Expected wheat health (peak NDVI)</div>
        <div class="range"><div class="band" style="left:${pos(lo)}%;width:${Math.max(2, pos(hi) - pos(lo))}%"></div><div class="mid" style="left:${pos(mid)}%"></div></div>
        <div class="rangeLbl"><span>${num(lo)}</span><span><b>${num(mid)}</b> likely</span><span>${num(hi)}</span></div></div>` : ""}
      <div class="kv">
        ${row("Water need", x.water_need_mm !== null && x.water_need_mm !== undefined ? `${Math.round(x.water_need_mm)} mm / year` : null)}
        ${row("Bare days", x.bare_days_per_year !== null && x.bare_days_per_year !== undefined ? `${Math.round(x.bare_days_per_year)} / year` : null)}
        ${row("Erosion risk", x.erosion_risk)}
        ${row("Nitrogen fixing", x.nitrogen_fixing === undefined ? null : x.nitrogen_fixing ? "Yes" : "No")}
      </div>
      <div><div class="hint">Low cover months</div><div class="cal">${MONTHS.map((_, i) => `<i class="${bare.has(i + 1) ? "bare" : ""}" title="${bare.has(i + 1) ? "Low cover" : "Covered"}"></i>`).join("")}</div>
        <div class="calLbl">${MONTHS.map((m) => `<span>${m}</span>`).join("")}</div></div>
      ${list("why", "Why", x.why)}${list("ev", "Evidence", x.evidence)}${list("risk", "Risks", x.risks)}${list("gap", "Data gaps", x.data_gaps)}
    </article>`;
  }

  // Trends
  let chart = null;
  async function loadTrends() {
    try {
      const t = await api(`/trends?lat=${state.lat}&lon=${state.lon}`);
      let rows = Array.isArray(t) ? t : Array.isArray(t.years) && typeof t.years[0] === "object" ? t.years : t.rows || t.yearly || null;
      if (!rows && Array.isArray(t.years)) rows = t.years.map((y, i) => ({ year: y, season_rain_mm: (t.season_rain_mm || t.growing_season_rain_mm || [])[i], heat_days: (t.heat_days || [])[i], peak_ndvi: (t.peak_ndvi || [])[i] }));
      if (!rows || !rows.length) { $("trendHint").textContent = "No trend data for this field."; return; }
      const years = rows.map((r) => r.year);
      const rain = rows.map((r) => r.season_rain_mm ?? r.growing_season_rain_mm ?? null);
      const heat = rows.map((r) => r.heat_days ?? null);
      const ndvi = rows.map((r) => (r.peak_ndvi === undefined ? null : r.peak_ndvi));
      $("trendHint").textContent = "Season rain (October to June) and heat days from NASA POWER, peak wheat greenness from NASA MODIS.";
      if (chart) chart.destroy();
      chart = new Chart($("trendChart"), {
        data: { labels: years, datasets: [
          { type: "bar", label: "Season rain (mm)", data: rain, backgroundColor: "#9CC0E0", yAxisID: "y" },
          { type: "line", label: "Heat days above 32 °C", data: heat, borderColor: "#B7791F", backgroundColor: "#B7791F", yAxisID: "y1", tension: .25 },
          { type: "line", label: "Peak NDVI", data: ndvi, borderColor: "#2E7D4F", backgroundColor: "#2E7D4F", yAxisID: "y2", tension: .25, spanGaps: true },
        ] },
        options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
          scales: { y: { title: { display: true, text: "Rain (mm)" } }, y1: { position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "Heat days" } }, y2: { display: false, min: 0, max: 1 } } },
      });
    } catch (err) { $("trendHint").textContent = `Trends unavailable: ${err.message}`; }
  }

  // ---- Full-screen intro -------------------------------------------------
  // Real USDA field photography, slowly cross-fading and drifting. Drop an
  // intro.mp4 beside this file and it plays over the stills instead.
  (function intro() {
    const el = $("intro");
    if (!el) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const shots = [...el.querySelectorAll(".shot")];
    let i = 0, timer = null;

    if (shots.length > 1 && !reduce) {
      timer = setInterval(() => {
        shots[i].classList.remove("on");
        i = (i + 1) % shots.length;
        shots[i].classList.add("on");
      }, 6000);
    }

    const vid = $("introVid");
    vid.addEventListener("canplay", () => {
      vid.classList.add("on");
      if (timer) { clearInterval(timer); timer = null; }
      vid.play().catch(() => {});
    }, { once: true });

    function close(goToCompare) {
      if (timer) clearInterval(timer);
      el.classList.add("out");
      document.body.classList.remove("locked");
      // Scroll only once the overlay is gone and the scroll lock has lifted,
      // otherwise the smooth scroll is swallowed.
      setTimeout(() => {
        el.remove();
        if (!goToCompare) return;
        const target = document.getElementById("explore");
        if (!target) return;
        const head = document.querySelector(".topbar");
        const offset = (head ? head.getBoundingClientRect().height : 0) + 12;
        const top = target.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top: Math.max(0, top), behavior: reduce ? "auto" : "smooth" });
      }, 430);
    }

    const cBtn = $("creditBtn"), cBox = $("creditBox");
    if (cBtn && cBox) {
      cBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const open = cBox.hidden;
        cBox.hidden = !open;
        cBtn.setAttribute("aria-expanded", String(open));
      });
      // click anywhere else closes it, without dismissing the intro
      el.addEventListener("click", () => {
        if (!cBox.hidden) { cBox.hidden = true; cBtn.setAttribute("aria-expanded", "false"); }
      });
    }

    document.body.classList.add("locked");
    $("introGo").addEventListener("click", () => close(true));
    $("introSkip").addEventListener("click", () => close(false));
    addEventListener("keydown", (e) => { if (e.key === "Escape") close(false); }, { once: true });
    $("introGo").focus();
  })();

})();
