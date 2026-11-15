// Where the Field Shift API lives.
//
//  - Served by the backend itself (Render Docker, or local uvicorn on :8000):
//    same origin, so API_BASE stays "".
//  - Static dev server ("python -m http.server 5500" from this folder):
//    talk to the local API on port 8000.
//  - Static host with no API (Vercel): point at the deployed backend below.
//
// After deploying the backend to Render, replace API_URL with that service URL,
// e.g. "https://field-shift-nasa.onrender.com" (no trailing slash).
window.API_URL = window.API_URL ?? "";

window.API_BASE =
  window.API_BASE ??
  (location.port === "5500"
    ? "http://localhost:8000"
    : location.hostname.endsWith("vercel.app") || window.API_URL
      ? window.API_URL
      : "");
