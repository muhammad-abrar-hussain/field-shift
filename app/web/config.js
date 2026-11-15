// Where the Field Shift API lives.
// Empty string = same server as this page (FastAPI serving app/web at "/", or Render).
// When the page is opened from "python -m http.server 5500", it talks to the local API on port 8000.
window.API_BASE = window.API_BASE ?? (location.port === "5500" ? "http://localhost:8000" : "");
