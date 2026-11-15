# Download USDA Cropland Data Layer GeoTIFFs for Finney County, KS (FIPS 20055), 2013–2025.

from __future__ import annotations

import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

REQUEST_TIMEOUT_S = 120
MAX_ATTEMPTS = 3
RETRY_WAIT_S = 10
YEARS = range(2013, 2026)
FIPS = "20055"
CDL_API = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "cdl"


def _parse_return_url(xml_text: str) -> str | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        match = re.search(r"<returnURL>(.*?)</returnURL>", xml_text, re.I | re.S)
        return match.group(1).strip() if match else None

    for elem in root.iter():
        if elem.tag.split("}")[-1].lower() == "returnurl" and elem.text:
            return elem.text.strip()
    return None


def _file_size_mb(path: Path) -> float | None:
    if not path.is_file():
        return None
    return path.stat().st_size / (1024 * 1024)


def _download_year(year: int, dest: Path) -> tuple[str, float | None, str | None]:
    """Returns (status, size_mb, error_message). status: skipped|downloaded|failed."""
    if dest.is_file():
        return "skipped", _file_size_mb(dest), None

    last_err: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            meta = requests.get(
                CDL_API,
                params={"year": year, "fips": FIPS},
                timeout=REQUEST_TIMEOUT_S,
            )
            meta.raise_for_status()
            return_url = _parse_return_url(meta.text)
            if not return_url:
                raise ValueError("No returnURL in CDL API response")

            tif = requests.get(return_url, timeout=REQUEST_TIMEOUT_S)
            tif.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tif.content)

            if dest.stat().st_size == 0:
                dest.unlink(missing_ok=True)
                raise ValueError("Downloaded file is empty")

            return "downloaded", _file_size_mb(dest), None
        except (requests.RequestException, ValueError, OSError) as exc:
            last_err = str(exc)
            if dest.exists():
                dest.unlink(missing_ok=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_S)

    return "failed", None, last_err


def main() -> int:
    rows: list[tuple[int, str, str, str]] = []

    for year in YEARS:
        dest = OUT_DIR / f"CDL_{year}_{FIPS}.tif"
        status, size_mb, err = _download_year(year, dest)
        size_str = f"{size_mb:.2f}" if size_mb is not None else "—"
        note = err or ""
        rows.append((year, status, size_str, note))
        print(f"{year}: {status}" + (f" ({size_mb:.2f} MB)" if size_mb is not None else "") + (f" — {err}" if err else ""))

    print()
    print(f"{'Year':<6} {'Status':<12} {'Size (MB)':<12}")
    print("-" * 32)
    for year, status, size_str, _ in rows:
        print(f"{year:<6} {status:<12} {size_str:<12}")

    failed = sum(1 for r in rows if r[1] == "failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())