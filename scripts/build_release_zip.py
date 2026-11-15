"""Build field_shift_final.zip for submission (excludes secrets and raw data)."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "field_shift_final.zip"
SKIP_FILES = {".env", "field_shift_final.zip", "field_shift_web.zip"}


def skip_file(p: Path) -> bool:
    rel = p.relative_to(ROOT)
    if p.name in SKIP_FILES:
        return True
    if "__pycache__" in rel.parts or ".pytest_cache" in rel.parts:
        return True
    if ".git" in rel.parts or ".venv" in rel.parts or "venv" in rel.parts:
        return True
    return bool(len(rel.parts) >= 2 and rel.parts[0] == "data" and rel.parts[1] == "raw")


def main() -> None:
    if OUT.is_file():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [
                d for d in dirnames if d not in (".venv", "venv", ".git", "__pycache__", ".pytest_cache")
            ]
            if Path(dirpath).name == "data":
                dirnames[:] = [d for d in dirnames if d != "raw"]
            for fn in filenames:
                p = Path(dirpath) / fn
                if skip_file(p):
                    continue
                zf.write(p, p.relative_to(ROOT).as_posix())

    size = OUT.stat().st_size
    print(f"ZIP: {OUT}")
    print(f"Size: {size:,} bytes ({size / 1024 / 1024:.2f} MiB)")
    with zipfile.ZipFile(OUT) as zf:
        names = zf.namelist()
        env_hits = [n for n in names if n == ".env" or n.endswith("/.env")]
        print(f".env in zip: {bool(env_hits)} {env_hits}")
        tops = sorted({n.split("/")[0] for n in names if n})
        print("Top level:")
        for t in tops:
            print(f"  {t}")


if __name__ == "__main__":
    main()