"""Check every data file against data/MANIFEST.json (checksum + row count).

    python pipeline/verify_data.py            # verify; exit 1 on any mismatch or stray file
    python pipeline/verify_data.py --update   # re-record checksums after an intended rebuild

A file under data/ that is not in the manifest counts as a failure. That is the
"lock": nothing enters or changes in data/ without a recorded producer and reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/MANIFEST.json"
IGNORED = {"data/MANIFEST.json", "data/DATA.md"}
OPTIONAL_LOCAL = {"data/raw/ipl_json.zip", "data/processed/player_matches.csv"}  # verified only once built


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(p: Path) -> int | None:
    if p.suffix == ".csv":
        with p.open("rb") as f:
            return sum(1 for _ in f) - 1
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    man = json.loads(MANIFEST.read_text())
    ok = True
    for rel, entry in man["files"].items():
        p = ROOT / rel
        if not p.exists():
            if rel in OPTIONAL_LOCAL:
                print(f"skip     {rel} (not present yet)"); continue
            print(f"MISSING  {rel}"); ok = False; continue
        digest, n = sha256(p), rows(p)
        if a.update:
            entry["sha256"], entry["rows"] = digest, n
            print(f"recorded {rel}")
        elif digest != entry["sha256"] or n != entry.get("rows"):
            print(f"CHANGED  {rel}  (rows {entry.get('rows')} -> {n})"); ok = False
        else:
            print(f"ok       {rel}")
    on_disk = {str(p.relative_to(ROOT)) for p in (ROOT / "data").rglob("*") if p.is_file()}
    stray = sorted(on_disk - set(man["files"]) - IGNORED - {"data/processed/legacy_check.csv"})
    for s in stray:
        print(f"UNTRACKED {s}  (add it to data/MANIFEST.json with its producer, or delete it)"); ok = False
    if a.update:
        MANIFEST.write_text(json.dumps(man, indent=2) + "\n")
        return 0
    print("data lock: OK" if ok else "data lock: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
