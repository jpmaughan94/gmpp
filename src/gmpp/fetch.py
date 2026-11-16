"""Download the raw GMPP files listed in config/sources.yaml into data/raw/.

Raw files are immutable: an existing file is never overwritten, and a SHA-256
manifest records exactly what was downloaded and when.

    python -m gmpp.fetch            # download anything missing
    python -m gmpp.fetch --check    # verify hashes against the manifest
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"


def sources() -> list[dict]:
    cfg = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text())
    return cfg["snapshots"] + [cfg["reset_list"]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def fetch_all(force: bool = False) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    man = load_manifest()
    for s in sources():
        dest = RAW / s["file"]
        if dest.exists() and not force:
            man.setdefault(s["file"], {"sha256": sha256(dest), "url": s["url"], "fetched": "pre-existing"})
            continue
        req = urllib.request.Request(s["url"], headers={"User-Agent": "gmpp-research (github)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        man[s["file"]] = {"sha256": sha256(dest), "url": s["url"],
                          "fetched": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
        print(f"fetched {s['file']}")
    MANIFEST.write_text(json.dumps(man, indent=2))
    return man


def check() -> list[str]:
    man = load_manifest()
    bad = [f for f, m in man.items() if not (RAW / f).exists() or sha256(RAW / f) != m["sha256"]]
    for f in bad:
        print(f"CHANGED OR MISSING: {f}")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    raise SystemExit(1 if (check() if args.check else (fetch_all() and False)) else 0)
