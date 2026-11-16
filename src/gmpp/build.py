"""raw files -> data/processed/{snapshots,panel,links}.

    python -m gmpp.build
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .clean import read_raw, to_canonical
from .match import link_snapshots

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def build(raw_dir: Path = RAW, out_dir: Path = OUT) -> pd.DataFrame:
    cfg = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, reports = [], []
    for s in cfg["snapshots"]:
        path = raw_dir / s["file"]
        if not path.exists():
            reports.append({"snapshot": s["id"], "missing_file": s["file"]})
            continue
        df, rep = to_canonical(read_raw(path), s["id"], str(s["snapshot"]))
        frames.append(df)
        reports.append(rep)
    snapshots = pd.concat(frames, ignore_index=True)
    panel, links = link_snapshots(snapshots)
    panel.to_parquet(out_dir / "panel.parquet", index=False)
    links.to_csv(out_dir / "links.csv", index=False)
    links[links.method == "review"].to_csv(out_dir / "links_to_review.csv", index=False)
    (out_dir / "build_report.json").write_text(json.dumps(reports, indent=2, default=str))
    return panel


if __name__ == "__main__":
    p = build()
    print(p.groupby("snapshot").agg(projects=("project_name", "size"), uids=("project_uid", "nunique")))
