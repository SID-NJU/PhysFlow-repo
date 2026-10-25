#!/usr/bin/env python3
"""Summarize CLIP-SIM and DINO/CLIP consistency means from detail CSVs."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute mean CLIP/DINO metrics from detail CSV files.")
    p.add_argument("--detail-csvs", nargs="+", required=True, help="detail_clip_consistency_*.csv files")
    p.add_argument("--output", default="", help="Optional output CSV path for summary")
    return p.parse_args()

def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)

def run() -> None:
    args = parse_args()
    rows_out = []

    for csv_path in [Path(p).expanduser().resolve() for p in args.detail_csvs]:
        rows = read_rows(csv_path)
        clip_sim = []
        clip_cons = []
        dino_cons = []

        for r in rows:
            try:
                clip_sim.append(float(r.get("clip_sim", "nan")))
            except Exception:
                pass
            try:
                clip_cons.append(float(r.get("clip_frame_consistency", "nan")))
            except Exception:
                pass
            try:
                val = float(r.get("dino_frame_consistency", "nan"))
                if val == val:
                    dino_cons.append(val)
            except Exception:
                pass

        rows_out.append({
            "detail_csv": str(csv_path),
            "n_videos": len(rows),
            "clip_sim_mean": mean([v for v in clip_sim if v == v]),
            "clip_frame_consistency_mean": mean([v for v in clip_cons if v == v]),
            "dino_frame_consistency_mean": mean([v for v in dino_cons if v == v]),
        })

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "detail_csv",
                    "n_videos",
                    "clip_sim_mean",
                    "clip_frame_consistency_mean",
                    "dino_frame_consistency_mean",
                ],
            )
            writer.writeheader()
            writer.writerows(rows_out)
    else:
        for r in rows_out:
            print(r)

if __name__ == "__main__":
    run()
