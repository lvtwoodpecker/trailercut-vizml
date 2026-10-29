"""Rank trailers by the angles worth showing in the report, and keep the extremes.

  order       trailer order vs film order (order_corr): chronological vs shuffled
  ending      share of matched shots from the film's last quarter: spoils vs avoids the ending
  sync        sound/picture sync rate: dialogue over its own scene vs laid over other footage
  model       per-film precision at trailer length: the model agrees with the editor vs misses

Minimum data per angle so a trailer with two lines can't top a list.
Writes data/features/case_studies.json (committed)

  python -m trailercut.case_studies [--n 5]
"""
import argparse
import json

import numpy as np
import pandas as pd

from .paths import FEATURES


def per_film_precision():
    p = pd.read_parquet(FEATURES / "model_preds.parquet")
    out = {}
    for i, g in p.groupby("imdb_id"):
        k = int(g.in_trailer.sum())
        if k >= 10:
            out[i] = float(g.in_trailer.values[np.argsort(-g.full.values)[:k]].mean())
    return pd.Series(out, name="model_precision")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    t = pd.read_parquet(FEATURES / "trailers.parquet").set_index("imdb_id")
    av = pd.read_csv(FEATURES / "av_sync.csv").set_index("imdb_id")
    t = t.join(av[["n_lines_paired"]]).join(per_film_precision())
    t["matched_shots"] = (t.match_rate * t.n_shots).round()
    angles = {
        "order": ("order_corr", t.matched_shots >= 15, ("chronological", "shuffled")),
        "ending": ("final_act_share", t.matched_shots >= 15, ("most from the ending", "least from the ending")),
        "sync": ("sync_rate", t.n_lines_paired >= 5, ("sound over its own scene", "sound over other footage")),
        "model": ("model_precision", t.model_precision.notna(), ("model agrees with the editor", "model misses")),
    }
    out = {}
    for name, (col, ok, (hi, lo)) in angles.items():
        d = t[ok & t[col].notna()].sort_values(col, ascending=False)
        row = lambda r: dict(imdb_id=r.Index, title=r.title, genre=r.primary_genre, value=round(float(getattr(r, col)), 3))
        out[name] = {"column": col, "n_trailers": len(d), hi: [row(r) for r in d.head(a.n).itertuples()],
                     lo: [row(r) for r in d.tail(a.n).iloc[::-1].itertuples()]}
    (FEATURES / "case_studies.json").write_text(json.dumps(out, indent=2))
    for name, v in out.items():
        print(f"\n{name} ({v['column']}, {v['n_trailers']} trailers)")
        for k, rows in v.items():
            if isinstance(rows, list):
                print(f"  {k}: " + "; ".join(f"{r['title']} {r['value']}" for r in rows))


if __name__ == "__main__":
    main()
