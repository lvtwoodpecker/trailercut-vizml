"""Grade alignment backbones with rulers that don't come from B/32.

The hand labels were picked among B/32's top 5, in its order, with its scores shown. On near-identical
candidates (the film cutting back to one camera setup) a labeler takes the first, so the labels may
lean toward B/32 and every other method loses points for disagreeing with it. Six methods so different
(H/14, L/14, DINOv2, SSCD, two VLM setups) all landing near 70% top-1 makes that worth checking.
No new labeling; three automatic checks instead:

  labels     the old labels, but also counting a pick 1 or 3 film shots away as near, and listing the
             shots where every other backbone agrees on one pick that isn't the label (to look at)
  dialogue   a trailer line found in the film's subtitles (dialogue_align, text only) says where in the
             film that moment is; does the shot's picked film shot sit within 5 / 30 s of it? Voice-over
             laid on other footage makes this noisy, but the noise is the same for every backbone
  synthetic  each film keyframe, made to look like a trailer frame (crop, color, compression,
             downscale), has to find its own shot among the film's other keyframes. The answer is known
             exactly, and cut-backs to the same setup are in the gallery

  cands   (per backbone, needs its embeddings)   -> data/features/ruler/cands_{tag}.csv
  synth   (per backbone, GPU, needs keyframes)   -> data/features/ruler/synth_{tag}.csv
  grade   (anywhere, after the two above)        -> data/features/ruler_eval.json, ruler_consensus.csv

  TRAILERCUT_CLIP=l14 python -m trailercut.ruler cands|synth [--ids tt... ]
  python -m trailercut.ruler grade
Default films: the hand-labeled ones.
"""
import argparse
import io
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance

from . import movienet_io as mn
from .label_eval import load_gold, one_per_shot
from .paths import CLIP_TAG, FEATURES

OUT = FEATURES / "ruler"
NEAR = (1, 3)       # film shots away that count as near
DIALOGUE_S = (5, 30)  # seconds; 30 is av_sync's SYNC
BOOT = 2000


def labeled_ids():
    return sorted(one_per_shot(load_gold()).imdb_id.unique())


def cands(ids):
    from .align import align_one
    al = pd.concat([align_one(i, threshold=0.0).assign(imdb_id=i) for i in ids])
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["imdb_id", "shot_idx", "t_start", "t_end", "film_shot", "score", "blank", "top5", "top5_scores"]
    al[cols].to_csv(OUT / f"cands_{CLIP_TAG}.csv", index=False)
    print(f"{CLIP_TAG}: {len(al)} trailer shots in {len(ids)} films -> {OUT / f'cands_{CLIP_TAG}.csv'}")


def trailerish(im, rng):
    """A film keyframe roughly as a YouTube trailer shows it: re-framed, re-graded, smaller, compressed."""
    w, h = im.size
    s = rng.uniform(0.85, 0.95)
    cw, ch = int(w * s), int(h * s)
    x, y = int(rng.uniform(0, w - cw)), int(rng.uniform(0, h - ch))
    im = im.crop((x, y, x + cw, y + ch))
    for enh, lo, hi in [(ImageEnhance.Brightness, 0.8, 1.2), (ImageEnhance.Contrast, 0.8, 1.25), (ImageEnhance.Color, 0.7, 1.3)]:
        im = enh(im).enhance(rng.uniform(lo, hi))
    im = im.resize((480, max(1, round(480 * ch / cw))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(rng.integers(30, 60)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


@torch.no_grad()
def synth(ids):
    from .align import BLANK, blankness, load
    from .embed import DEVICE, load_model
    model, pre, _ = load_model()
    rows = []
    for i in ids:
        F = load("film", i)                     # [N, 3, d], normalized
        ok = ~(blankness(F) > BLANK)
        G = F[:, [0, 2]]                        # gallery: first and last keyframe of every shot
        shots = np.where(ok.numpy())[0]
        rng = np.random.default_rng(int(i[2:]))
        for b in range(0, len(shots), 128):
            batch = shots[b:b + 128]
            x = torch.stack([pre(trailerish(mn.open_image(mn.keyframe_paths(i, int(s))[1]).convert("RGB"), rng)) for s in batch])
            q = torch.nn.functional.normalize(model.encode_image(x.to(DEVICE).half()).float(), dim=-1).cpu()
            sim = torch.einsum("qd,njd->qnj", q, G).amax(-1)
            sim[:, ~ok] = -1
            top = sim.topk(2, dim=1)
            rows += [dict(imdb_id=i, shot_idx=int(s), pick=int(p[0]), score=float(v[0]), second=float(v[1]))
                     for s, p, v in zip(batch, top.indices, top.values)]
        print(i, len(shots), "queries")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"synth_{CLIP_TAG}.csv", index=False)
    d = pd.DataFrame(rows)
    print(f"{CLIP_TAG}: synthetic top-1 {(d.pick == d.shot_idx).mean():.3f} on {len(d)} queries")


def boot(hit, film):
    """Mean and a 95% interval, resampling films (shots within a film aren't independent)."""
    hit, film = np.asarray(hit, float), np.asarray(film)
    fs = np.unique(film)
    by = {f: hit[film == f] for f in fs}
    rng = np.random.default_rng(0)
    m = [np.concatenate([by[f] for f in rng.choice(fs, len(fs))]).mean() for _ in range(BOOT)]
    return dict(mean=round(float(hit.mean()), 3), ci=[round(float(np.quantile(m, q)), 3) for q in (0.025, 0.975)], n=int(len(hit)))


def grade():
    c = {p.stem.removeprefix("cands_"): pd.read_csv(p, dtype={"imdb_id": str}) for p in sorted(OUT.glob("cands_*.csv"))}
    assert "b32" in c, f"need {OUT}/cands_b32.csv"
    res = dict(backbones=sorted(c), labels={}, dialogue={}, synthetic={})

    # 1. old labels, exact and near
    g = one_per_shot(load_gold())
    g = g[g.label.str.isdigit()].assign(label=lambda x: x.label.astype(int))
    for t, d in c.items():
        m = g.merge(d, on=["imdb_id", "shot_idx"])
        m = m[~m.blank]
        off = (m.film_shot - m.label).abs()
        res["labels"][t] = dict(exact=boot(off == 0, m.imdb_id), **{f"within_{k}": boot(off <= k, m.imdb_id) for k in NEAR})
    others = [t for t in c if t != "b32"]
    if others:
        picks = pd.concat({t: c[t].set_index(["imdb_id", "shot_idx"]).film_shot for t in c}, axis=1)
        m = g.set_index(["imdb_id", "shot_idx"]).join(picks, how="inner")
        same = m[others].nunique(axis=1).eq(1) & m[others[0]].ne(m.label)
        cons = m[same].reset_index()[["imdb_id", "shot_idx", "label", "b32", others[0]]].rename(columns={others[0]: "others_pick"})
        cons.to_csv(FEATURES / "ruler_consensus.csv", index=False)
        res["labels"]["others_agree_against_label"] = dict(
            n=int(len(cons)), of=int(len(m)), b32_equals_label=int((cons.b32 == cons.label).sum()), file="ruler_consensus.csv")

    # 2. dialogue
    shots = {}
    for i in c["b32"].imdb_id.unique():
        p = FEATURES / "dialogue_alignment" / f"{i}.csv"
        if not p.exists():
            continue
        dl = pd.read_csv(p)
        dl = dl[dl.matched.astype(bool)]
        fs = mn.shots(i)
        mid = ((fs.t_start + fs.t_end) / 2).values
        for r in c["b32"][c["b32"].imdb_id == i].itertuples():
            ov = dl[(dl.t_start < r.t_end) & (dl.t_end > r.t_start)]
            if len(ov):
                line = ov.iloc[0]
                tm = (max(r.t_start, line.t_start) + min(r.t_end, line.t_end)) / 2   # overlap midpoint
                k = (tm - line.t_start) / max(line.t_end - line.t_start, 1e-6)
                shots[(i, r.shot_idx)] = (line.film_t_start + k * (line.film_t_end - line.film_t_start), fs, mid)
    for t, d in c.items():
        hit = {s: [] for s in DIALOGUE_S}
        film = []
        for r in d[~d.blank.astype(bool)].itertuples():
            if (r.imdb_id, r.shot_idx) in shots:
                ft, fs, _ = shots[(r.imdb_id, r.shot_idx)]
                a, b = fs.t_start.iloc[r.film_shot], fs.t_end.iloc[r.film_shot]
                gap = max(a - ft, ft - b, 0)        # 0 when the line's film time falls inside the picked shot
                for s in DIALOGUE_S:
                    hit[s].append(gap <= s)
                film.append(r.imdb_id)
        res["dialogue"][t] = {f"within_{s}s": boot(hit[s], film) for s in DIALOGUE_S} if film else None

    # 3. synthetic
    for p in sorted(OUT.glob("synth_*.csv")):
        t, d = p.stem.removeprefix("synth_"), pd.read_csv(p, dtype={"imdb_id": str})
        off = (d.pick - d.shot_idx).abs()
        res["synthetic"][t] = dict(exact=boot(off == 0, d.imdb_id), **{f"within_{k}": boot(off <= k, d.imdb_id) for k in NEAR},
                                   by_film={i: round(float((x.pick == x.shot_idx).mean()), 3) for i, x in d.groupby("imdb_id")})
    (FEATURES / "ruler_eval.json").write_text(json.dumps(res, indent=2))
    for part in ("labels", "dialogue", "synthetic"):
        print(f"== {part}")
        for t, v in res[part].items():
            if isinstance(v, dict) and t in c or part == "synthetic":
                print(f"  {t:8s}", {k: (x["mean"], x["ci"]) for k, x in v.items() if isinstance(x, dict) and "mean" in x})
            elif v is not None:
                print(f"  {t}", v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cands", "synth", "grade"])
    ap.add_argument("--ids", nargs="*", help="films (default: the hand-labeled ones)")
    a = ap.parse_args()
    if a.cmd == "grade":
        return grade()
    ids = a.ids or labeled_ids()
    (cands if a.cmd == "cands" else synth)(ids)


if __name__ == "__main__":
    main()
