"""Match each trailer shot to its most similar film shot, with an explicit "no match" option.

Similarity between trailer shot m and film shot n: for each of m's 3 keyframes take the best
cosine over n's 3 keyframes, then average. A shot is only matched if its best score beats the
impostor threshold: the 95th percentile of best scores when the same trailer shot is matched
against a *different* film (so ~5% of non-source shots would pass). A 1% cutoff rejects many
correct matches in the 0.72-0.79 range.

Blank shots (fades, black frames) match blank shots in any film and inflate the null tail.
Shots whose keyframes look like a solid black/white/grey frame are flagged on both sides:
blank trailer shots are never matched, blank film shots are never candidates.

The impostor null can't measure the main real error (right film, wrong shot), so the final cutoff
is picked from hand labels in labels/ (see trailercut.label_eval); raw scores are kept for that.

Outputs
  data/features/alignment/{imdb_id}.csv   one row per trailer shot
  data/features/alignment_threshold.json  impostor threshold + null stats

  python -m trailercut.align [--fpr 0.05] [--threshold T]
--threshold overrides the impostor cutoff (e.g. the one picked from hand labels); the null is still
computed and saved so the report shows its false-match rate.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import movienet_io as mn
from .paths import CACHE, CLIP, CLIP_TAG, FEATURES

OUT = FEATURES / "alignment"
TOPK = 5
# film shots within this many indices of the best are treated as the same moment when computing margin
# (shot/reverse-shot and continuous coverage make neighbours look alike)
NEIGHBOR = 3
BLANK = 0.9
BLANK_EMB = CACHE / f"blank_emb_{CLIP_TAG}.npy"


def blank_embeddings():
    if not BLANK_EMB.exists():
        from PIL import Image
        from .embed import DEVICE, load_model
        model, pre, _ = load_model()
        ims = [Image.new("RGB", (320, 180), c) for c in [(0, 0, 0), (255, 255, 255), (128, 128, 128)]]
        with torch.no_grad():
            e = model.encode_image(torch.stack([pre(i) for i in ims]).to(DEVICE).half()).float()
        np.save(BLANK_EMB, torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return torch.from_numpy(np.load(BLANK_EMB))


def load(kind, imdb_id):
    return torch.from_numpy(np.load(CLIP / kind / f"{imdb_id}.npy").astype(np.float32))


def blankness(E):
    """[n,3,d] -> [n]: mean over keyframes of the closest solid-color frame similarity."""
    return (E @ blank_embeddings().T).amax(-1).mean(-1)


def shot_sim(T, F):
    """T [M,3,d], F [N,3,d] -> [M,N]: mean over trailer frames of max over film frames.
    Blank film shots get -1 so they are never picked."""
    s = torch.einsum("mid,njd->mnij", T, F).amax(-1).mean(-1)
    s[:, blankness(F) > BLANK] = -1
    return s


def best_scores(T, F):
    return shot_sim(T, F).amax(1)


def available():
    t = {p.stem for p in (CLIP / "trailer").glob("*.npy")}
    f = {p.stem for p in (CLIP / "film").glob("*.npy")}
    return sorted(t & f)


def impostor_null(ids, rng, n_impostors=3):
    """Best score of each trailer shot against random other films."""
    null = []
    for imdb_id in tqdm(ids, desc="null"):
        T = load("trailer", imdb_id)
        T = T[blankness(T) <= BLANK]
        for other in rng.choice([i for i in ids if i != imdb_id], size=min(n_impostors, len(ids) - 1), replace=False):
            null.append(best_scores(T, load("film", other)).numpy())
    return np.concatenate(null)


def align_one(imdb_id, threshold):
    T, F = load("trailer", imdb_id), load("film", imdb_id)
    S = shot_sim(T, F)
    top = S.topk(TOPK, dim=1)
    best = top.indices[:, 0]
    idx = torch.arange(S.shape[1])
    far = (idx[None, :] - best[:, None]).abs() > NEIGHBOR
    second = torch.where(far, S, torch.full_like(S, -1)).amax(1)

    tshots = pd.read_csv(FEATURES / "trailer_shots" / f"{imdb_id}.csv")
    fshots = mn.shots(imdb_id)
    df = tshots[["shot_idx", "t_start", "t_end", "shot_len", "trailer_pos"]].copy()
    df["film_shot"] = best.numpy()
    df["score"] = top.values[:, 0].numpy()
    df["margin"] = df.score - second.numpy()
    df["blank"] = (blankness(T) > BLANK).numpy()
    df["matched"] = (df.score >= threshold) & ~df.blank
    df["film_story_pos"] = fshots.story_pos.values[df.film_shot]
    df.loc[~df.matched, "film_story_pos"] = np.nan
    df["top5"] = [" ".join(map(str, r)) for r in top.indices.numpy()]
    df["top5_scores"] = [" ".join(f"{v:.3f}" for v in r) for r in top.values.numpy()]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    ids = available()
    print(len(ids), "films with both trailer and film embeddings")
    null = impostor_null(ids, np.random.default_rng(0))
    threshold = float(np.quantile(null, 1 - a.fpr)) if a.threshold is None else a.threshold
    meta = dict(fpr=a.fpr if a.threshold is None else float((null >= threshold).mean()), threshold=threshold,
                source="impostor_null" if a.threshold is None else "manual", n_null=len(null),
                null_quantiles={q: float(np.quantile(null, q)) for q in (0.5, 0.9, 0.99, 0.999)})
    (FEATURES / "alignment_threshold.json").write_text(json.dumps(meta, indent=2))
    np.save(CACHE / f"alignment_null_{CLIP_TAG}.npy", null)
    print(meta)

    OUT.mkdir(parents=True, exist_ok=True)
    for imdb_id in tqdm(ids, desc="align"):
        align_one(imdb_id, threshold).round(4).to_csv(OUT / f"{imdb_id}.csv", index=False)


if __name__ == "__main__":
    main()
