"""Re-rank CLIP's alignment candidates with a copy-detection model (SSCD).

CLIP matches what a frame shows ("two people talking in a red bathroom"), so a look-alike shot from
elsewhere in the film can outscore the real source. SSCD (Pizzi et al. 2022) is trained to recognize
the same image after crops, color edits and overlays, which is the trailer-vs-film situation. CLIP
still narrows each trailer shot to its top-k film shots; SSCD only re-scores those.

Shot similarity uses the same rule as align.py: mean over the trailer shot's 3 keyframes of the best
match among the film shot's 3 keyframes. Trailer frames get the same per-trailer letterbox crop.

  python -m trailercut.rerank --eval     compare CLIP, SSCD and a blend on the hand-labeled trailers
Writes data/features/rerank_eval.json (committed)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from . import movienet_io as mn
from .align import BLANK, blankness, load, shot_sim
from .align_tune import score
from .embed import DEVICE, TRAILER_FRAMES, bar_box
from .label_eval import load_gold
from .paths import CACHE, FEATURES

WEIGHTS = CACHE / "models" / "sscd_disc_mixup.torchscript.pt"
URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
K = 20


def sscd():
    if not WEIGHTS.exists():
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(URL, str(WEIGHTS))
    return torch.jit.load(str(WEIGHTS), map_location="cpu").to(DEVICE).eval()


def to_tensor(ref, box=None):
    im = mn.open_image(ref).convert("RGB")
    if box is not None:
        im = im.crop(box)
    x = torch.from_numpy(np.asarray(im.resize((288, 288)), dtype=np.float32) / 255).permute(2, 0, 1)
    return (x - MEAN) / STD


@torch.no_grad()
def describe(model, refs, box=None, bs=96):
    out = []
    for i in range(0, len(refs), bs):
        x = torch.stack([to_tensor(r, box) for r in refs[i:i + bs]]).to(DEVICE)
        out.append(torch.nn.functional.normalize(model(x), dim=-1).cpu())
    return torch.cat(out).numpy()


def rerank_film(model, imdb_id, k=K):
    """-> CLIP similarity matrix, blank mask, top-k candidates, their CLIP and SSCD scores."""
    T, F = load("trailer", imdb_id), load("film", imdb_id)
    S = shot_sim(T, F).numpy()
    blank = (blankness(T) > BLANK).numpy()
    cand = np.argsort(-S, 1)[:, :k]
    m = len(S)
    tpaths = [TRAILER_FRAMES / imdb_id / mn.keyframe_name(s, j) for s in range(m) for j in range(3)]
    tq = describe(model, tpaths, bar_box(tpaths)).reshape(m, 3, -1)
    shots = np.unique(cand)
    fq = describe(model, [p for s in shots for p in mn.keyframe_paths(imdb_id, s)]).reshape(len(shots), 3, -1)
    pos = {s: n for n, s in enumerate(shots)}
    fc = fq[np.vectorize(pos.get)(cand)]  # [m, k, 3, d]
    q = np.einsum("mid,mkjd->mkij", tq, fc).max(-1).mean(-1)  # [m, k]
    return S, blank, cand, np.take_along_axis(S, cand, 1), q


def evaluate(films, gold, how, t):
    preds, labels = [], []
    for i, (S, blank, cand, c, q) in films.items():
        g = gold[gold.imdb_id == i].sort_values("shot_idx")
        s = {"clip": c, "sscd": q, "blend": c + q}[how]
        best = s.argmax(1)
        shot = np.where((s.max(1) >= t) & ~blank, cand[np.arange(len(cand)), best], -1)
        preds.append(shot[g.shot_idx.values])
        labels.append(g.label.values)
    return score(np.concatenate(preds), pd.Series(np.concatenate(labels)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    gold = load_gold().drop_duplicates("key")
    gold = gold[gold.label != "u"]
    gold["imdb_id"] = gold.key.str.split(":").str[0]
    gold["shot_idx"] = gold.key.str.split(":").str[1].astype(int)
    model = sscd()
    films = {i: rerank_film(model, i) for i in tqdm(sorted(gold.imdb_id.unique()), desc="labeled trailers")}
    grids = {"clip": np.arange(0.40, 0.90, 0.005), "sscd": np.arange(0.0, 0.9, 0.005), "blend": np.arange(0.4, 1.8, 0.005)}
    report = {}
    for how, ts in grids.items():
        rows = [dict(threshold=float(t), **evaluate(films, gold, how, t)) for t in ts]
        top1 = evaluate(films, gold, how, -np.inf)  # no threshold: is the top pick the labeled shot?
        best = max(rows, key=lambda r: r["f1"])
        report[how] = dict(best_f1=best, recall_no_threshold=top1["recall"])
        print(f"{how:6s} best F1 {best['f1']:.3f} (P {best['precision']:.3f}, R {best['recall']:.3f} at {best['threshold']:.3f})"
              f"  top pick right for {top1['recall']:.1%} of labeled sources")
    (FEATURES / "rerank_eval.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
