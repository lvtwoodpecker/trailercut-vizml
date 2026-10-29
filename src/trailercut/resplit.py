"""Split trailer shots that likely hide a cut TransNetV2 missed.

A shot is suspect when its 3 keyframes disagree more than 99% of MovieNet's film shots ever do
(align_diagnostics: lowest keyframe cosine below the film floor). For each trailer with suspect shots,
TransNetV2's per-frame cut probabilities are recomputed and each suspect shot is split once, at its
most likely frame, if that probability is at least MIN_P (the default cutoff is 0.5) and both halves
are at least MIN_FRAMES long. The trailer's shot table and keyframes are rewritten and its CLIP
embedding removed, so `embed trailer` and everything after picks up the new shots.

Hand-labeled trailers are skipped (their labels index the current shots).
Writes data/features/resplit_log.csv (committed), one row per split, plus a shot_idx -1 row for a
trailer checked with nothing to split; saves per trailer, and a rerun skips trailers already logged.

  python -m trailercut.resplit
"""
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transnetv2_pytorch import TransNetV2

from .align import load
from .align_diagnostics import consistency
from .paths import CACHE, CLIP, FEATURES, LABELS, TRAILERS
from .trailer_prep import FRAMES_DIR, SHOTS_DIR, extract_keyframes, probe, read_frames

LOG = FEATURES / "resplit_log.csv"
MIN_P, MIN_FRAMES = 0.1, 6


def floor():
    return json.loads((FEATURES / "alignment_diagnostics.json").read_text())["cuts"]["film_floor_1pct"]


def main():
    labeled = set(pd.read_csv(LABELS / "label_trailers.csv", dtype=str).imdb_id) if (LABELS / "label_trailers.csv").exists() else set()
    done = set(pd.read_csv(LOG, dtype=str).imdb_id) if LOG.exists() else set()
    f0 = floor()
    model = TransNetV2(device="cpu").eval()
    torch.set_num_threads(8)
    todo = []
    for p in sorted(SHOTS_DIR.glob("*.csv")):
        i = p.stem
        shutil.rmtree(FRAMES_DIR / f"{i}.resplit", ignore_errors=True)  # leftover from an interrupted run
        if i in labeled or i in done or not (CLIP / "trailer" / f"{i}.npy").exists():
            continue
        c = consistency(load("trailer", i).numpy())
        if (c < f0).any():
            todo.append((i, np.flatnonzero(c < f0)))
    print(f"{len(todo)} trailers with suspect shots, {sum(len(s) for _, s in todo)} shots")
    for i, suspect in tqdm(todo):
        path = TRAILERS / f"{i}.mp4"
        fps, _, _ = probe(path)
        frames = torch.from_numpy(read_frames(path, 48, 27).copy())
        single, _ = model.predict_frames(frames, quiet=True)
        prob = single.numpy().ravel()
        sh = pd.read_csv(SHOTS_DIR / f"{i}.csv")
        cuts, rows = [], []
        for k in suspect:
            a, b = int(sh.frame_start[k]), int(sh.frame_end[k])
            inner = np.arange(a + MIN_FRAMES, b - MIN_FRAMES + 1)
            if len(inner) == 0:
                continue
            f = int(inner[prob[inner].argmax()])
            if prob[f] >= MIN_P:
                cuts.append(f)
                rows.append(dict(imdb_id=i, shot_idx=int(k), split_frame=f, prob=round(float(prob[f]), 3)))
        if cuts:
            spans = []
            for r in sh.itertuples():
                c = [f for f in cuts if r.frame_start < f <= r.frame_end]
                spans.append((r.frame_start, c[0] - 1) if c else (r.frame_start, r.frame_end))
                if c:
                    spans.append((c[0], r.frame_end))
            n = len(frames)
            new = pd.DataFrame(spans, columns=["frame_start", "frame_end"])
            new.insert(0, "shot_idx", np.arange(len(new)))
            new["t_start"] = new.frame_start / fps
            new["t_end"] = (new.frame_end + 1) / fps
            new["shot_len"] = new.t_end - new.t_start
            new["trailer_pos"] = (new.t_start + new.t_end) / 2 / (n / fps)
            # build the new keyframes and shot table next to the old ones, then swap them in, so an
            # interrupted run never leaves keyframes that don't match the shot table
            tmp = FRAMES_DIR / f"{i}.resplit"
            shutil.rmtree(tmp, ignore_errors=True)
            extract_keyframes(path, new, tmp, fps)
            new.round(4).to_csv(tmp / "shots.csv", index=False)
            shutil.rmtree(FRAMES_DIR / i)
            os.replace(tmp, FRAMES_DIR / i)
            os.replace(FRAMES_DIR / i / "shots.csv", SHOTS_DIR / f"{i}.csv")
            (CLIP / "trailer" / f"{i}.npy").unlink(missing_ok=True)
        # log even when nothing split, so a rerun skips this trailer
        out = pd.DataFrame(rows or [dict(imdb_id=i, shot_idx=-1, split_frame=-1, prob=np.nan)])
        out.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    log = pd.read_csv(LOG)
    print(f"{int((log.shot_idx >= 0).sum())} shots split in {log[log.shot_idx >= 0].imdb_id.nunique()} trailers")


if __name__ == "__main__":
    main()
