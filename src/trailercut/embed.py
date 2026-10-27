"""CLIP ViT-B/32 embeddings for film and trailer keyframes (3 per shot), plus concept-prompt text embeddings.

Outputs (gitignored, regenerate with this script)
  data/cache/clip/film/{imdb_id}.npy      float16 [n_shots, 3, 512], L2-normalized
  data/cache/clip/trailer/{imdb_id}.npy   float16 [n_shots, 3, 512]

  python -m trailercut.embed film|trailer [--limit N]
"""
import argparse

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import movienet_io as mn
from .paths import CACHE, FEATURES, FRAMES

# B/32 over B/16: ~3.5x faster on MPS (257 vs 72 img/s) and plenty for near-duplicate matching
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = CACHE / "clip"
TRAILER_FRAMES = CACHE / "trailer_frames"


def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    return model.to(DEVICE).half().eval(), preprocess, open_clip.get_tokenizer(MODEL)


def bar_box(paths, thresh=20, min_keep=0.3):
    """Letterbox/pillarbox crop for one trailer. Trailers on YouTube are often boxed while MovieNet
    keyframes are not, and the bars pull CLIP similarity down. Bars are found from the per-pixel max
    over all of the trailer's frames: real bars stay black in every frame, dark scenes don't."""
    g = np.max([np.asarray(Image.open(p).convert("L")) for p in paths], axis=0).astype(np.float32)
    rows, cols = np.where(g.mean(1) > thresh)[0], np.where(g.mean(0) > thresh)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    box = (int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1))
    return box if (box[2] - box[0]) * (box[3] - box[1]) >= min_keep * g.size else None


class Frames(Dataset):
    def __init__(self, paths, preprocess, boxes=None):
        self.paths, self.preprocess, self.boxes = paths, preprocess, boxes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        if self.boxes is not None and self.boxes[i] is not None:
            im = im.crop(self.boxes[i])
        return self.preprocess(im)


@torch.no_grad()
def encode_stream(model, preprocess, jobs, out, bs=256, crop=False):
    """Run every job's frames through one DataLoader (worker startup is slow on macOS) and save per film."""
    paths = [p for _, ps, _ in jobs for p in ps]
    boxes = [b for _, ps, _ in jobs for b in [bar_box(ps)] * len(ps)] if crop else None
    dl = DataLoader(Frames(paths, preprocess, boxes), batch_size=bs, num_workers=6)
    buf, j = [], 0
    for x in tqdm(dl, total=len(dl)):
        e = model.encode_image(x.to(DEVICE).half())
        buf.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().half())
        cat = torch.cat(buf)
        while j < len(jobs) and len(cat) >= 3 * jobs[j][2]:
            imdb_id, _, n = jobs[j]
            np.save(out / f"{imdb_id}.npy", cat[: 3 * n].numpy().reshape(n, 3, -1))
            cat, j = cat[3 * n:], j + 1
        buf = [cat]


@torch.no_grad()
def encode_text(model, tokenizer, prompts):
    e = model.encode_text(tokenizer(prompts).to(DEVICE))
    return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()


def film_jobs():
    for imdb_id in mn.core_ids():
        d = FRAMES / imdb_id
        n = len(mn.shots(imdb_id))
        if d.is_dir() and len(list(d.glob("*.jpg"))) == 3 * n:
            yield imdb_id, [p for s in range(n) for p in mn.keyframe_paths(imdb_id, s)], n


def trailer_jobs():
    for csv in sorted((FEATURES / "trailer_shots").glob("*.csv")):
        imdb_id, n = csv.stem, len(pd.read_csv(csv))
        d = TRAILER_FRAMES / imdb_id
        yield imdb_id, [d / f"shot_{s:04d}_img_{i}.jpg" for s in range(n) for i in range(3)], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["film", "trailer"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = OUT / a.kind
    out.mkdir(parents=True, exist_ok=True)
    model, preprocess, _ = load_model()
    jobs = [j for j in (film_jobs() if a.kind == "film" else trailer_jobs()) if not (out / f"{j[0]}.npy").exists()]
    if a.limit:
        jobs = jobs[: a.limit]
    encode_stream(model, preprocess, jobs, out, crop=a.kind == "trailer")


if __name__ == "__main__":
    main()
