"""CLIP embeddings for film and trailer keyframes (3 per shot), plus concept-prompt text embeddings.

The backbone is picked with TRAILERCUT_CLIP (b32 default, l14, h14); each gets its own cache dir.

Outputs (gitignored, regenerate with this script)
  data/cache/clip/{tag}/film/{imdb_id}.npy      float16 [n_shots, 3, d], L2-normalized
  data/cache/clip/{tag}/trailer/{imdb_id}.npy   float16 [n_shots, 3, d]

  TRAILERCUT_CLIP=h14 python -m trailercut.embed film|trailer [--ids ID ...] [--limit N] [--bs 256] [--workers 6]
"""
import argparse

import numpy as np
import open_clip
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import movienet_io as mn
from .paths import CACHE, CLIP, CLIP_TAG, FEATURES

# b32 is what fits on a laptop (~257 img/s on M1 Max MPS); h14 matches Argaw et al. and needs a GPU
MODELS = {
    "b32": ("ViT-B-32", "laion2b_s34b_b79k"),
    "l14": ("ViT-L-14", "laion2b_s32b_b82k"),
    "h14": ("ViT-H-14", "laion2b_s32b_b79k"),
}
MODEL, PRETRAINED = MODELS[CLIP_TAG]
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
TRAILER_FRAMES = CACHE / "trailer_frames"


def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    return model.to(DEVICE).half().eval(), preprocess, open_clip.get_tokenizer(MODEL)


def bar_box(paths, thresh=20, min_keep=0.3):
    """Letterbox/pillarbox crop for one trailer. Trailers on YouTube are often boxed while MovieNet
    keyframes are not, and the bars pull CLIP similarity down. Bars are found from the per-pixel max
    over all of the trailer's frames: real bars stay black in every frame, dark scenes don't."""
    g = np.max([np.asarray(mn.open_image(p).convert("L")) for p in paths], axis=0).astype(np.float32)
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
        im = mn.open_image(self.paths[i]).convert("RGB")
        if self.boxes is not None and self.boxes[i] is not None:
            im = im.crop(self.boxes[i])
        return self.preprocess(im)


@torch.no_grad()
def encode_stream(model, preprocess, jobs, out, bs=256, workers=6, crop=False):
    """Run every job's frames through one DataLoader (worker startup is slow on macOS) and save per film."""
    paths = [p for _, ps, _ in jobs for p in ps]
    boxes = [b for _, ps, _ in jobs for b in [bar_box(ps)] * len(ps)] if crop else None
    dl = DataLoader(Frames(paths, preprocess, boxes), batch_size=bs, num_workers=workers)
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
        n = len(mn.shots(imdb_id))
        if mn.keyframe_count(imdb_id) == 3 * n:
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
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ids", nargs="*", default=None, help="only these films (default: every one not yet embedded)")
    a = ap.parse_args()
    out = CLIP / a.kind
    out.mkdir(parents=True, exist_ok=True)
    model, preprocess, _ = load_model()
    jobs = [j for j in (film_jobs() if a.kind == "film" else trailer_jobs()) if not (out / f"{j[0]}.npy").exists() and (a.ids is None or j[0] in a.ids)]
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{MODEL} on {DEVICE}: {len(jobs)} {a.kind}s to embed")
    encode_stream(model, preprocess, jobs, out, a.bs, a.workers, crop=a.kind == "trailer")


if __name__ == "__main__":
    main()
