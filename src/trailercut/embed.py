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
from PIL import Image

from . import movienet_io as mn
from .paths import CACHE, CLIP, CLIP_TAG, FEATURES, MOVIENET
import struct
import zlib
from io import BytesIO

from .movienet import RangeFile, FRAME_PARTS, load_index

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


# Cached remote MovieNet index. This is a small lookup structure
# compared with downloading the actual keyframes.
_REMOTE_INDEX = None


def remote_index():
    global _REMOTE_INDEX

    if _REMOTE_INDEX is None:
        cache = MOVIENET / "frames_index.pkl"
        _REMOTE_INDEX = load_index(str(cache))

    return _REMOTE_INDEX


def remote_frame(imdb_id, shot_idx, keyframe_idx):
    """Read one MovieNet keyframe directly from the remote split ZIP."""

    filename = mn.keyframe_name(shot_idx, keyframe_idx)

    entries = remote_index().get(imdb_id, [])

    entry = next(
        (
            e for e in entries
            if e[0].rsplit("/", 1)[-1] == filename
        ),
        None
    )

    if entry is None:
        raise FileNotFoundError(
            f"Remote MovieNet frame missing: {imdb_id}/{filename}"
        )

    name, offset, compressed_size, method = entry

    rf = RangeFile(FRAME_PARTS)
    rf.seek(offset)

    header = rf.read(30)

    if len(header) != 30:
        raise IOError("Incomplete remote ZIP header")

    fields = struct.unpack("<IHHHHHIIIHH", header)

    if fields[0] != 0x04034B50:
        raise IOError("Invalid remote ZIP local header")

    filename_length = fields[9]
    extra_length = fields[10]

    rf.read(filename_length + extra_length)

    compressed = rf.read(compressed_size)

    if len(compressed) != compressed_size:
        raise IOError("Incomplete remote frame data")

    if method == 0:
        data = compressed
    elif method == 8:
        data = zlib.decompress(compressed, -15)
    else:
        raise IOError(f"Unsupported ZIP method: {method}")

    return Image.open(BytesIO(data)).convert("RGB")



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
        self.paths = paths
        self.preprocess = preprocess
        self.boxes = boxes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        ref = self.paths[i]

        if isinstance(ref, tuple):
            # Remote MovieNet frame:
            # (imdb_id, shot_idx, keyframe_idx)
            imdb_id, shot_idx, keyframe_idx = ref
            im = remote_frame(
                imdb_id, shot_idx, keyframe_idx
            )
        else:
            # Existing local JPEG or ZIP workflow.
            im = mn.open_image(ref).convert("RGB")

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


def film_jobs(ids=None):
    """Yield film embedding jobs, using local frames when available
    and remote MovieNet frames otherwise."""

    wanted = set(ids) if ids else None

    for imdb_id in mn.core_ids():
        if wanted is not None and imdb_id not in wanted:
            continue

        n = len(mn.shots(imdb_id))

        # Prefer existing complete local keyframes.
        if mn.keyframe_count(imdb_id) == 3 * n:
            paths = [
                p
                for s in range(n)
                for p in mn.keyframe_paths(imdb_id, s)
            ]

        else:
            # Fall back to remote MovieNet ZIP access.
            entries = remote_index().get(imdb_id, [])

            available = {
                e[0].rsplit("/", 1)[-1]
                for e in entries
            }

            expected = [
                mn.keyframe_name(s, k)
                for s in range(n)
                for k in range(3)
            ]

            if not all(name in available for name in expected):
                print(f"Skipping {imdb_id}: incomplete MovieNet keyframes")
                continue

            paths = [
                (imdb_id, s, k)
                for s in range(n)
                for k in range(3)
            ]

            print(f"{imdb_id}: using remote MovieNet frames")

        yield imdb_id, paths, n


def trailer_jobs(ids=None):
    wanted = set(ids) if ids else None

    for csv in sorted((FEATURES / "trailer_shots").glob("*.csv")):
        imdb_id = csv.stem

        if wanted is not None and imdb_id not in wanted:
            continue

        n = len(pd.read_csv(csv))
        d = TRAILER_FRAMES / imdb_id

        paths = [
            d / f"shot_{s:04d}_img_{i}.jpg"
            for s in range(n)
            for i in range(3)
        ]

        yield imdb_id, paths, n


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
    source = (
        film_jobs(a.ids)
        if a.kind == "film"
        else trailer_jobs(a.ids)
    )

    jobs = [
        j for j in source
        if not (out / f"{j[0]}.npy").exists()
    ]
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{MODEL} on {DEVICE}: {len(jobs)} {a.kind}s to embed")
    # Remote HTTP reads run in the main process to avoid
    # sharing requests sessions across DataLoader workers.
    has_remote = any(
        isinstance(p, tuple)
        for _, paths, _ in jobs
        for p in paths
    )

    workers = 0 if has_remote else a.workers
    encode_stream(
        model,
        preprocess,
        jobs,
        out,
        a.bs,
        workers,
        crop=a.kind == "trailer"
    )


if __name__ == "__main__":
    main()
