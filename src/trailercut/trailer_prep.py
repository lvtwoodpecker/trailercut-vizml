
"""Per-trailer TransNetV2 preparation.

Outputs:
  data/features/trailer_shots/{imdb_id}.csv
  data/cache/trailer_frames/{imdb_id}/shot_XXXX_img_{0,1,2}.jpg
#   data/cache/trailer_audio/{imdb_id}.wav
  data/cache/transnet/{imdb_id}_{video_bytes}.npy

Examples:
  python -m trailercut.trailer_prep --ids tt0032138
  python -m trailercut.trailer_prep --ids tt0032138 --workers 1
  python -m trailercut.trailer_prep --limit 10
  python -m trailercut.trailer_prep --probs-only
"""
import argparse
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transnetv2_pytorch import TransNetV2

from .paths import CACHE, FEATURES, TRAILERS

SHOTS_DIR = FEATURES / "trailer_shots"
FRAMES_DIR = CACHE / "trailer_frames"
# AUDIO_DIR = CACHE / "trailer_audio"
PROBS_DIR = CACHE / "transnet"
LOCKS_DIR = CACHE / "trailer_locks"

KEYFRAME_FRACS = (0.25, 0.5, 0.75)

if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    import static_ffmpeg
    static_ffmpeg.add_paths()


def probe(path):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,width,height",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(out.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found: {path}")

    s = streams[0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    if den == 0:
        raise RuntimeError(f"Invalid frame rate: {path}")

    return num / den, s["width"], s["height"]


def read_frames(path, w, h):
    out = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "pipe:",
        ],
        capture_output=True, check=True,
    )
    frame_bytes = w * h * 3
    if not out.stdout or len(out.stdout) % frame_bytes:
        raise RuntimeError(f"Invalid or incomplete decoded frames: {path}")

    return np.frombuffer(out.stdout, np.uint8).reshape(-1, h, w, 3)


def validate_probs(path):
    try:
        p = np.load(path, allow_pickle=False)
        return (
            p.ndim == 1
            and len(p) > 0
            and np.issubdtype(p.dtype, np.number)
            and np.isfinite(p).all()
        )
    except Exception:
        return False


def cut_probs(model, path):
    """Load or generate cached per-frame TransNetV2 probabilities."""
    out = PROBS_DIR / f"{path.stem}_{path.stat().st_size}.npy"

    if out.exists() and validate_probs(out):
        return np.load(out, allow_pickle=False)

    frames = torch.from_numpy(read_frames(path, 48, 27).copy())
    frames = frames.to(model.device)

    single, _ = model.predict_frames(frames, quiet=True)
    p = single.cpu().numpy().ravel().astype(np.float32)

    if not len(p) or not np.isfinite(p).all():
        raise RuntimeError(f"TransNetV2 returned invalid probabilities: {path}")

    PROBS_DIR.mkdir(parents=True, exist_ok=True)

    # Unique temporary filename prevents concurrent cache writers colliding.
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp.npy")
    try:
        np.save(tmp, p)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)

    return p


def detect_shots(model, path):
    p = cut_probs(model, path)
    return TransNetV2.predictions_to_scenes(p), len(p)


def extract_keyframes(path, shots_df, outdir, fps):
    outdir.mkdir(parents=True, exist_ok=True)

    frames = read_frames(path, 320, 180)
    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded: {path}")

    from PIL import Image

    for r in shots_df.itertuples():
        for i, frac in enumerate(KEYFRAME_FRACS):
            f = int(r.frame_start + frac * (r.frame_end - r.frame_start))
            f = max(0, min(f, len(frames) - 1))

            target = outdir / f"shot_{r.shot_idx:04d}_img_{i}.jpg"
            tmp = target.with_name(
                f".{target.stem}.{os.getpid()}.tmp.jpg"
            )

            try:
                Image.fromarray(frames[f]).save(tmp, format="JPEG", quality=90)
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)


def extract_audio(path, out):
    out.parent.mkdir(parents=True, exist_ok=True)

    # Keep .wav as the final extension so FFmpeg selects the correct format.
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp.wav")

    try:
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(path),
                "-vn", "-ac", "1", "-ar", "22050",
                str(tmp),
            ],
            check=True,
        )

        if not validate_audio(tmp):
            raise RuntimeError(f"Extracted audio failed validation: {tmp}")

        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def validate_audio(path):
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        return bool(
            streams
            and streams[0].get("sample_rate") == "22050"
            and streams[0].get("channels") == 1
        )
    except Exception:
        return False


def expected_frame_paths(imdb_id, shots_df):
    outdir = FRAMES_DIR / imdb_id
    return [
        outdir / f"shot_{int(r.shot_idx):04d}_img_{i}.jpg"
        for r in shots_df.itertuples()
        for i in range(3)
    ]


def is_prepared(imdb_id, video_path=None):
    """Return True only when all expected preparation outputs are valid."""
    csv_path = SHOTS_DIR / f"{imdb_id}.csv"
    # audio_path = AUDIO_DIR / f"{imdb_id}.wav"

    if not csv_path.is_file():
        return False

    try:
        df = pd.read_csv(csv_path)

        required = {
            "shot_idx", "frame_start", "frame_end",
            "t_start", "t_end", "shot_len", "trailer_pos",
        }

        if not required.issubset(df.columns):
            return False

        if df.empty or df[list(required)].isna().any().any():
            return False

        if not (df.shot_idx.astype(int).to_numpy() == np.arange(len(df))).all():
            return False

        if (df.frame_end < df.frame_start).any():
            return False

        frames = expected_frame_paths(imdb_id, df)

        for frame in frames:
            if not frame.is_file() or frame.stat().st_size == 0:
                return False

        if video_path is not None:
            prob_path = PROBS_DIR / (
                f"{imdb_id}_{video_path.stat().st_size}.npy"
            )
            if not validate_probs(prob_path):
                return False

        return True

    except Exception:
        return False


@contextmanager
def trailer_lock(imdb_id):
    """Atomic directory creation prevents two processes preparing one ID."""
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    lock = LOCKS_DIR / imdb_id

    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(
            f"{imdb_id} is already being prepared by another process "
            f"(lock: {lock})"
        )

    try:
        yield
    finally:
        lock.rmdir()


def process(model, path):
    imdb_id = path.stem

    with trailer_lock(imdb_id):
        if is_prepared(imdb_id, path):
            print(f"{imdb_id} READY (all outputs validated)", flush=True)
            return "READY"

        print(f"{imdb_id} PROCESSING", flush=True)

        fps, _, _ = probe(path)
        scenes, n = detect_shots(model, path)

        if n <= 0:
            raise RuntimeError("Video contains no decoded frames")

        df = pd.DataFrame(scenes, columns=["frame_start", "frame_end"])
        df.insert(0, "shot_idx", np.arange(len(df)))
        df["t_start"] = df.frame_start / fps
        df["t_end"] = (df.frame_end + 1) / fps
        df["shot_len"] = df.t_end - df.t_start
        df["trailer_pos"] = (
            (df.t_start + df.t_end) / 2 / (n / fps)
        )

        extract_keyframes(path, df, FRAMES_DIR / imdb_id, fps)
        # extract_audio(path, AUDIO_DIR / f"{imdb_id}.wav")

        SHOTS_DIR.mkdir(parents=True, exist_ok=True)

        csv_out = SHOTS_DIR / f"{imdb_id}.csv"
        tmp = csv_out.with_name(
            f".{csv_out.stem}.{os.getpid()}.tmp.csv"
        )

        try:
            df.round(4).to_csv(tmp, index=False)
            os.replace(tmp, csv_out)
        finally:
            tmp.unlink(missing_ok=True)

        if not is_prepared(imdb_id, path):
            raise RuntimeError("Final output validation failed")

        print(
            f"{imdb_id} COMPLETED ({len(df)} shots, "
            f"{len(df) * 3} keyframes)",
            flush=True,
        )
        return "COMPLETED"


def prepare_probs(model, path):
    with trailer_lock(path.stem):
        cut_probs(model, path)
    return "COMPLETED"


_model = None
_probs_only = False


def _init(probs_only):
    global _probs_only
    _probs_only = probs_only


def _worker(path):
    global _model

    try:
        if _model is None:
            torch.set_num_threads(2)
            _model = TransNetV2(device="cpu").eval()

        if _probs_only:
            return prepare_probs(_model, path)

        return process(_model, path)

    except Exception as e:
        return f"{path.stem} FAILED: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--ids", nargs="+", default=None,
        help="only these IMDb IDs (default: all downloaded trailers)",
    )
    ap.add_argument(
        "--probs-only", action="store_true",
        help="only fill the cut-probability cache",
    )
    a = ap.parse_args()

    if a.limit < 0 or a.workers < 1:
        ap.error("--limit must be >= 0 and --workers must be >= 1")

    available = {
        p.stem: p
        for p in TRAILERS.glob("tt*.mp4")
        if p.stem.isalnum()
    }

    if a.ids is not None:
        missing = sorted(set(a.ids) - set(available))
        if missing:
            ap.error(
                "Downloaded video not found for: "
                + ", ".join(missing)
                + ". Download it first with trailercut.trailers."
            )
        vids = [available[i] for i in sorted(set(a.ids))]
    else:
        vids = [available[i] for i in sorted(available)]

    if a.limit:
        vids = vids[:a.limit]

    if not vids:
        print("No matching downloaded trailers found.")
        return

    print(f"Found {len(vids)} trailer(s).", flush=True)

    results = []

    if torch.cuda.is_available():
        model = TransNetV2(device="cuda").eval()

        for path in tqdm(vids):
            try:
                if a.probs_only:
                    result = prepare_probs(model, path)
                else:
                    result = process(model, path)
            except Exception as e:
                result = f"{path.stem} FAILED: {e}"

            results.append(result)
    else:
        with ProcessPoolExecutor(
            max_workers=a.workers,
            initializer=_init,
            initargs=(a.probs_only,),
        ) as executor:
            for result in tqdm(
                executor.map(_worker, vids), total=len(vids)
            ):
                results.append(result)

    failures = [r for r in results if isinstance(r, str) and "FAILED:" in r]
    completed = sum(r == "COMPLETED" for r in results)
    ready = sum(r == "READY" for r in results)

    print(
        f"\nSummary: {completed} completed, "
        f"{ready} already ready, {len(failures)} failed.",
        flush=True,
    )

    for failure in failures:
        print(failure)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
