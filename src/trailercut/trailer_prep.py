"""Per trailer: TransNetV2 shot boundaries, 3 keyframes per shot (MovieNet style), mono audio.

Outputs
  data/features/trailer_shots/{imdb_id}.csv   committed (small)
  data/cache/trailer_frames/{imdb_id}/shot_XXXX_img_{0,1,2}.jpg
  data/cache/trailer_audio/{imdb_id}.wav
  data/cache/transnet/{imdb_id}_{video bytes}.npy   TransNetV2's per-frame cut probability (float32), so
      re-splitting or re-tuning cuts never runs the network again. Keyed on the video's size, so a swapped
      trailer gets its own entry and a restored one finds its old entry.

  python -m trailercut.trailer_prep [--ids ...] [--limit N] [--workers 4]
  python -m trailercut.trailer_prep --probs-only    fill the cut-probability cache for trailers done before it existed
On a CUDA machine TransNetV2 runs on the GPU in a single process (seconds per trailer).
"""
import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transnetv2_pytorch import TransNetV2

from .paths import CACHE, FEATURES, TRAILERS

SHOTS_DIR = FEATURES / "trailer_shots"
FRAMES_DIR = CACHE / "trailer_frames"
AUDIO_DIR = CACHE / "trailer_audio"
PROBS_DIR = CACHE / "transnet"
KEYFRAME_FRACS = (0.25, 0.5, 0.75)

if shutil.which("ffmpeg") is None:
    # clusters often lack ffmpeg; static-ffmpeg ships ffmpeg + ffprobe binaries
    import static_ffmpeg
    static_ffmpeg.add_paths()


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate,width,height", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return num / den, s["width"], s["height"]


def read_frames(path, w, h):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "pipe:"], capture_output=True, check=True)
    return np.frombuffer(out.stdout, np.uint8).reshape(-1, h, w, 3)


def cut_probs(model, path):
    """TransNetV2's cut probability for every frame of a trailer video, from the cache when this exact
    video (same size in bytes) was run before."""
    out = PROBS_DIR / f"{path.stem}_{path.stat().st_size}.npy"
    if out.exists():
        return np.load(out)
    frames = torch.from_numpy(read_frames(path, 48, 27).copy()).to(model.device)
    single, _ = model.predict_frames(frames, quiet=True)
    p = single.cpu().numpy().ravel().astype(np.float32)
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npy")  # write aside, then rename: parallel workers never see half a file
    np.save(tmp, p)
    os.replace(tmp, out)
    return p


def detect_shots(model, path):
    p = cut_probs(model, path)
    return TransNetV2.predictions_to_scenes(p), len(p)


def extract_keyframes(path, shots_df, outdir, fps):
    outdir.mkdir(parents=True, exist_ok=True)
    w, h = 320, 180
    frames = read_frames(path, w, h)
    from PIL import Image
    for r in shots_df.itertuples():
        for i, frac in enumerate(KEYFRAME_FRACS):
            f = int(r.frame_start + frac * (r.frame_end - r.frame_start))
            Image.fromarray(frames[min(f, len(frames) - 1)]).save(outdir / f"shot_{r.shot_idx:04d}_img_{i}.jpg", quality=90)


def extract_audio(path, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", "22050", str(out)], check=True)


def process(model, path):
    imdb_id = path.stem
    csv_out = SHOTS_DIR / f"{imdb_id}.csv"
    if csv_out.exists() and (AUDIO_DIR / f"{imdb_id}.wav").exists():
        return
    fps, _, _ = probe(path)
    scenes, n = detect_shots(model, path)
    df = pd.DataFrame(scenes, columns=["frame_start", "frame_end"])
    df.insert(0, "shot_idx", np.arange(len(df)))
    df["t_start"] = df.frame_start / fps
    df["t_end"] = (df.frame_end + 1) / fps
    df["shot_len"] = df.t_end - df.t_start
    df["trailer_pos"] = (df.t_start + df.t_end) / 2 / (n / fps)
    extract_keyframes(path, df, FRAMES_DIR / imdb_id, fps)
    extract_audio(path, AUDIO_DIR / f"{imdb_id}.wav")
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(csv_out, index=False)


_model = None
_probs_only = False


def _init(probs_only):
    global _probs_only
    _probs_only = probs_only


def _worker(path):
    global _model
    if _model is None:
        torch.set_num_threads(2)
        _model = TransNetV2(device="cpu").eval()
    try:
        cut_probs(_model, path) if _probs_only else process(_model, path)
    except Exception as e:
        return f"{path.stem} failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ids", nargs="*", default=None, help="only these films (default: every trailer not yet done)")
    ap.add_argument("--probs-only", action="store_true", help="only fill the cut-probability cache")
    a = ap.parse_args()
    # yt-dlp leaves tt123.f605.mp4 fragments mid-merge; only take finished files
    vids = sorted(p for p in TRAILERS.glob("tt*.mp4") if p.stem.isalnum() and (a.ids is None or p.stem in a.ids))
    if a.limit:
        vids = vids[: a.limit]
    if torch.cuda.is_available():
        model = TransNetV2(device="cuda").eval()
        for p in tqdm(vids):
            try:
                cut_probs(model, p) if a.probs_only else process(model, p)
            except Exception as e:
                print(p.stem, "failed:", e)
        return
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.probs_only,)) as ex:
        for err in tqdm(ex.map(_worker, vids), total=len(vids)):
            if err:
                print(err)


if __name__ == "__main__":
    main()
