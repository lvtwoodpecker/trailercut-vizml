
"""TrailerCut viewer API: on-demand preparation and frame serving."""

import struct
import zlib
import threading
import subprocess
import sys

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response,RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageStat

from .movienet_io import keyframe_paths, open_image, keyframe_name
from .paths import MOVIENET, TRAILERS, ROOT, FRAMES, CACHE
from .movienet import RangeFile, FRAME_PARTS, load_index, fetch_film
from .trailer_prep import probe

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

remote_lock = threading.Lock()

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAILER_FRACTIONS = (0.25, 0.50, 0.75)

OUT = ROOT / "viz" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Job state and per-ID synchronization.
jobs = {}
locks = {}
jobs_lock = threading.Lock()

def set_job(imdb_id, status):
    with jobs_lock:
        jobs[imdb_id] = status


def valid_id(imdb_id):
    return (
        imdb_id.startswith("tt")
        and imdb_id[2:].isdigit()
    )


def get_lock(imdb_id):
    with jobs_lock:
        if imdb_id not in locks:
            locks[imdb_id] = threading.Lock()
        return locks[imdb_id]


def viewer_outputs_exist(imdb_id):
    return (OUT / f"{imdb_id}.json").is_file()


# --------------------------------------------------
# On-demand preparation
# --------------------------------------------------


def prepare_trailer(imdb_id):
    with get_lock(imdb_id):
        try:
            if viewer_outputs_exist(imdb_id):
                set_job(imdb_id, "ready")
                return True

            set_job(imdb_id, "Checking film data...")

            # 1. MovieNet keyframes
            set_job(imdb_id, "Checking / downloading film keyframes...")
            ensure_film_keyframes(imdb_id)

            # 2. Trailer download
            set_job(imdb_id, "Downloading trailer...")
            subprocess.run(
                [
                    sys.executable, "-m", "trailercut.trailers",
                    "--ids", imdb_id,
                ],
                cwd=ROOT,
                check=True,
            )

            # 3. Trailer processing
            set_job(imdb_id, "Processing trailer and extracting features...")
            subprocess.run(
                [
                    sys.executable, "-m", "trailercut.trailer_prep",
                    "--ids", imdb_id,
                ],
                cwd=ROOT,
                check=True,
            )

            # 3a. Transcribe trailer audio -> cache/trailer_asr/{imdb_id}.json
            # set_job(imdb_id, "Transcribing trailer speech...")
            # subprocess.run(
            #     [
            #         sys.executable, "-m", "trailercut.trailer_asr",
            #         "--ids", imdb_id,
            #     ],
            #     cwd=ROOT,
            #     check=False,
            # )

            # # 3b. Align transcript to subtitles -> features/dialogue_alignment/{imdb_id}.csv
            # set_job(imdb_id, "Aligning trailer dialogue...")
            # subprocess.run(
            #     [
            #         sys.executable, "-m", "trailercut.dialogue_align",
            #         "--ids", imdb_id,
            #     ],
            #     cwd=ROOT,
            #     check=False,
            # )

            # 4. Viewer export and colors
            set_job(imdb_id, "Generating film colors and preparing visualization...")
            subprocess.run(
                [
                    sys.executable, "-m", "trailercut.viewer_export",
                    "--ids", imdb_id,
                ],
                cwd=ROOT,
                check=True,
            )

            if not viewer_outputs_exist(imdb_id):
                raise RuntimeError("Viewer JSON was not generated")

            # trailer video and audio stay: the pipeline's audio features, ASR and reruns read them

            set_job(imdb_id, "ready")
            print(f"Preparation complete: {imdb_id}", flush=True)
            return True

        except Exception as e:
            set_job(imdb_id, f"error: {e}")
            print(f"Preparation failed: {e}", flush=True)
            return False



@app.get("/api/trailer/{imdb_id}")
def trailer(imdb_id: str):
    if not valid_id(imdb_id):
        raise HTTPException(400, "Invalid IMDb ID")

    if viewer_outputs_exist(imdb_id):
        return {"status": "ready"}

    with jobs_lock:
        status = jobs.get(imdb_id)

        if status is None or status.startswith("error:"):
            jobs[imdb_id] = "Starting preparation..."

            threading.Thread(
                target=prepare_trailer,
                args=(imdb_id,),
                daemon=True,
            ).start()

        return {"status": jobs[imdb_id]}


@app.get("/api/trailer/{imdb_id}/status")
def trailer_status(imdb_id: str):
    """Check the status of an on-demand preparation job."""

    if not valid_id(imdb_id):
        raise HTTPException(400, "Invalid IMDb ID")

    if viewer_outputs_exist(imdb_id):
        with jobs_lock:
            jobs[imdb_id] = "ready"

        return {"status": "ready"}

    with jobs_lock:
        return {
            "status": jobs.get(imdb_id, "pending")
        }


# --------------------------------------------------
# MovieNet index
# --------------------------------------------------

@lru_cache(maxsize=1)
def remote_index():
    path = MOVIENET / "frames_index.pkl"
    return load_index(str(path))

def ensure_film_keyframes(imdb_id):
    frames_dir = FRAMES
    film_dir = frames_dir / imdb_id
    film_zip = frames_dir / f"{imdb_id}.zip"

    print(f"Checking: {film_zip}", flush=True)

    if film_zip.is_file() and film_zip.stat().st_size > 0:
        print("ZIP already exists", flush=True)
        return

    if film_dir.is_dir() and any(film_dir.glob("*.jpg")):
        print("Film frames directory already exists", flush=True)
        return

    print(f"ZIP MISSING. Fetching {imdb_id}", flush=True)

    index_path = MOVIENET / "frames_index.pkl"
    idx = load_index(str(index_path))

    print(f"Remote index contains ID: {imdb_id in idx}", flush=True)

    entries = idx.get(imdb_id)

    if not entries:
        raise FileNotFoundError(
            f"No MovieNet keyframes available for {imdb_id}"
        )

    print(f"Found {len(entries)} keyframes. Downloading...", flush=True)

    with remote_lock:
        fetch_film(
            RangeFile(FRAME_PARTS),
            entries,
            str(film_dir),
            pack=True
        )

    if not film_zip.is_file():
        raise RuntimeError(f"ZIP was not created: {film_zip}")

    print(
        f"DOWNLOAD COMPLETE: {film_zip} ({film_zip.stat().st_size:,} bytes)",
        flush=True
    )


# --------------------------------------------------
# Image validation
# --------------------------------------------------

def usable(data):
    try:
        with Image.open(BytesIO(data)) as im:
            im = im.convert("RGB")
            brightness = sum(ImageStat.Stat(im).mean) / 3
            return brightness >= 1
    except Exception:
        return False


def validate_image(data):
    with Image.open(BytesIO(data)) as im:
        im.verify()


# --------------------------------------------------
# Remote MovieNet ZIP access
# --------------------------------------------------

def read_remote_frame(imdb_id, shot_idx, keyframe_idx):
    filename = keyframe_name(shot_idx, keyframe_idx)

    entries = remote_index().get(imdb_id, [])

    entry = next(
        (
            e for e in entries
            if e[0].rsplit("/", 1)[-1] == filename
        ),
        None
    )

    if entry is None:
        raise FileNotFoundError(filename)

    name, offset, compressed_size, method = entry

    rf = RangeFile(FRAME_PARTS)

    with remote_lock:
        rf.seek(offset)
        header = rf.read(30)

        if len(header) != 30:
            raise IOError("Incomplete ZIP header")

        fields = struct.unpack("<IHHHHHIIIHH", header)

        if fields[0] != 0x04034B50:
            raise IOError("Invalid ZIP local header")

        filename_length = fields[9]
        extra_length = fields[10]

        rf.read(filename_length + extra_length)
        compressed = rf.read(compressed_size)

    if len(compressed) != compressed_size:
        raise IOError("Incomplete compressed frame")

    if method == 0:
        data = compressed
    elif method == 8:
        data = zlib.decompress(compressed, -15)
    else:
        raise IOError(
            f"Unsupported ZIP compression method: {method}"
        )

    validate_image(data)
    return data


# --------------------------------------------------
# Film frame extraction
# --------------------------------------------------

def get_one_film_frame(imdb_id, shot_idx, keyframe_idx):
    try:
        path = keyframe_paths(imdb_id, shot_idx)[keyframe_idx]

        with open_image(path) as original:
            output = BytesIO()

            original.convert("RGB").save(
                output,
                format="JPEG",
                quality=90
            )

            data = output.getvalue()

        if usable(data):
            return data

    except (
        FileNotFoundError,
        KeyError,
        IndexError,
        OSError
    ):
        pass

    data = read_remote_frame(
        imdb_id,
        shot_idx,
        keyframe_idx
    )

    if usable(data):
        return data

    raise FileNotFoundError(
        keyframe_name(shot_idx, keyframe_idx)
    )


@lru_cache(maxsize=256)
def get_film_frame_bytes(imdb_id, shot_idx, keyframe_idx):
    if shot_idx < 0 or keyframe_idx not in (0, 1, 2):
        raise ValueError("Invalid frame index")

    order = [
        keyframe_idx,
        *[i for i in range(3) if i != keyframe_idx]
    ]

    for idx in order:
        try:
            return get_one_film_frame(
                imdb_id,
                shot_idx,
                idx
            )
        except (
            FileNotFoundError,
            KeyError,
            IndexError,
            OSError
        ):
            continue

    raise FileNotFoundError(
        f"No usable frame for shot {shot_idx}"
    )


# --------------------------------------------------
# Trailer frame extraction
# --------------------------------------------------

@lru_cache(maxsize=64)
def trailer_metadata(imdb_id):
    video = next(
        iter(TRAILERS.glob(f"{imdb_id}.*")),
        None
    )

    if video is None:
        raise FileNotFoundError(
            f"Trailer not found: {imdb_id}"
        )

    fps, _, _ = probe(video)

    if not fps or fps <= 0:
        raise ValueError("Invalid trailer FPS")

    import pandas as pd
    from .paths import FEATURES

    csv_path = (
        FEATURES
        / "trailer_shots"
        / f"{imdb_id}.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Shot metadata not found: {imdb_id}"
        )

    shots = pd.read_csv(csv_path)

    return str(video), fps, shots


@lru_cache(maxsize=512)
def get_trailer_frame_bytes(
    imdb_id,
    shot_idx,
    keyframe_idx
):
    if shot_idx < 0 or keyframe_idx not in (0, 1, 2):
        raise ValueError("Invalid frame index")

    video, fps, shots = trailer_metadata(imdb_id)

    match = shots[shots.shot_idx == shot_idx]

    if match.empty:
        raise FileNotFoundError(
            f"Trailer shot not found: {shot_idx}"
        )

    shot = match.iloc[0]

    frame_number = int(
        shot.frame_start
        + TRAILER_FRACTIONS[keyframe_idx]
        * (shot.frame_end - shot.frame_start)
    )

    timestamp = frame_number / fps

    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-ss", str(timestamp),
            "-i", video,
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "pipe:1"
        ],
        capture_output=True,
        check=False,
        timeout=30
    )

    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(
            "Trailer frame extraction failed"
        )

    validate_image(result.stdout)
    return result.stdout


# --------------------------------------------------
# HTTP response helper
# --------------------------------------------------

def image_response(data):
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=3600"
        }
    )


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

app.mount("/viz", StaticFiles(directory=str(ROOT / "viz"), html=True), name="viz")
app.mount("/data", StaticFiles(directory=str(ROOT / "data")), name="data")

@app.get("/")
def home():
    return RedirectResponse(url="/viz/")

@app.get("/api/films")
def get_available_films():
    import pandas as pd

    status_path = ROOT / "data" / "ids" / "trailer_status.csv"
    audit_path = ROOT / "data" / "ids" / "trailer_audit.csv"

    status_df = pd.read_csv(status_path, dtype=str)
    audit_df = pd.read_csv(audit_path, dtype=str)

    # Strip whitespace from column names and values
    status_df.columns = status_df.columns.str.strip()
    audit_df.columns = audit_df.columns.str.strip()

    ok_ids = (
        status_df.loc[status_df["status"].str.strip().str.lower() == "ok", "imdb_id"]
        .dropna()
        .str.strip()
        .unique()
    )

    audit_clean = (
        audit_df.dropna(subset=["imdb_id", "film"])
        .assign(
            imdb_id=lambda d: d["imdb_id"].str.strip(),
            film=lambda d: d["film"].str.strip(),
        )
    )
    audit_clean = audit_clean[audit_clean["film"] != ""]
    film_lookup = dict(zip(audit_clean["imdb_id"], audit_clean["film"]))

    return [
        {"id": iid, "title": film_lookup[iid]}
        for iid in ok_ids
        if iid in film_lookup
    ]

@app.get(
    "/api/film-frame/{imdb_id}/{shot_idx}/{keyframe_idx}"
)
def film_frame(imdb_id: str, shot_idx: int, keyframe_idx: int):
    if shot_idx < 0 or keyframe_idx not in (0, 1, 2):
        raise HTTPException(400, "Invalid frame index")

    try:
        return image_response(
            get_film_frame_bytes(
                imdb_id, shot_idx, keyframe_idx
            )
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get(
    "/api/trailer-frame/{imdb_id}/{shot_idx}/{keyframe_idx}"
)
def trailer_frame(imdb_id: str, shot_idx: int, keyframe_idx: int):
    if shot_idx < 0 or keyframe_idx not in (0, 1, 2):
        raise HTTPException(400, "Invalid frame index")

    try:
        return image_response(
            get_trailer_frame_bytes(
                imdb_id, shot_idx, keyframe_idx
            )
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))