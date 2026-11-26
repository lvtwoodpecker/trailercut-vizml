from pathlib import Path

from .paths import MOVIENET
from .movienet import RangeFile, FRAME_PARTS, load_index, fetch_film

def ensure_film_keyframes(imdb_id):
    frames_dir = MOVIENET / "frames"
    film_dir = frames_dir / imdb_id
    film_zip = frames_dir / f"{imdb_id}.zip"

    if film_dir.is_dir() or film_zip.is_file():
        print(f"Keyframes already available: {imdb_id}")
        return

    print(f"Fetching MovieNet keyframes: {imdb_id}")

    index_path = MOVIENET / "frames_index.pkl"
    idx = load_index(str(index_path))

    entries = idx.get(imdb_id)

    if not entries:
        raise FileNotFoundError(
            f"No MovieNet keyframes available for {imdb_id}"
        )

    fetch_film(
        RangeFile(FRAME_PARTS),
        entries,
        str(film_dir),
        pack=True
    )

    print(f"Downloaded keyframes: {imdb_id}")