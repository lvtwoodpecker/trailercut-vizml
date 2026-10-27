from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MOVIENET = DATA / "movienet"
FRAMES = MOVIENET / "frames"
TRAILERS = DATA / "trailers"
FEATURES = DATA / "features"
CACHE = DATA / "cache"
IDS = DATA / "ids" / "core_films_293.csv"
LABELS = ROOT / "labels"
FIGURES = ROOT / "figures"

for p in (FEATURES, CACHE, LABELS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)
