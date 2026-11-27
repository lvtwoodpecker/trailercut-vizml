"""What a trailer sounds like: music, its mood, sound effects, and whether the cuts land on the sound.

Sound tags come from AST fine-tuned on AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593), a trained
audio tagger, not a prompt: every 10 s window of the trailer's audio gets a probability for each AudioSet
class, and we keep the classes in TAGS. AudioSet's own mood classes ("Exciting music", "Scary music", ...)
are what the sound evokes, as its human raters labeled it.

Cuts on the sound: an audio hit is an onset (librosa). A cut lands on a hit if one is within HIT seconds.
Chance is the same rate for the cuts shifted by random offsets (same cut pattern, same audio), so
cut_on_hit_lift = 1 means the editor ignored the sound, above 1 means cuts follow it.

Per trailer (data/features/trailer_sound.csv, committed)
  music_share        share of 10 s windows where Music > 0.5
  <tag>              mean probability over windows, for each tag in TAGS (snake_case)
  cut_on_hit         share of cuts with an audio hit within HIT seconds
  cut_on_hit_lift    cut_on_hit over its chance rate
  boom_whoosh        any window with Boom or Whoosh above 0.3 (the modern trailer hit)
  orchestra_clear    any window with Orchestra above 0.2
  mood               the mood tag the trailer ranks highest on, as a percentile among all trailers
                     (the tags' raw levels differ a lot, so the raw maximum would always be the same one);
                     "Exciting music" is left out: it doesn't separate action trailers from the rest
  dark_mood          mood is scary or angry

Cache: data/cache/trailer_sound/{imdb_id}.json (per-window probabilities; the run resumes from it)

  python -m trailercut.trailer_sound [--ids tt..,tt..]
"""
import argparse
import json

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .paths import CACHE, FEATURES
from .trailer_prep import AUDIO_DIR

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
TAGS = ["Music", "Speech", "Silence",
        "Exciting music", "Scary music", "Sad music", "Happy music", "Tender music", "Angry music", "Funny music",
        "Orchestra", "Choir", "Electronic music", "Rock music", "Hip hop music", "Pop music", "Drum",
        "Explosion", "Gunshot, gunfire", "Whoosh, swoosh, swish", "Boom"]
# moods for the per-trailer `mood` (AudioSet's mood classes, minus Exciting, see above)
MOODS = ["Scary music", "Angry music", "Sad music", "Tender music", "Happy music", "Funny music"]
WIN = 10.0
HIT = 0.07
OUT = CACHE / "trailer_sound"


def col(tag):
    return tag.lower().replace(", ", "_").replace(" ", "_").split("_swoosh")[0].split("_gunfire")[0]


def tag_windows(path, fx, model, idx, dev):
    y, sr = librosa.load(path, sr=16000)
    n = max(1, int(np.ceil(len(y) / (WIN * sr))))
    wins = [y[int(i * WIN * sr):int((i + 1) * WIN * sr)] for i in range(n)]
    wins = [w for w in wins if len(w) > sr]   # skip a last sliver under 1 s
    probs = []
    for b in range(0, len(wins), 16):
        x = fx(wins[b:b + 16], sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            p = torch.sigmoid(model(x["input_values"].to(dev)).logits)[:, idx]
        probs.append(p.cpu().numpy())
    return np.concatenate(probs).round(4).tolist()


def cut_hits(imdb_id, rng):
    shots = FEATURES / "trailer_shots" / f"{imdb_id}.csv"
    if not shots.exists():
        return None, None
    cuts = pd.read_csv(shots).t_start.values[1:]
    y, sr = librosa.load(AUDIO_DIR / f"{imdb_id}.wav", sr=22050)
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    if len(cuts) < 5 or len(on) < 5:
        return None, None
    dur = len(y) / sr
    hit = lambda c: (np.abs(on[None, :] - c[:, None]).min(1) <= HIT).mean()
    rate = hit(cuts)
    chance = np.mean([hit((cuts + rng.uniform(0.3, dur - 0.3)) % dur) for _ in range(20)])
    return float(rate), float(rate / chance) if chance else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids")
    a = ap.parse_args()
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    ids = a.ids.split(",") if a.ids else sorted(p.stem for p in AUDIO_DIR.glob("*.wav"))
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (OUT / f"{i}.json").exists()]
    if todo:
        dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        fx = ASTFeatureExtractor.from_pretrained(MODEL)
        model = ASTForAudioClassification.from_pretrained(MODEL).to(dev).eval()
        label2id = {v: k for k, v in model.config.id2label.items()}
        idx = [label2id[t] for t in TAGS]
        rng = np.random.default_rng(0)
        for i in tqdm(todo, desc="trailer sound"):
            try:
                rec = dict(windows=tag_windows(AUDIO_DIR / f"{i}.wav", fx, model, idx, dev))
                rec["cut_on_hit"], rec["cut_on_hit_lift"] = cut_hits(i, rng)
            except Exception as e:  # a broken wav shouldn't stop the run
                rec = dict(error=str(e))
            (OUT / f"{i}.json").write_text(json.dumps(rec))   # saved per trailer, so the run resumes
    rows = []
    for p in sorted(OUT.glob("*.json")):
        r = json.loads(p.read_text())
        if "error" in r or not r["windows"]:
            continue
        w = np.array(r["windows"])
        row = dict(imdb_id=p.stem, music_share=float((w[:, 0] > 0.5).mean()))
        row.update({col(t): float(w[:, k].mean()) for k, t in enumerate(TAGS)})
        row.update(cut_on_hit=r["cut_on_hit"], cut_on_hit_lift=r["cut_on_hit_lift"],
                   boom_whoosh=bool((w[:, [TAGS.index("Boom"), TAGS.index("Whoosh, swoosh, swish")]] > 0.3).any()),
                   orchestra_clear=bool((w[:, TAGS.index("Orchestra")] > 0.2).any()))
        rows.append(row)
    df = pd.DataFrame(rows)
    pct = df[[col(t) for t in MOODS]].rank(pct=True)
    df["mood"] = pct.idxmax(axis=1).str.replace("_music", "")
    df["dark_mood"] = df.mood.isin(["scary", "angry"])
    df.round(4).to_csv(FEATURES / "trailer_sound.csv", index=False)
    print(f"-> {FEATURES / 'trailer_sound.csv'} ({len(rows)} trailers)")


if __name__ == "__main__":
    main()
