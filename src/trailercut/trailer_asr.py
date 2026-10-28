"""Transcribe trailer speech with Whisper, with word timestamps.

Trailer dialogue is cut from the film's soundtrack, so a transcribed line can be found in the film's
subtitles (dialogue_align.py), which places that moment of the trailer's audio in the film without
looking at pixels. Voice activity detection skips music-only stretches.

Outputs (gitignored, under the data root: transcripts are copyrighted text)
  cache/trailer_asr/{imdb_id}.json   language + segments [{start, end, text, words: [{start, end, word, p}]}]

  python -m trailercut.trailer_asr [--model small] [--limit N]
"""
import argparse
import json

import librosa
import soundfile as sf
import torch
from tqdm import tqdm

from .paths import CACHE

AUDIO = CACHE / "trailer_audio"
OUT = CACHE / "trailer_asr"


def transcribe(model, path):
    # decode ourselves: faster-whisper's own decoder breaks with some PyAV versions
    y, sr = sf.read(path, dtype="float32")
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    segs, info = model.transcribe(y, word_timestamps=True, vad_filter=True)
    return dict(language=info.language, language_p=round(info.language_probability, 3), segments=[
        dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip(),
             words=[dict(start=round(w.start, 2), end=round(w.end, 2), word=w.word.strip(), p=round(w.probability, 3))
                    for w in (s.words or [])])
        for s in segs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    cuda = torch.cuda.is_available()
    model = WhisperModel(a.model, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(AUDIO.glob("*.wav")) if not (OUT / f"{p.stem}.json").exists()]
    if a.limit:
        todo = todo[: a.limit]
    for p in tqdm(todo):
        try:
            (OUT / f"{p.stem}.json").write_text(json.dumps(transcribe(model, p)))
        except Exception as e:
            print(p.stem, "failed:", e)


if __name__ == "__main__":
    main()
