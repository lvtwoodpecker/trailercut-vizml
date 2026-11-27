"""What a trailer makes you feel, judged from picture and sound together (Qwen2.5-Omni).

Qwen2.5-Omni takes a video with its audio track as one input. Each trailer is cut into CHUNK-second
pieces; for each piece the model rates EMOTIONS from 0 (not at all) to 3 (strongly) as JSON. A trailer's
score per emotion is the mean over its pieces.

Modes, each with its own cache, so they can be compared on the same pieces:
  av      picture and sound (the main run)
  video   picture only (the audio track is not given), to measure what the sound adds

Checks without labels (see docs/design.md): genre (fear highest in horror, humor in comedy, ...), and
agreement with trailer_sound's AudioSet moods and the CLIP concepts.

Cache: cache/av_judge/{mode}/{imdb_id}.json (per piece: ratings and the raw answer; reruns resume)
Writes data/features/av_judge_{mode}.csv (one row per trailer, mean rating per emotion)

  python -m trailercut.av_judge [--mode av|video] [--shard 0 --shards 4] [--limit N] [--ids tt..,tt..]
  python -m trailercut.av_judge --collect [--mode av]      # cache -> csv only
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .paths import CACHE, FEATURES, TRAILERS

MODEL = "Qwen/Qwen2.5-Omni-7B"
CHUNK = 30
EMOTIONS = ["fear", "tension", "excitement", "sadness", "joy", "humor", "romance", "awe"]
SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
          "auditory and visual inputs, as well as generating text and speech.")   # the model card's system prompt
PROMPT = ("This is part of a movie trailer. Rate how strongly it makes a viewer feel each emotion, from 0 (not at "
          "all) to 3 (strongly), using everything you {senses}. Answer with JSON only, exactly these keys: "
          + json.dumps({e: 0 for e in EMOTIONS}))
SENSES = {"av": "see and hear (picture, music, voices, sound effects)", "video": "see"}
OUT = CACHE / "av_judge"


def ffmpeg():
    if shutil.which("ffmpeg") is None:
        import static_ffmpeg
        static_ffmpeg.add_paths()
    return "ffmpeg"


def duration(path):
    out = subprocess.run([ffmpeg().replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return float(out.strip() or 0)


def pieces(path, tmp):
    """Cut the trailer into CHUNK-second mp4 pieces (picture and sound) in tmp."""
    n = max(1, int(np.ceil(duration(path) / CHUNK)))
    out = []
    for k in range(n):
        p = Path(tmp) / f"{k:03d}.mp4"
        subprocess.run([ffmpeg(), "-v", "error", "-y", "-ss", str(k * CHUNK), "-t", str(CHUNK), "-i", str(path),
                        "-vf", "scale=-2:360", "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-ac", "1",
                        str(p)], check=True)
        if p.exists() and p.stat().st_size > 2000:   # a last sliver can come out empty
            out.append(p)
    return out


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return None
    vals = {e: d.get(e) for e in EMOTIONS}
    return vals if all(isinstance(v, (int, float)) and 0 <= v <= 3 for v in vals.values()) else None


class Judge:
    def __init__(self, mode):
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        self.torch, self.mode = torch, mode
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.disable_talker()   # text answers only: the speech generator isn't needed (saves ~2 GB)
        self.proc = Qwen2_5OmniProcessor.from_pretrained(MODEL)
        import inspect
        self.v5 = "generation_mode" in inspect.getsource(type(self.model).generate)

    def rate(self, piece):
        from qwen_omni_utils import process_mm_info
        audio = self.mode == "av"
        conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "video", "video": str(piece), "fps": 2.0, "max_pixels": 360 * 640},
                                             {"type": "text", "text": PROMPT.format(senses=SENSES[self.mode])}]}]
        text = self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=audio)
        inp = self.proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                        use_audio_in_video=audio).to(self.model.device).to(self.model.dtype)
        # text only: transformers 5 asks for generation_mode="text", 4.x for return_audio=False
        text_only = dict(generation_mode="text") if self.v5 else dict(return_audio=False)
        with self.torch.no_grad():
            ids = self.model.generate(**inp, use_audio_in_video=audio, thinker_max_new_tokens=96, thinker_do_sample=False, **text_only)
        ids = ids[0] if isinstance(ids, tuple) else ids
        return self.proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def collect(mode):
    rows = []
    for p in sorted((OUT / mode).glob("*.json")):
        ok = [c["ratings"] for c in json.loads(p.read_text())["pieces"] if c["ratings"]]
        if ok:
            rows.append(dict(imdb_id=p.stem, pieces=len(ok), **{e: float(np.mean([c[e] for c in ok])) for e in EMOTIONS}))
    df = pd.DataFrame(rows).round(3)
    df.to_csv(FEATURES / f"av_judge_{mode}.csv", index=False)
    print(f"-> {FEATURES / f'av_judge_{mode}.csv'} ({len(df)} trailers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(SENSES), default="av")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N trailers of this shard (a pilot)")
    ap.add_argument("--ids")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        return collect(a.mode)
    ids = a.ids.split(",") if a.ids else sorted(p.stem for p in TRAILERS.glob("tt*.mp4"))
    ids = ids[a.shard::a.shards][: a.limit or None]
    (OUT / a.mode).mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (OUT / a.mode / f"{i}.json").exists()]
    print(f"{len(todo)} of {len(ids)} trailers to judge ({a.mode}, shard {a.shard}/{a.shards})")
    if not todo:
        return
    judge = Judge(a.mode)
    bad = 0
    for i in tqdm(todo, desc=f"judge {a.mode}"):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            try:
                rec = []
                for k, p in enumerate(pieces(TRAILERS / f"{i}.mp4", tmp)):
                    raw = judge.rate(p)
                    rec.append(dict(t0=k * CHUNK, raw=raw, ratings=parse(raw)))
                    bad += rec[-1]["ratings"] is None
            except Exception as e:   # one broken video shouldn't stop the shard; it gets retried next run
                print(i, "failed:", e)
                continue
        (OUT / a.mode / f"{i}.json").write_text(json.dumps(dict(model=MODEL, chunk=CHUNK, pieces=rec)))   # saved per trailer
    print("answers that weren't valid JSON:", bad)


if __name__ == "__main__":
    main()
