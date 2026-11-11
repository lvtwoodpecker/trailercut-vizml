"""Can a vision-language model pick the right film shot? Trial on the hand-labeled films.

For every labeled trailer shot, Qwen2.5-VL sees the trailer shot's middle keyframe and the middle
keyframes of its 5 CLIP candidates (the B/32 alignment in data/features/alignment/), in shuffled
order so it can't favor CLIP's first pick, and answers which one is the same shot, or 0 for none.
Graded against the labels:
  pick_acc     it picks the labeled film shot (shots whose labeled source is among the 5)
  clip_acc     CLIP's top-1 on the same shots, for comparison
  none_rate    it answers 0, for shots labeled "not in film" (n) and "same scene, other shot" (s)
  combined     CLIP decides when sure (score >= the impostor cutoff: keep its top-1; below LO: no
               match), the VLM decides in between; precision / recall / F1 as label_eval
  vlm_all      the VLM decides every shot
Answers are cached per shot (cache/vlm/{model}/answers.jsonl), so a rerun resumes. Runs on a GPU
(the 7B model fits a 48 GB card; 32B wants an H200). Frames never leave the machine.

  python -m trailercut.vlm_rerank [--model Qwen/Qwen2.5-VL-7B-Instruct] [--limit N] [--dry-run]
Writes data/features/vlm_trial_{model}.json
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import movienet_io as mn
from .label_eval import load_gold, one_per_shot
from .paths import CACHE, FEATURES

LO = 0.40  # below this, no match (the bottom of label_eval's cutoff sweep); fixed before the run
QUESTION = ("Which numbered candidate shows the same shot as image T: the same moment, from the same "
            "camera angle, with the same people and framing? Small differences in color, brightness, "
            "cropping or black bars don't matter. If none of them is the same shot, answer 0. "
            "Answer with a single number.")


def shots_to_ask(rng):
    gold = one_per_shot(load_gold())
    al = pd.concat([pd.read_csv(FEATURES / "alignment" / f"{i}.csv").assign(imdb_id=i) for i in gold.imdb_id.unique()])
    d = gold.merge(al, on=["imdb_id", "shot_idx"])
    d = d[(d.label != "u") & ~d.blank].copy()
    d["cands"] = d.top5.str.split().apply(lambda c: [int(x) for x in c])
    d["order"] = [rng.permutation(5).tolist() for _ in range(len(d))]  # shown position -> top5 index
    return d


def images(r):
    t = mn.open_image(CACHE / "trailer_frames" / r.imdb_id / f"shot_{r.shot_idx:04d}_img_1.jpg").convert("RGB")
    c = [mn.open_image(mn.keyframe_paths(r.imdb_id, r.cands[k])[1]).convert("RGB") for k in r.order]
    return [t, *c]


def messages():
    content = [{"type": "text", "text": "Image T is one frame from a movie trailer."}, {"type": "image"},
               {"type": "text", "text": "Candidate frames from the film:"}]
    for k in range(1, 6):
        content += [{"type": "text", "text": f"{k}:"}, {"type": "image"}]
    content.append({"type": "text", "text": QUESTION})
    return [{"role": "user", "content": content}]


def parse(text):
    m = re.search(r"[0-5]", text)
    return int(m.group()) if m else -1


def ask_all(d, model_name, path, limit):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name, min_pixels=128 * 28 * 28, max_pixels=448 * 448)
    model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    done = {json.loads(l)["key"] for l in open(path)} if path.exists() else set()
    prompt = proc.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
    todo = [r for r in d.itertuples() if r.key not in done][: limit or None]
    with open(path, "a") as f:
        for r in tqdm(todo):
            x = proc(text=[prompt], images=images(r), return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**x, max_new_tokens=4, do_sample=False)
            raw = proc.batch_decode(out[:, x.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            f.write(json.dumps(dict(key=r.key, raw=raw, answer=parse(raw))) + "\n")
            f.flush()


def grade(d, hi):
    has = d.label.str.isdigit()
    src = d.label.where(has, "-1").astype(int)
    in5 = has & np.array([s in c for s, c in zip(src, d.cands)])
    vlm_shot = np.array([c[o[a - 1]] if a > 0 else -1 for a, c, o in zip(d.answer, d.cands, d.order)])
    out = dict(n=int(len(d)), answered=int((d.answer >= 0).sum()), source_in_top5=int(in5.sum()),
               pick_acc=float((vlm_shot[in5] == src[in5]).mean()),
               clip_acc=float((d.film_shot[in5] == src[in5]).mean()),
               none_rate={k: float((d.answer[d.label == k] == 0).mean()) for k in ("n", "s")})

    def prf(pred):
        m = pred >= 0
        right = m & has & (pred == src)
        p, r = right[m].mean(), right.sum() / has.sum()
        return dict(precision=float(p), recall=float(r), f1=float(2 * p * r / (p + r)))

    clip = np.where(d.score >= 0.515, d.film_shot, -1)  # the cutoff in use
    band = (d.score >= LO) & (d.score < hi)
    combined = np.where(d.score >= hi, d.film_shot, np.where(band, vlm_shot, -1))
    out.update(clip_at_0515=prf(clip), combined=dict(lo=LO, hi=hi, n_asked=int(band.sum()), **prf(combined)),
               vlm_all=prf(vlm_shot))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="build the inputs for one shot, no model")
    a = ap.parse_args()
    d = shots_to_ask(np.random.default_rng(0))
    if a.dry_run:
        r = next(d.itertuples())
        print(len(d), "shots;", r.key, "label", r.label, "cands", r.cands, "shown order", r.order,
              "images", [im.size for im in images(r)])
        return
    tag = a.model.split("/")[-1]
    path = CACHE / "vlm" / tag / "answers.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    ask_all(d, a.model, path, a.limit)
    ans = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates("key", keep="last")
    d = d.merge(ans, on="key")
    hi = json.loads((FEATURES / "alignment_threshold.json").read_text())["impostor_threshold"]
    res = dict(model=a.model, **grade(d, hi))
    (FEATURES / f"vlm_trial_{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
