"""Can a vision-language model tell whether two shots are the same? Trial on the hand-labeled films.

Pairwise, as a verifier. For every labeled trailer shot and each of CLIP's 5 candidates (the B/32
alignment in data/features/alignment/), Qwen2.5-VL sees the trailer shot and one film shot and answers
whether they are the same shot, allowing a different moment of it. The score is P(yes) from the
next-token logits, so it can rerank the 5, reject footage that isn't in the film, and blend with CLIP.
(A first version showed all 6 frames in one prompt and asked for a number: 54% right where CLIP's top
pick was 94%; one middle frame per shot also meant comparing different moments of the same shot.)

Settings, each with its own answer cache (cache/vlm_pair/{model}_{frames}f.jsonl, so reruns resume):
  --model    Qwen/Qwen2.5-VL-7B-Instruct (fits a 48 GB card) | Qwen/Qwen2.5-VL-32B-Instruct (an H200)
  --frames   1: middle keyframe; 3: the shot's 3 keyframes side by side
Trailer frames come from cache/trailer_frames_hr/ (the video's own size, `frames` subcommand, needs the
trailer videos) when present, else the 320x180 keyframes. Film frames are MovieNet's (577x240 at most).

Graded on the same shots as CLIP:
  rescue     CLIP's top-1 is wrong but the labeled shot is among the 5: share the VLM ranks first
  keep       CLIP's top-1 is right: share the VLM still ranks it first
  reject     labeled "not in the film": share where the VLM's best candidate has P(yes) < 0.5
  combined   CLIP decides when sure (score >= the impostor cutoff: its top-1; below LO: none); in between,
             the candidate with the best blend (1 - w) z(CLIP score) + w z(logit P(yes)), if it clears a
             cutoff. w and the cutoff are chosen leave-one-film-out (each film graded with settings
             picked on the other 8), so the F1 is not tuned on the shots it is graded on
--pilot N grades a fixed sample of N shots (a third CLIP misses that its top 5 holds, a third "not in the
film", a third CLIP gets right) to compare settings cheaply.

  python -m trailercut.vlm_rerank frames                         # laptop: high-res trailer frames
  python -m trailercut.vlm_rerank [--model M] [--frames 3] [--pilot 60]
Writes data/features/vlm_pair_{model}_{frames}f[_pilot{N}].json
"""
import argparse
import json
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import movienet_io as mn
from .label_eval import load_gold, one_per_shot
from .paths import CACHE, FEATURES, TRAILERS

LO = 0.40  # below this, no match (the bottom of label_eval's cutoff sweep); fixed before the run
HR = CACHE / "trailer_frames_hr"
FRACS = (0.25, 0.5, 0.75)  # same keyframe positions as trailer_prep
QUESTION = ("Do A and B show the same shot: the same scene filmed from the same camera setup, with the same "
            "people and objects in the same places? B may be a different moment of that shot. Color, "
            "brightness, cropping, black bars and text overlays may differ. Answer Yes or No.")


def gold_shots():
    gold = one_per_shot(load_gold())
    al = pd.concat([pd.read_csv(FEATURES / "alignment" / f"{i}.csv").assign(imdb_id=i) for i in gold.imdb_id.unique()])
    d = gold.merge(al, on=["imdb_id", "shot_idx"])
    d = d[(d.label != "u") & ~d.blank].copy()
    d["cands"] = d.top5.str.split().apply(lambda c: [int(x) for x in c])
    d["cand_scores"] = d.top5_scores.str.split().apply(lambda c: [float(x) for x in c])
    d["has"] = d.label.str.isdigit()
    d["src"] = d.label.where(d.has, "-1").astype(int)
    d["in5"] = [s in c for s, c in zip(d.src, d.cands)]
    return d.reset_index(drop=True)


def pilot_sample(d, n, seed=0):
    rng = np.random.default_rng(seed)
    groups = [d[d.has & d.in5 & (d.film_shot != d.src)], d[d.label == "n"], d[d.has & (d.film_shot == d.src)]]
    take = [g.index[rng.permutation(len(g))[: n // 3]] for g in groups]
    return d.loc[np.concatenate(take)]


# ---- images ----
def hires_frames():
    """Re-extract the trailer keyframes of the labeled films at the video's own size (laptop, needs videos)."""
    from PIL import Image
    from .trailer_prep import probe, read_frames
    ids = sorted(one_per_shot(load_gold()).imdb_id.unique())
    for i in tqdm(ids, desc="frames"):
        sh = pd.read_csv(FEATURES / "trailer_shots" / f"{i}.csv")
        _, w, h = probe(TRAILERS / f"{i}.mp4")
        frames = read_frames(TRAILERS / f"{i}.mp4", w, h)
        out = HR / i
        out.mkdir(parents=True, exist_ok=True)
        for r in sh.itertuples():
            for k, frac in enumerate(FRACS):
                f = int(r.frame_start + frac * (r.frame_end - r.frame_start))
                Image.fromarray(frames[min(f, len(frames) - 1)]).save(out / f"shot_{r.shot_idx:04d}_img_{k}.jpg", quality=92)


def strip(paths):
    """Frames side by side at a common height."""
    from PIL import Image
    ims = [mn.open_image(p).convert("RGB") for p in paths]
    h = min(im.height for im in ims)
    ims = [im.resize((round(im.width * h / im.height), h)) for im in ims]
    out = Image.new("RGB", (sum(im.width for im in ims) + 8 * (len(ims) - 1), h), (0, 0, 0))
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width + 8
    return out


def trailer_image(i, s, frames):
    d = HR / i if (HR / i).exists() else CACHE / "trailer_frames" / i
    ks = [1] if frames == 1 else [0, 1, 2]
    return strip([d / f"shot_{s:04d}_img_{k}.jpg" for k in ks])


def film_image(i, s, frames):
    p = mn.keyframe_paths(i, s)
    return strip([p[1]] if frames == 1 else p)


# ---- the model ----
def messages():
    return [{"role": "user", "content": [
        {"type": "text", "text": "Image A: from one shot of a movie trailer."}, {"type": "image"},
        {"type": "text", "text": "Image B: from one shot of the film."}, {"type": "image"},
        {"type": "text", "text": QUESTION}]}]


def score_all(d, model_name, frames, path):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_name, min_pixels=64 * 28 * 28, max_pixels=1920 * 360)
    model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    tok = proc.tokenizer
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]
    prompt = proc.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
    done = {(r["key"], r["k"]) for r in map(json.loads, open(path))} if path.exists() else set()
    with open(path, "a") as f, torch.no_grad():
        for r in tqdm(list(d.itertuples()), desc=f"{model_name.split('/')[-1]} {frames}f"):
            a = trailer_image(r.imdb_id, r.shot_idx, frames)
            for k, c in enumerate(r.cands):
                if (r.key, k) in done:
                    continue
                x = proc(text=[prompt], images=[a, film_image(r.imdb_id, c, frames)], return_tensors="pt").to("cuda")
                logits = model(**x).logits[0, -1].float()
                ly, ln = torch.logsumexp(logits[yes], 0), torch.logsumexp(logits[no], 0)
                f.write(json.dumps(dict(key=r.key, k=k, film_shot=c, logit=round(float(ly - ln), 4))) + "\n")
                f.flush()


# ---- grading ----
def prf(pred, d):
    m = pred >= 0
    right = m & d.has.values & (pred == d.src.values)
    p, r = (right[m].mean() if m.any() else 0.0), right.sum() / d.has.sum()
    return dict(precision=float(p), recall=float(r), f1=float(2 * p * r / (p + r)) if p + r else 0.0)


def combined(d, w, cut, hi):
    """Per shot: CLIP's top-1 when sure, none when weak, else the best blended candidate if it clears cut."""
    out = np.full(len(d), -1)
    for n, r in enumerate(d.itertuples()):
        if r.score >= hi:
            out[n] = r.film_shot
        elif r.score >= LO:
            s = (1 - w) * np.array(r.z_clip) + w * np.array(r.z_vlm)
            if s.max() >= cut:
                out[n] = r.cands[int(s.argmax())]
    return out


def grade(d, hi):
    lg = np.array(d.logits.tolist())
    best = lg.argmax(1)
    vlm_top = np.array([c[b] for c, b in zip(d.cands, best)])
    miss = (d.has & d.in5 & (d.film_shot != d.src)).values
    hit = (d.has & (d.film_shot == d.src)).values
    out = dict(n=int(len(d)), clip_wrong_in_top5=int(miss.sum()), not_in_film=int((d.label == "n").sum()),
               rescue=float((vlm_top[miss] == d.src.values[miss]).mean()) if miss.any() else None,
               keep=float((vlm_top[hit] == d.src.values[hit]).mean()) if hit.any() else None,
               reject=float((lg[(d.label == "n").values].max(1) < 0).mean()) if (d.label == "n").any() else None,
               vlm_top1_acc=float((vlm_top[d.in5.values] == d.src.values[d.in5.values]).mean()),
               clip_top1_acc=float((d.film_shot.values[d.in5.values] == d.src.values[d.in5.values]).mean()))
    # z-scores over all candidate pairs, so the two scores blend on one scale
    cs = np.array(d.cand_scores.tolist())
    d = d.assign(z_clip=list((cs - cs.mean()) / cs.std()), z_vlm=list((lg - lg.mean()) / lg.std()))
    ws, cuts = np.linspace(0, 1, 11), np.linspace(-2, 2, 41)
    out["clip_at_0515"] = prf(np.where(d.score >= 0.515, d.film_shot, -1), d)
    pred = np.full(len(d), -1)
    chosen = {}
    for film in d.imdb_id.unique():   # leave one film out: settings picked on the other films
        tr, te = d[d.imdb_id != film], d.imdb_id.values == film
        w, cut = max(((w, c) for w in ws for c in cuts), key=lambda wc: prf(combined(tr, *wc, hi), tr)["f1"])
        pred[te] = combined(d[te], w, cut, hi)
        chosen[film] = dict(w=float(w), cut=float(cut))
    out["combined_loo"] = dict(lo=LO, hi=hi, n_band=int(((d.score >= LO) & (d.score < hi)).sum()), **prf(pred, d), settings=chosen)
    out["vlm_only"] = prf(np.where(lg.max(1) > 0, vlm_top, -1), d)   # the VLM decides every shot: its best, if P(yes) > 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="run", choices=["run", "frames"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=3, choices=[1, 3])
    ap.add_argument("--pilot", type=int, default=0, help="grade a fixed sample of this many shots")
    a = ap.parse_args()
    if a.what == "frames":
        return hires_frames()
    d = gold_shots()
    if a.pilot:
        d = pilot_sample(d, a.pilot)
    tag = f"{a.model.split('/')[-1]}_{a.frames}f"
    path = CACHE / "vlm_pair" / f"{tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    score_all(d, a.model, a.frames, path)
    ans = pd.DataFrame(map(json.loads, open(path))).drop_duplicates(["key", "k"], keep="last")
    lg = ans.pivot(index="key", columns="k", values="logit")
    d = d[d.key.isin(lg.index)].copy()
    d["logits"] = [lg.loc[k].values.tolist() for k in d.key]
    hi = json.loads((FEATURES / "alignment_threshold.json").read_text())["impostor_threshold"]
    res = dict(model=a.model, frames=a.frames, hires_trailer_frames=HR.exists(), pilot=a.pilot, **grade(d, hi))
    name = f"vlm_pair_{tag}" + (f"_pilot{a.pilot}" if a.pilot else "")
    (FEATURES / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "combined_loo"}, indent=1))
    print("combined (leave one film out):", {k: v for k, v in res["combined_loo"].items() if k != "settings"})


if __name__ == "__main__":
    main()
