"""Split the glued trailer shots that resplit (TransNetV2) left behind, using CLIP on every frame.

A shot is glued when its 3 keyframes disagree more than 99% of MovieNet's film shots ever do (the
film floor, align_diagnostics). For each such shot every frame is embedded with CLIP; the shot is
cut where two neighboring frames look least alike, and each piece is checked again the same way
(up to MAX_DEPTH rounds, pieces at least MIN_FRAMES long). Real shots that change a lot (a fast pan,
a flash) also get cut this way, so nothing is applied until a sample has been checked by eye:

  propose  write data/features/split_proposals.csv (one row per proposed cut, with `sim`: how alike
           the two frames around the cut are); saves per trailer and resumes; changes nothing else
  review   page to check a sample of proposed cuts (labels/split_review.html): --n of them spread
           evenly over `sim`, shown shuffled; keys y real cut, x same shot, u unsure; Export gives
           split_review_<name>.json (verdicts `y`, `n`, `u`)
  report   from the exported verdicts (labels/split_review_*.json): share of real cuts at each `sim`
           cutoff, and at each `sharp` cutoff (how much the cut dips below the frames around it)
  apply    cut at the proposals with sim <= --max-sim (picked from the report), log the trailers in
           data/features/split_applied.csv, then rerun `embed trailer` and everything after it

Hand-labeled trailers are skipped (their labels index the current shots).

  python -m trailercut.resplit_clip propose [--core]
  python -m trailercut.resplit_clip review [--n 50]
  python -m trailercut.resplit_clip report
  python -m trailercut.resplit_clip apply --max-sim S
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .align import load
from .align_diagnostics import consistency
from .paths import CLIP, FEATURES, IDS_DIR, LABELS, TRAILERS
from .resplit import floor, labeled_now, rewrite
from .trailer_prep import FRAMES_DIR, KEYFRAME_FRACS, SHOTS_DIR, probe, read_frames

PROPOSALS = FEATURES / "split_proposals.csv"
APPLIED = FEATURES / "split_applied.csv"
PAGE = LABELS / "split_review.html"
IMG = LABELS / "split_img"
MIN_FRAMES, MAX_DEPTH = 6, 3


@torch.no_grad()
def embed(model, preprocess, frames, box, device, bs=256):
    out = []
    for k in range(0, len(frames), bs):
        ims = [Image.fromarray(f) for f in frames[k:k + bs]]
        x = torch.stack([preprocess(im.crop(box) if box else im) for im in ims]).to(device).half()
        out.append(torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def cuts_in(a, b, E, f0, depth=0):
    """Cuts for frames a..b (E: their embeddings, one row per frame) -> [(frame, sim, sharp, depth)]."""
    n = b - a + 1
    if n < 2 * MIN_FRAMES or depth >= MAX_DEPTH:
        return []
    k = [int(f * (n - 1)) for f in KEYFRAME_FRACS]  # same keyframes as trailer_prep
    if consistency(E[k][None])[0] >= f0:
        return []
    sim = (E[:-1] * E[1:]).sum(1)  # sim[j]: frame a+j vs a+j+1
    j = MIN_FRAMES - 1 + int(sim[MIN_FRAMES - 1:n - MIN_FRAMES].argmin())
    c = a + j + 1
    # sharpness: a hard cut is a one-frame dip; a pan, flash or dissolve stays low over several frames
    near = np.r_[sim[max(j - 5, 0):j], sim[j + 1:j + 6]]
    sharp = float(np.median(near) - sim[j]) if len(near) else 0.0
    return [(c, float(sim[j]), sharp, depth)] + cuts_in(a, c - 1, E[:j + 1], f0, depth + 1) + cuts_in(c, b, E[j + 1:], f0, depth + 1)


def current(df, now):
    """Rows made on the trailer in use now."""
    return df[df.youtube_id == df.imdb_id.map(now)]


def propose(core):
    from .embed import DEVICE, bar_box, load_model
    now, labeled = labeled_now()
    done = set(current(pd.read_csv(PROPOSALS, dtype={"youtube_id": str}), now).imdb_id) if PROPOSALS.exists() else set()
    films = pd.read_csv(IDS_DIR / "films_1100.csv", dtype=str)
    keep = set(films[films.core == "1"].imdb_id) if core else set(films.imdb_id)
    f0 = floor()
    todo = []
    for p in sorted(SHOTS_DIR.glob("*.csv")):
        i = p.stem
        if i in keep and i not in labeled and i not in done and (CLIP / "trailer" / f"{i}.npy").exists():
            todo.append((i, np.flatnonzero(consistency(load("trailer", i).numpy()) < f0)))
    print(f"{len(todo)} trailers to check, {sum(len(s) for _, s in todo)} glued shots")
    model, preprocess, _ = load_model()
    for i, glued in tqdm(todo):
        rows = []
        if len(glued):
            frames = read_frames(TRAILERS / f"{i}.mp4", 320, 180)  # same size as the keyframes
            box = bar_box(sorted((FRAMES_DIR / i).glob("*.jpg")))
            sh = pd.read_csv(SHOTS_DIR / f"{i}.csv")
            for s in glued:
                a, b = int(sh.frame_start[s]), min(int(sh.frame_end[s]), len(frames) - 1)
                E = embed(model, preprocess, frames[a:b + 1], box, DEVICE)
                rows += [dict(imdb_id=i, youtube_id=now.get(i), shot_idx=int(s), frame_start=a, frame_end=int(sh.frame_end[s]),
                              cut_frame=c, sim=round(x, 4), sharp=round(h, 4), depth=d) for c, x, h, d in cuts_in(a, b, E, f0)]
        # a shot_idx -1 row marks a trailer checked with nothing to cut, so a rerun skips it
        out = pd.DataFrame(rows or [dict(imdb_id=i, youtube_id=now.get(i), shot_idx=-1)])
        out.reindex(columns=["imdb_id", "youtube_id", "shot_idx", "frame_start", "frame_end", "cut_frame", "sim", "sharp", "depth"]) \
           .to_csv(PROPOSALS, mode="a", header=not PROPOSALS.exists(), index=False)
    df = pd.read_csv(PROPOSALS)
    cuts = df[df.shot_idx >= 0]
    print(f"{len(cuts)} cuts proposed in {cuts.imdb_id.nunique()} trailers, {cuts.groupby(['imdb_id', 'shot_idx']).ngroups} shots")


def proposals():
    now, labeled = labeled_now()
    df = current(pd.read_csv(PROPOSALS, dtype={"youtube_id": str}), now)
    return df[(df.shot_idx >= 0) & ~df.imdb_id.isin(labeled)]


def review(n):
    from . import movienet_io as mn
    df = proposals()
    seen = set(verdicts())
    df = df[~(df.imdb_id + ":" + df.cut_frame.astype(int).astype(str)).isin(seen)].sort_values("sim").reset_index(drop=True)
    # spread the sample evenly over sim, so the verdicts show where real cuts stop
    pick = df.iloc[np.unique(np.linspace(0, len(df) - 1, n).round().astype(int))]
    IMG.mkdir(parents=True, exist_ok=True)
    items = []
    for i, g in tqdm(pick.groupby("imdb_id")):
        path = TRAILERS / f"{i}.mp4"
        fps, _, _ = probe(path)
        frames = read_frames(path, 320, 180)
        for r in g.itertuples():
            c, last = int(r.cut_frame), min(int(r.frame_end), len(frames) - 1)
            ims = []
            for name, f in [("start", r.frame_start), ("before", c - 1), ("after", c), ("end", last)]:
                out = IMG / f"{i}_{c}_{name}.jpg"
                Image.fromarray(frames[int(f)]).save(out, quality=85)
                ims.append(f"split_img/{out.name}")
            items.append(dict(key=f"{i}:{c}", film=mn.meta(i)["title"], video=f"../data/trailers/{i}.mp4",
                              t0=round(max(r.frame_start / fps - 0.5, 0), 2), t1=round((r.frame_end + 1) / fps + 0.5, 2),
                              cut=round(c / fps, 2), ims=ims))
    rng = np.random.default_rng(0)
    rng.shuffle(items)  # not in sim order, so the page doesn't hint at the answer
    PAGE.write_text(HTML.replace("__ITEMS__", json.dumps(items)))
    print(f"{len(items)} proposed cuts -> {PAGE}")


def verdicts():
    """All exported verdicts, key imdb_id:cut_frame -> y / n / u (later files win)."""
    v = {}
    for p in sorted(LABELS.glob("split_review_*.json")):
        v.update(json.loads(p.read_text())["verdicts"])
    return v


def report():
    df = proposals()
    df["key"] = df.imdb_id + ":" + df.cut_frame.astype(int).astype(str)
    v = verdicts()
    r = df[df.key.isin(v)].assign(verdict=lambda d: d.key.map(v))
    r = r[r.verdict != "u"].sort_values("sim")
    print(f"{len(r)} verdicts ({(r.verdict == 'y').mean():.0%} real cuts); {len(df)} proposed cuts in all")
    print("max-sim  real cuts kept  share of proposals kept")
    for s in np.unique(np.quantile(r.sim, np.linspace(0.1, 1, 10)).round(2)):
        k = r[r.sim <= s]
        print(f"{s:7.2f}  {(k.verdict == 'y').mean():14.0%}  {(df.sim <= s).mean():23.0%}")
    if "sharp" in r and r.sharp.notna().any():
        print("min-sharp  real cuts kept  share of proposals kept")
        for s in np.unique(np.quantile(r.sharp.dropna(), np.linspace(0, 0.9, 10)).round(2)):
            k = r[r.sharp >= s]
            print(f"{s:9.2f}  {(k.verdict == 'y').mean():14.0%}  {(df.sharp >= s).mean():23.0%}")


def apply(max_sim):
    now, _ = labeled_now()
    done = set(current(pd.read_csv(APPLIED, dtype={"youtube_id": str}), now).imdb_id) if APPLIED.exists() else set()
    df = proposals()
    df = df[(df.sim <= max_sim) & ~df.imdb_id.isin(done)]
    for i, g in tqdm(df.groupby("imdb_id")):
        sh = pd.read_csv(SHOTS_DIR / f"{i}.csv")
        s = sh.set_index("shot_idx")
        if any(s.frame_start.get(r.shot_idx) != r.frame_start or s.frame_end.get(r.shot_idx) != r.frame_end for r in g.itertuples()):
            print(f"{i}: shots changed since the proposals, skipped")
            continue
        fps, _, _ = probe(TRAILERS / f"{i}.mp4")
        rewrite(i, sh, [int(c) for c in g.cut_frame], fps, int(round(sh.t_end.iloc[-1] * fps)))
        pd.DataFrame([dict(imdb_id=i, youtube_id=now.get(i), cuts=len(g), max_sim=max_sim)]) \
          .to_csv(APPLIED, mode="a", header=not APPLIED.exists(), index=False)
    print(f"{len(df)} cuts in {df.imdb_id.nunique()} trailers; now run `embed trailer` and everything after it")


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Shot splits</title>
<style>
body{font:14px system-ui;margin:16px;background:#151515;color:#eee}
#bar{position:sticky;top:0;background:#151515;padding:8px 0;border-bottom:1px solid #333;margin-bottom:12px;z-index:2}
h2{margin:6px 0 8px;font-size:20px}
video{width:560px;max-width:100%;border-radius:6px;background:#000;display:block;margin-bottom:10px}
.strip{display:flex;gap:6px;align-items:flex-start;flex-wrap:wrap}
.strip figure{margin:0}
.strip img{width:240px;max-width:44vw;border-radius:4px;display:block}
.strip figcaption{color:#999;font-size:11px;text-align:center}
.cut{width:3px;align-self:stretch;background:#e34948;border-radius:2px}
.v{display:inline-block;padding:2px 10px;border-radius:4px;margin-left:8px;font-weight:600}
.v.y{background:#1e7d32}.v.n{background:#b3261e}.v.u{background:#8a6d00}
small{color:#999}
</style></head><body>
<div id="bar">Reviewer <input id="who" size="10"> <span id="prog"></span>
<button onclick="go(-1)">prev (k)</button><button onclick="go(1)">next (j)</button>
<button onclick="nextOpen()">next unreviewed (n)</button><button onclick="exp()">Export JSON</button>
<small>y real cut · x same shot · u unsure · r replay · j/k next/prev · n next unreviewed</small></div>
<div id="view"></div>
<script>
const ITEMS = __ITEMS__;
let rev = {}; try { rev = JSON.parse(localStorage.getItem('splitrev') || '{}'); } catch (e) {}
let i = 0;
const NAMES = {y: 'real cut', n: 'same shot', u: 'unsure'};
function save(){ try { localStorage.setItem('splitrev', JSON.stringify(rev)); } catch (e) {} }
function set(v){ rev[ITEMS[i].key] = v; save(); go(1); }
function go(d){ i = Math.max(0, Math.min(ITEMS.length - 1, i + d)); draw(); }
function nextOpen(){ for (let k = 1; k <= ITEMS.length; k++) { const j = (i + k) % ITEMS.length; if (!rev[ITEMS[j].key]) { i = j; return draw(); } } }
function replay(){ const v = document.getElementById('vid'), it = ITEMS[i]; v.currentTime = it.t0; v.play(); }
function draw(){
  const it = ITEMS[i], v = rev[it.key], cap = ['shot start', 'just before', 'just after', 'shot end'];
  document.getElementById('prog').textContent = `${i + 1}/${ITEMS.length} · reviewed ${Object.keys(rev).length}`;
  document.getElementById('view').innerHTML = `<h2>${it.film}${v ? `<span class="v ${v}">${NAMES[v]}</span>` : ''}</h2>
  <video id="vid" src="${it.video}#t=${it.t0},${it.t1}" controls muted preload="metadata"></video>
  <div class="strip">${it.ims.map((s, k) => (k == 2 ? '<div class="cut"></div>' : '') +
    `<figure><img src="${s}"><figcaption>${cap[k]}</figcaption></figure>`).join('')}</div>
  <p><small>proposed cut at ${it.cut}s (red line). Is it a real cut to a different shot, or the same shot
  (camera move, flash, fast action)?</small></p>`;
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'y' || e.key === 'u') set(e.key); else if (e.key === 'x') set('n');
  else if (e.key === 'j') go(1); else if (e.key === 'k') go(-1); else if (e.key === 'n') nextOpen();
  else if (e.key === 'r') replay();
});
function exp(){
  const who = document.getElementById('who').value || 'anon';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify({reviewer: who, verdicts: rev}, null, 1)], {type: 'application/json'}));
  a.download = `split_review_${who}.json`; a.click();
}
draw();
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["propose", "review", "report", "apply"])
    ap.add_argument("--core", action="store_true", help="only the original 293-film set")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-sim", type=float)
    a = ap.parse_args()
    if a.step == "propose":
        propose(a.core)
    elif a.step == "review":
        review(a.n)
    elif a.step == "report":
        report()
    else:
        if a.max_sim is None:
            ap.error("apply needs --max-sim (pick it from the report)")
        apply(a.max_sim)
