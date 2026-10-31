"""A page for checking replaced trailers by eye, one film at a time.

Weakest matches first. For every film whose trailer is a replacement (source=search): the old and new YouTube title, the new
trailer itself (local file), and its 8 best-matching shots next to the film keyframes they matched.
Mark each film good (g), bad (b) or unsure (u); Export saves labels/swap_review_<name>.json.
Every replaced core trailer is on the page; of the films new to the dataset only a random sample (--new,
default 30), enough to estimate how often a new film's trailer is wrong.
Images are copied to labels/swap_img/ (gitignored, film frames); open the page from labels/.

  python -m trailercut.swap_review [--new 30]
"""
import argparse
import json
import shutil

import pandas as pd

from . import movienet_io as mn
from .paths import CACHE, FEATURES, IDS_DIR, LABELS
from .release import youtube_meta

PAGE = LABELS / "swap_review.html"
IMG = LABELS / "swap_img"
PAIRS = 8


def share(i):
    al = pd.read_csv(FEATURES / "alignment" / f"{i}.csv")
    return round(float(al.matched[~al.blank].mean()), 3)


def pairs(i):
    al = pd.read_csv(FEATURES / "alignment" / f"{i}.csv")
    al = al[al.matched & ~al.blank].sort_values("score", ascending=False)
    pick = []
    for x in al.itertuples():  # spread across the trailer: skip shots next to one already picked
        if all(abs(x.shot_idx - p.shot_idx) > 2 for p in pick):
            pick.append(x)
        if len(pick) == PAIRS:
            break
    d = IMG / i
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for x in sorted(pick, key=lambda p: p.shot_idx):
        t, f = d / f"t_{x.shot_idx:04d}.jpg", d / f"f_{int(x.film_shot):04d}.jpg"
        if not t.exists():
            shutil.copyfile(CACHE / "trailer_frames" / i / mn.keyframe_name(x.shot_idx, 1), t)
        if not f.exists():
            mn.open_image(mn.keyframe_paths(i, int(x.film_shot))[1]).convert("RGB").save(f, quality=85)
        out.append(dict(t=f"swap_img/{i}/{t.name}", f=f"swap_img/{i}/{f.name}", time=round(x.t_start, 1),
                        pos=round(float(x.film_story_pos), 2)))
    return out


def main(n_new):
    status = pd.read_csv(IDS_DIR / "trailer_status.csv", dtype=str)
    f = pd.read_csv(IDS_DIR / "films_1100.csv", dtype=str).set_index("imdb_id")
    films, core = f.youtube_id, set(f.index[f.core == "1"])
    meta = youtube_meta()
    swapped = status[(status.source == "search") & (status.status == "ok")]
    new = swapped[~swapped.imdb_id.isin(core)]
    pick = swapped[swapped.imdb_id.isin(core) | swapped.imdb_id.isin(new.sample(min(n_new, len(new)), random_state=0).imdb_id)]
    items = []
    for r in pick.itertuples():
        i, old = r.imdb_id, films.get(r.imdb_id)
        if not (FEATURES / "alignment" / f"{i}.csv").exists():
            continue
        old = old if isinstance(old, str) else ""
        new = meta.get(r.youtube_id, {})
        items.append(dict(id=i, film=mn.meta(i)["title"] + ("" if i in core else " (new film)"),
                          old=meta.get(old, {}).get("title") or ("dead link" if old else "no MovieNet link"),
                          old_url=f"https://www.youtube.com/watch?v={old}" if old else "",
                          new=new.get("title") or r.title, channel=new.get("channel") or "",
                          url=f"https://www.youtube.com/watch?v={r.youtube_id}", pairs=pairs(i), share=share(i)))
    items.sort(key=lambda x: x["share"])  # weakest matches first: the likeliest wrong trailers
    PAGE.write_text(HTML.replace("__ITEMS__", json.dumps(items)))
    print(f"{len(items)} replaced trailers -> {PAGE}")


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Trailer swaps</title>
<style>
body{font:14px system-ui;margin:16px;background:#151515;color:#eee}
a{color:#8ab4f8}
#bar{position:sticky;top:0;background:#151515;padding:8px 0;border-bottom:1px solid #333;margin-bottom:12px;z-index:2}
h2{margin:6px 0 2px;font-size:20px}
.old{color:#999;text-decoration:line-through}
.new{color:#cfe3ff}
.main{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
video{width:560px;max-width:100%;border-radius:6px;background:#000}
.pairs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;flex:1;min-width:520px}
.pair{display:flex;gap:4px;background:#222;padding:4px;border-radius:6px}
.pair img{width:50%;display:block;border-radius:3px}
.pair small{display:block;color:#999;font-size:11px}
.v{display:inline-block;padding:2px 10px;border-radius:4px;margin-left:8px;font-weight:600}
.v.g{background:#1e7d32}.v.b{background:#b3261e}.v.u{background:#8a6d00}
button{margin-right:4px}
small{color:#999}
</style></head><body>
<div id="bar">Reviewer <input id="who" size="10"> <span id="prog"></span>
<button onclick="go(-1)">prev (k)</button><button onclick="go(1)">next (j)</button>
<button onclick="nextOpen()">next unreviewed (n)</button><button onclick="exp()">Export JSON</button>
<small>g good · b bad · u unsure · j/k next/prev · n next unreviewed · space play/pause</small></div>
<div id="view"></div>
<script>
const ITEMS = __ITEMS__;
let rev = {}; try { rev = JSON.parse(localStorage.getItem('swaprev') || '{}'); } catch (e) {}
let i = 0;
const NAMES = {g: 'good', b: 'bad', u: 'unsure'};
function save(){ try { localStorage.setItem('swaprev', JSON.stringify(rev)); } catch (e) {} }
function set(v){ rev[ITEMS[i].id] = v; save(); go(1); }
function go(d){ i = Math.max(0, Math.min(ITEMS.length - 1, i + d)); draw(); }
function nextOpen(){ for (let k = 1; k <= ITEMS.length; k++) { const j = (i + k) % ITEMS.length; if (!rev[ITEMS[j].id]) { i = j; return draw(); } } }
function draw(){
  const it = ITEMS[i], v = rev[it.id];
  document.getElementById('prog').textContent = `${i + 1}/${ITEMS.length} · reviewed ${Object.keys(rev).length}`;
  document.getElementById('view').innerHTML = `<h2>${it.film}${v ? `<span class="v ${v}">${NAMES[v]}</span>` : ''}</h2>
  <div class="old">${it.old_url ? `<a href="${it.old_url}" target="_blank">${it.old}</a>` : it.old}</div>
  <div class="new"><a href="${it.url}" target="_blank">${it.new}</a> <small>${it.channel}</small></div>
  <div class="main" style="margin-top:10px"><video id="vid" src="../data/trailers/${it.id}.mp4" controls preload="metadata"></video>
  <div class="pairs">${it.pairs.map(p => `<div><div class="pair"><img src="${p.t}"><img src="${p.f}"></div>
    <small>trailer ${p.time}s · film ${Math.round(p.pos * 100)}%</small></div>`).join('')}</div></div>
  <p><small>left of each pair: the new trailer · right: the film keyframe it matched</small></p>`;
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if ('gbu'.includes(e.key)) set(e.key);
  else if (e.key === 'j') go(1); else if (e.key === 'k') go(-1); else if (e.key === 'n') nextOpen();
  else if (e.key === ' ') { e.preventDefault(); const v = document.getElementById('vid'); v.paused ? v.play() : v.pause(); }
});
function exp(){
  const who = document.getElementById('who').value || 'anon';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify({reviewer: who, verdicts: rev}, null, 1)], {type: 'application/json'}));
  a.download = `swap_review_${who}.json`; a.click();
}
draw();
</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=int, default=30, help="random sample of new films' trailers to include")
    main(ap.parse_args().new)
