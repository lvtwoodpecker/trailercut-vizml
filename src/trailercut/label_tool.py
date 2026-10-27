"""Build a static page for hand-checking trailer-to-film matches.

Samples trailers stratified by decade and lists every shot with its top-5 film candidates.
Open labels/label_tool.html locally (images load from data/, which stays on your machine),
label with the keyboard, and hit Export: save the JSON as labels/gold_<name>.json.

  python -m trailercut.label_tool [--n 10] [--seed 0]
"""
import argparse
import json

import pandas as pd

from . import movienet_io as mn
from .paths import FEATURES, LABELS

PAGE = LABELS / "label_tool.html"


def sample(n, seed):
    ids = sorted(p.stem for p in (FEATURES / "alignment").glob("*.csv"))
    meta = pd.DataFrame([mn.meta(i) for i in ids])
    meta["decade"] = meta.year // 10 * 10
    # round-robin over decades so old and new trailers are both checked
    groups = [g.sample(frac=1, random_state=seed).imdb_id.tolist() for _, g in meta.groupby("decade")]
    out = []
    while len(out) < n and any(groups):
        for g in groups:
            if g and len(out) < n:
                out.append(g.pop(0))
    return out


def items(imdb_id):
    al = pd.read_csv(FEATURES / "alignment" / f"{imdb_id}.csv")
    title = mn.meta(imdb_id)["title"]
    for r in al.itertuples():
        cands = list(map(int, r.top5.split()))
        scores = list(map(float, r.top5_scores.split()))
        yield dict(
            key=f"{imdb_id}:{r.shot_idx}", imdb_id=imdb_id, title=title, shot=r.shot_idx,
            trailer=[f"../data/cache/trailer_frames/{imdb_id}/shot_{r.shot_idx:04d}_img_{i}.jpg" for i in range(3)],
            cands=[dict(shot=c, score=s, img=f"../data/movienet/frames/{imdb_id}/shot_{c:04d}_img_1.jpg")
                   for c, s in zip(cands, scores)],
            score=r.score, matched=bool(r.matched), blank=bool(r.blank),
        )


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Alignment labels</title>
<style>
body{font:14px system-ui;margin:16px;background:#fafafa;color:#222}
.row{display:flex;gap:8px;align-items:flex-start}
.t img{width:200px;display:block;margin-bottom:4px}
.c{width:200px;border:3px solid transparent;cursor:pointer}
.c img{width:100%;display:block}
.c.sel{border-color:#2a7}
.btns button{margin:4px 4px 0 0}
.btns button.sel{background:#2a7;color:#fff}
#bar{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd;margin-bottom:12px}
small{color:#777}
</style></head><body>
<div id="bar">Labeler <input id="who" size="10"> <span id="prog"></span>
<button onclick="go(-1)">prev (k)</button><button onclick="go(1)">next (j)</button>
<button onclick="exp()">Export JSON</button>
<small>keys: 1-5 pick candidate, s same scene/other shot, n not in film, u unsure, j/k next/prev</small></div>
<div id="view"></div>
<script>
const ITEMS = __ITEMS__;
let lab = {}; try { lab = JSON.parse(localStorage.getItem('lab') || '{}'); } catch (e) {}
let i = 0;
const OPTS = [['s', 'same scene, other shot'], ['n', 'not in film'], ['u', 'unsure']];
function save(){ try { localStorage.setItem('lab', JSON.stringify(lab)); } catch (e) {} }
function set(v){ lab[ITEMS[i].key] = v; save(); go(1); }
function go(d){ i = Math.max(0, Math.min(ITEMS.length - 1, i + d)); draw(); }
function draw(){
  const it = ITEMS[i], cur = lab[it.key];
  document.getElementById('prog').textContent = `${i + 1}/${ITEMS.length} labeled ${Object.keys(lab).length}`;
  document.getElementById('view').innerHTML = `<h3>${it.title} <small>trailer shot ${it.shot} | best ${it.score.toFixed(3)}${it.blank ? ' | blank' : ''}</small></h3>
  <div class="row"><div class="t"><b>trailer</b>${it.trailer.map(s => `<img src="${s}">`).join('')}</div>
  ${it.cands.map((c, k) => `<div class="c ${cur === String(c.shot) ? 'sel' : ''}" onclick="set('${c.shot}')"><b>${k + 1}</b> film shot ${c.shot} <small>${c.score.toFixed(3)}</small><img src="${c.img}"></div>`).join('')}</div>
  <div class="btns">${OPTS.map(([k, t]) => `<button class="${cur === k ? 'sel' : ''}" onclick="set('${k}')">${k}: ${t}</button>`).join('')}</div>`;
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  const it = ITEMS[i];
  if (e.key >= '1' && e.key <= '5') set(String(it.cands[+e.key - 1].shot));
  else if ('snu'.includes(e.key)) set(e.key);
  else if (e.key === 'j') go(1); else if (e.key === 'k') go(-1);
});
function exp(){
  const who = document.getElementById('who').value || 'anon';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify({labeler: who, labels: lab}, null, 1)], {type: 'application/json'}));
  a.download = `gold_${who}.json`; a.click();
}
draw();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ids = sample(a.n, a.seed)
    its = [x for i in ids for x in items(i)]
    PAGE.write_text(HTML.replace("__ITEMS__", json.dumps(its)))
    (LABELS / "label_sample.json").write_text(json.dumps(ids))
    print(f"{len(its)} trailer shots from {len(ids)} trailers -> {PAGE}")


if __name__ == "__main__":
    main()
