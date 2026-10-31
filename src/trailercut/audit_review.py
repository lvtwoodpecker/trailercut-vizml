"""A page for checking the audit's flags by eye: one flagged trailer at a time, playable, with a
swap / keep verdict. Export saves labels/audit_review_<name>.json; `swap search` then replaces the
films marked swap, and keeps the ones marked keep even if a rule flagged them (decisions()).

  python -m trailercut.audit_review
"""
import json

import pandas as pd

from .paths import IDS_DIR, LABELS

PAGE = LABELS / "audit_review.html"


def decisions():
    """Films to replace: the audit's `swap` rows, overridden by any exported verdicts (later files win)."""
    audit = pd.read_csv(IDS_DIR / "trailer_audit.csv")
    swap = dict(zip(audit.imdb_id, audit["swap"]))
    for p in sorted(LABELS.glob("audit_review_*.json")):
        swap.update({i: v == "swap" for i, v in json.loads(p.read_text())["verdicts"].items() if v in ("swap", "keep")})
    return [i for i, s in swap.items() if s]


def main():
    a = pd.read_csv(IDS_DIR / "trailer_audit.csv", dtype=str).fillna("")
    a = a[a["flags"] != ""]
    items = [dict(id=r.imdb_id, film=r.film, title=r.yt_title, channel=r.channel, flags=r._asdict()["flags"].split(),
                  rule=r._asdict()["swap"] == "True", lang=r.language, want=r.expected_language, span=r.span,
                  url=f"https://www.youtube.com/watch?v={r.youtube_id}") for r in a.itertuples()]
    PAGE.write_text(HTML.replace("__ITEMS__", json.dumps(items)))
    print(f"{len(items)} flagged trailers -> {PAGE}")


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Trailer audit</title>
<style>
body{font:14px system-ui;margin:16px;background:#151515;color:#eee}
a{color:#8ab4f8}
#bar{position:sticky;top:0;background:#151515;padding:8px 0;border-bottom:1px solid #333;margin-bottom:12px}
h2{margin:6px 0 4px;font-size:20px}
.f{display:inline-block;padding:1px 8px;border-radius:10px;margin-right:4px;background:#444;font-size:12px}
.f.hard{background:#7a2a26}
.chip{cursor:pointer;margin-right:4px}.chip.on{outline:2px solid #8ab4f8}
video{width:560px;max-width:100%;border-radius:6px;background:#000;display:block;margin:10px 0}
.v{display:inline-block;padding:2px 10px;border-radius:4px;margin-left:8px;font-weight:600}
.v.swap{background:#b3261e}.v.keep{background:#1e7d32}
small{color:#999}
</style></head><body>
<div id="bar">Reviewer <input id="who" size="10"> <span id="prog"></span>
<button onclick="exp()">Export JSON</button><br><span id="chips"></span>
<small>s swap · x keep · j/k next/prev · n next unreviewed · space play</small></div>
<div id="view"></div>
<script>
const ALL = __ITEMS__, HARD = ['not_trailer', 'ad', 'clip', 'dub', 'subtitled', 'short'];
let rev = {}; try { rev = JSON.parse(localStorage.getItem('auditrev') || '{}'); } catch (e) {}
let flag = null, i = 0, items = ALL;
function save(){ try { localStorage.setItem('auditrev', JSON.stringify(rev)); } catch (e) {} }
function chips(){
  const n = f => ALL.filter(x => x.flags.includes(f)).length, fs = [...new Set(ALL.flatMap(x => x.flags))];
  document.getElementById('chips').innerHTML = [`<span class="f chip ${flag ? '' : 'on'}" onclick="pick(null)">all ${ALL.length}</span>`]
    .concat(fs.map(f => `<span class="f chip ${HARD.includes(f) ? 'hard' : ''} ${flag == f ? 'on' : ''}" onclick="pick('${f}')">${f} ${n(f)}</span>`)).join('');
}
function pick(f){ flag = f; items = f ? ALL.filter(x => x.flags.includes(f)) : ALL; i = 0; chips(); draw(); }
function go(d){ i = Math.max(0, Math.min(items.length - 1, i + d)); draw(); }
function set(v){ rev[items[i].id] = v; save(); go(1); }
function nextOpen(){ for (let k = 1; k <= items.length; k++) { const j = (i + k) % items.length; if (!rev[items[j].id]) { i = j; return draw(); } } }
function draw(){
  const it = items[i], v = rev[it.id];
  document.getElementById('prog').textContent = `${i + 1}/${items.length} · reviewed ${Object.keys(rev).length}/${ALL.length}`;
  document.getElementById('view').innerHTML = `<h2>${it.film}${v ? `<span class="v ${v}">${v}</span>` : ''}</h2>
  ${it.flags.map(f => `<span class="f ${HARD.includes(f) ? 'hard' : ''}">${f}</span>`).join('')}
  <small>rule says ${it.rule ? 'swap' : 'keep'}</small>
  <div style="margin-top:6px"><a href="${it.url}" target="_blank">${it.title}</a> <small>${it.channel}</small></div>
  <small>heard ${it.lang || '?'} · expected ${it.want || '?'} · covers ${Math.round(it.span * 100)}% of the film</small>
  <video id="vid" src="../data/trailers/${it.id}.mp4" controls preload="metadata"></video>`;
}
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 's') set('swap'); else if (e.key === 'x') set('keep');
  else if (e.key === 'j') go(1); else if (e.key === 'k') go(-1); else if (e.key === 'n') nextOpen();
  else if (e.key === ' ') { e.preventDefault(); const v = document.getElementById('vid'); v.paused ? v.play() : v.pause(); }
});
function exp(){
  const who = document.getElementById('who').value || 'anon';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify({reviewer: who, verdicts: rev}, null, 1)], {type: 'application/json'}));
  a.download = `audit_review_${who}.json`; a.click();
}
chips(); draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
