# Viz design system

The rules both D3 pages (`viz/index.html`, `viz/rq1.html`) and the static figures (`figures/*.png`)
follow, and each page's own design decisions (at the end); the lessons behind them are in
[learnings.md](learnings.md). Method and
pipeline decisions stay in [design.md](design.md).

## Tokens

Colors are CSS custom properties on `:root`, redefined for dark mode under
`@media (prefers-color-scheme: dark)` and `:root[data-theme="dark"]`. The figures use the same values
from `src/trailercut/style.py`. Never write a hex in a chart; use a token.

| token | light | dark | role |
|---|---|---|---|
| `--page` / `--surface` | `#f9f9f7` / `#fcfcfb` | `#0d0d0d` / `#1a1a19` | page, cards |
| `--ink` / `--ink-2` | `#0b0b0b` / `#52514e` | `#f0efec` / `#c3c2b7` | primary, secondary text |
| `--muted` | `#6f6d68` | `#898781` | axis ticks, captions (both clear 4.5:1 on the surface) |
| `--grid` / `--axis` | `#e1e0d9` / `#c3c2b7` | `#2c2c2a` / `#383835` | gridlines, axis rules (hairline, solid) |
| `--matched` = `--pos` | `#2a78d6` | `#3987e5` | series 1: matched, "toward the trailer", "picked more often" |
| `--used` | `#eb6834` | `#d95926` | film shots a trailer used |
| `--speech` | `#1baf7a` | `#199e70` | dialogue found in the film |
| `--neg` | `#e34948` | `#e66767` | "away / less often" (diverging pole only) |
| `--unmatched` | `#d6d5ce` | `#383835` | neutral fills (loudness area); never a "no data" mark |

Data-driven colors are the one exception: the trailer strip and the film barcode fill each shot with
its frame's average color.

## Color roles

- **Categorical (identity):** blue, orange, green, in that order. Validated together with
  `--pairs all` in both modes (worst colorblind ΔE 9.2 light / 9.4 dark, normal vision ≥ 20.9). A
  4th categorical hue on one chart needs a new validation run, or fold it into "other".
- **Diverging (direction):** blue vs red, with grey (`--muted`) when the interval crosses 1×. Grey
  means "no clear difference", not a third category.
- **Sequential (magnitude):** one hue, light to dark (`style.SEQ`, blue). No rainbow or viridis.
- **Text never wears a data color.** Values, labels and legends use the text tokens; a colored dot or
  swatch beside the text carries identity.
- **Absence beats a faint gray.** "No match" is an empty cell (legend: outline swatch), not a 1.4:1
  gray block.
- **Check, don't eyeball.** Run the palette validator from the dataviz skill on any new color set, in
  both modes, before shipping.

## Marks

| mark | spec |
|---|---|
| bars | ≤ 24px thick; 4px rounded data end, square at the baseline (`bar()` helper in both pages) |
| lines | 2px; curves of many links 1.2px at 35% opacity, the highlighted one 2.5px solid |
| dots | r ≥ 4 with a 2px ring in the surface color; the current one gets an ink ring |
| areas | the series hue as a light wash (≈ 10 to 20% opacity), never a saturated block |
| touching cells | a 2px surface gap, no borders |
| grid / axes | 1px solid hairlines; dashes only for reference lines (1×, "in film order", "same moment") |
| secondary channel | dashed = sound (dialogue link) vs solid = picture (match links) |

## Text

- **Label, don't explain.** Short titles and labels ("Shot 6 · 0:37"); explanations live in the docs.
- **Subtitles come from the data**, so they can't drift after a rerun (e.g. the strongest effect whose
  interval excludes 1×).
- `tabular-nums` only where numbers line up (axes, tables, clocks).
- Escape anything that comes from data before it goes into HTML (transcripts, concept names).

## Interaction

- **One source of truth for state.** On the Trailer View it is the video clock: every click seeks,
  every highlight follows the time. A "lock" that's separate from the playhead is how the view
  drifted out of sync before.
- **Hover previews only while paused**; leaving the mark returns to the current state.
- **Hit targets ≥ 24px** (transparent circles behind 9px dots, full-row targets on dot plots).
- **Crosshair on lines** (hairline + dot on the curve), per-mark tooltips on bars and dots; the value
  leads the tooltip, the label follows.
- **Keyboard:** space or `k` play, `←`/`→` one shot, `⇧←`/`⇧→` 5 s, `c` captions, `m` sound, Esc. A
  focused button keeps its own space/Enter; buttons don't take focus on mouse click.
- **Review pages share keys:** `j`/`k` next/prev, `n` next unreviewed, then one verdict letter per
  choice (`x` = leave it as is: keep the trailer, don't split).
- **Deep links:** `?id=` and `?t=` open a trailer at a moment (for the report and screenshots).

## Accessibility

- Every chart has a **table twin** (collapsed `Table` under it); a value is never hover-only.
- A legend for every chart with ≥ 2 colors; color is never the only channel (dash, outline, position).
- Dark mode is its own token set, validated against the dark surface, not an automatic flip.

## Layout

- Cards on the page background, 8px radius, one hairline border.
- On the Trailer View, the video and the "now" panel sit side by side above the timeline, so the
  playing video and its highlight are on screen together at ~900px height; video capped at 42vh.
- Reserve space for media (`aspect-ratio: 16/9`) so switching films doesn't jump the layout.
- Desktop first (the timeline needs ~560px); no phone layout.

## Before shipping a view

1. Palette validator passes in light and dark (`--pairs all` for scatters).
2. Render it headless in both themes and look (label collisions, clipped axes, empty states).
3. Every chart: legend if ≥ 2 colors, table twin, hit targets ≥ 24px, keyboard path to its values.
4. No DOM rebuild on `pointermove`; render only when the hovered or current item changes.
5. Text from data is escaped; no frames or video in anything committed.

## Lessons

Moved to [learnings.md](learnings.md) (views and charts).

## Page decisions

### Trailer View (`viz/index.html`, `viewer_export.py`)

**One JSON per trailer.** The page loads only the trailer it shows plus a small index for the picker.
Named-feature detail (value + SHAP per feature) is only exported for in-trailer shots and the top 60 by
model score, the shots someone will actually hover; the rest get grouped SHAP only, which keeps files
small.

**Movie barcode from average color.** The film is drawn as a strip of its shots, each filled with the
average color of its middle keyframe and as wide as its share of runtime. It gives the film axis a
recognizable look (night scenes, color grading, act changes) without showing any frames, so it works
in the committed page too. Average color is cheap (JPEG decoded at reduced size, ~1 ms per keyframe)
and cached per film in `cache/shot_colors/`. We give up any notion of shot content; the hover frames
cover that.

**Links plus an order scatter.** Curves from each trailer shot down to its film position show where
footage comes from, but with 70+ shots they cross into a tangle that's hard to read for order. The
scatter (trailer position vs film position, diagonal = film order) answers "does the trailer follow the
film?" at a glance, so both are shown.

**No trailerness chart.** We first drew the model's score as a bar per film shot, then replaced it
with a "Would the model have picked the same shots?" panel (top-k thumbnails, rank of each used shot,
hits vs random). Neither told a clear story (~700 thin bars per film were hard to read), so both were dropped. The model score now shows in the
film-shot tooltip and in the "why this shot" bars of the "now" panel. We give up a view of the score profile along the film.

**The video clock is the single source of truth.** Every highlight (current shot, its link, the film
outline, the scatter dot, the "now" panel) follows the video's time, and every click seeks the video:
the trailer strip and sound track scrub, a film barcode shot the trailer uses jumps to the trailer
shot that uses it (again for its next use), and the order and sync dots seek to their moment. Hover
previews a shot only while paused; leaving the mark goes back to the shot under the playhead. A film
shot the trailer doesn't use just fills the panel (Esc clears it). Before, hover previewed and a click
locked a shot without moving the video (and a strip click seeked twice), so the video and the
highlight could disagree; we gave up the click lock for one rule: what you see is where the video is.
The playhead runs on `requestVideoFrameCallback` where the browser has it, so it tracks the frame on
screen.

**Video and "now" panel side by side, above the timeline.** The panel shows the current trailer frame
next to the film frame it matched and the "why this shot" bars, so judging a match needs no extra
click. Keyboard: space or `k` play/pause, left/right step one shot, shift+left/right 5 s, Home to the
start, `c` captions, `m` sound, Esc. `?t=` opens a trailer at a moment, so a case can be linked.

**Thumbnails are local only.** Frame images load from the viewer's own `data/movienet/frames/` and
`data/cache/trailer_frames/`. Frames are copyrighted, so they are never committed, exported into
`viz/data/` or hosted; on a machine without the data the page still works, just without pictures.
The paths are fixed to `data/` and loose JPEGs, so they ignore `TRAILERCUT_DATA` and `--pack` zips
(unlike the label tool, which copies its images).

**Playable trailer with a custom player.** No native control bar: a play/pause button, the time, CC
and sound toggles, click the video to play/pause, and drag the trailer strip or sound track to scrub. Trailers
are AV1, VP9 or H.264 in MP4, so Safari can't play all of them; Chrome and Firefox can.
The native controls look different in every browser and duplicate the timeline; with our own controls the trailer strip itself is the seek bar.

**Captions from the transcript, on by default** (`viz/data/{id}.vtt`). Whisper can merge ~30 s of
speech into one segment, so `viewer_export` splits cues at about 6 s, 12 words or a pause over 1.5 s,
using the word timestamps.

**One sound track.** Loudness as an area, rescaled to its 10th percentile so quiet ends don't flatten
it, tinted green where a line was found in the film's dialogue (it was violet until a palette check showed blue vs violet collapses under red-green colorblindness: ΔE 1.9 to 4.5; blue, orange and green pass). One sound link at a time runs from the
current or hovered line to its film moment; picture links are drawn only in the main chart. A "Sound
vs picture" panel plots each found line's film time against its picture's film time and counts lines
within 30 s. Drawing every sound link next to every picture link was too crowded to read, so sound shows on demand and the sync panel carries the overview.

**Terse labels, no act labels** (also on the RQ1 page): "Keyframe 249", "Shot 7 · 0:39", short
subtitles, no setup/middle/ending marks. The design language is to label things, not explain them; explanations live in these docs.

### RQ1 page (`viz/rq1.html`, `rq1_export.py`)

**One page across all films**, linked from the Trailer View header. The Trailer View answers "what did
this trailer take"; this page answers "what kind of shot gets picked" over the whole corpus.

**Everything is a lift**: pick rate of a kind of shot over the pick rate of the average shot. One
unit for story position, shot scale, within-film shot length, distinctiveness, dialogue and 20
content concepts, so panels compare directly. Shot length and distinctiveness are binned by
within-film percentile (`rel_*`), and a concept counts as present in the film's top 10% for it, since
raw CLIP scores and shot lengths shift from film to film.

**Shared log scale, 0.5x to 2x** (clamped). Log so 0.5x and 2x sit the same distance from 1x; shared
so a long bar in one panel means the same as in another.

**95% intervals from resampling films, not shots.** Shots within a film aren't independent; resampling
films keeps one film from carrying a result.

**Grey when the interval crosses 1x**, blue above, red below. Color then only marks effects that hold up.

**Subtitles are written from the data.** Each panel's line quotes only the strongest effect above and
below 1x whose interval excludes 1x, or says there is no clear difference. Text can't drift from the
numbers after a rerun.

**Model lift bars** reuse `model_eval.json` (exact-shot lift) so the page shows how far the features
get together, next to the single-feature lifts.

**Effect of each feature: EBM curves** (from `model_ebm.json`). The lifts above look at one feature at
a time; the EBM curves show each feature's effect with the others held fixed. The page shows the 6
curves with the highest importance, as odds multipliers on one shared log scale (same reason as the
lift bars), its range taken from the data so no curve is clipped, cut to the feature's 1st to 99th percentile since the outer bins hold few shots. The
section is hidden when `model_ebm.json` doesn't exist.
