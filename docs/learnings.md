# Learnings

Things we learned building this project, kept as notes. What the course lectures taught is in
[course_learnings.md](course_learnings.md); the decisions themselves are in [design.md](design.md) and
[viz_design.md](viz_design.md), and the tried-and-dropped list is in [experiments.md](experiments.md).

---

## Views and charts

- **Two colors can look different and not be.** Blue (picture) and violet (dialogue) collapsed under
  red-green colorblindness (ΔE 1.9), right where both kinds of link cross. Only the palette validator
  caught it. → Validate every color set, both themes.
- **A click that doesn't move the video breaks the view.** A highlight "lock" separate from the
  playhead (plus a strip click that seeked twice) made the picture and the chart disagree. → One source
  of truth for state (the video clock); every click seeks.
- **"Choppy" was a panel rebuilt on every mouse move.** → Compare, then re-render only on change.
- **Fixed axis ranges clip silently.** The EBM panels had a hard 0.55 to 2.3× range. → Domains from the
  data, shared across small multiples.
- **Faint grey "no data" hides the data.** → Leave it empty.
- **Look at the render.** Every issue above was obvious in a screenshot and invisible in the code.
  Headless Chrome screenshots in both themes are cheap. (Headless can't decode mp4, so the video box is
  black there; that's not a bug.)

## Data and checks

- **A strict accept rule rejects real items.** "25% of shots above the strict cutoff" rejected
  official trailers; even the Bourne Identity's accepted trailer only had 10%. → Test a rule on
  known-good cases before trusting its rejects; "matches this film better than other films" fixed it
  (56 of 117 rejected films accepted on re-check).
- **A metric can't grade its own fix.** CLIP flags glued shots and also picks where to cut, so the
  glued-shot rate drops partly by construction. → Propose, check a sample by eye, then apply.
- **If a score doesn't separate good from bad, no cutoff will.** The split review was ~80% real cuts at
  every similarity level. → Find a second signal (sharpness) or accept the rate.
- **48 checks is a wide interval** (77% could be 65 to 88%). → Bigger samples before refining.
- **A cutoff picked on the labels and scored on the same labels flatters itself.** → Compare methods
  with the cutoff re-picked inside each fold; the fancier classifiers lost.
- **Title filters must ignore the film's own title.** "vs." in the skip list removed every candidate
  for Freddy vs. Jason.
- **Audio language checks miss burned-in subtitles.** 12 Years a Slave was English audio with Spanish
  subtitles. → A `subtitled` flag from the title.
- **More films changed the story a bit.** At 1,058 films editing edged out CLIP as the biggest feature
  group; on the 293-film core CLIP led. → Re-check claims after scaling the data.

## Pipeline and jobs

- **Save per item, resume, never hold everything in memory.** Interrupted jobs (sleep, crash, Ctrl-C)
  then cost minutes, not hours.
- **Two jobs rewriting the same file undo each other.** An overnight batch rewrote the whole status
  file from its stale copy and reverted 12 swaps. → Each save re-reads the file and updates only its own
  rows, written aside and renamed.
- **"Already backed up" markers go stale.** Empty backup folders from finished restores made a later run
  skip its backup, then delete the originals. → A backup counts only if the files are actually there.
- **Scope shared steps to their own items.** `swap apply` ran shot detection on every pending trailer,
  including another job's. → `--ids`.
- **Cache expensive model outputs, keyed by content.** TransNetV2's cut probabilities, keyed by the
  video's size: re-splitting went from ~4 h to 21 min, and a restored trailer finds its old entry.
- **Parallel workers beat one process with many threads** for TransNetV2 on CPU (~4×).
- **Don't edit a script that's running; relaunch it.** And wait on PIDs, not on name patterns that can
  match the waiter itself.
- **Sleep pauses jobs, it doesn't kill them.** `caffeinate -s` keeps the Mac awake only on power.
- **Other apps can starve a long job.** 71 browser processes held ~24 GB during the RQ1 run.
- **Some GPU paths change results.** TransNetV2 on Apple's MPS gives different cuts than CPU, so it
  stays on CPU for consistency.
- **Benchmark against your own past runs.** The core re-split's logged time (55 s/trailer) corrected an
  estimate that was off by 2×.
