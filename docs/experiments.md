# Experiments log

Things we tried, what happened, and what we kept. Includes the ones that didn't work, so nobody
tries them again by accident. The reasoning behind what we kept is in `design.md`.

| date | tried | result | kept? |
|---|---|---|---|
| 2026-10-27 | Impostor threshold at 1% FPR, no blank handling | threshold 0.9975; fades and black frames match any film, almost nothing matched | no |
| 2026-10-27 | Blank masking (CLIP similarity > 0.9 to solid frames) on both sides | 1% threshold dropped to ~0.79 | yes |
| 2026-10-27 | 1% vs 5% FPR | contact sheets: matches between 0.72 and 0.79 almost all correct, so 1% lost real matches | 5% for now, hand labels decide |
| 2026-10-27 | Letterbox crop per frame | dark films got cropped wrongly: *The Godfather* 0.65 to 0.60 | no |
| 2026-10-27 | Letterbox crop per trailer (max over all frames) | *Rebel Without a Cause* 32% to 43% matched, no loss on unboxed trailers | yes |
| 2026-10-27 | CLIP B/32 vs B/16 vs L/14 speed on laptop GPU | 257 vs 72 vs 15 images/s | B/32 locally, H/14 on the cluster |
| 2026-10-27 | TransNetV2 on laptop GPU (MPS) vs CPU | MPS no faster (~37 s/trailer either way under load; ~15 s/trailer CPU when the machine is idle) | CPU, 4 processes |
| 2026-10-27 | New DataLoader per film for CLIP | worker startup dominated on macOS | one streaming loader for all films |
| 2026-10-27 | Augmented film shots as an alignment test (proposal) | measures robustness to edits, not correctness on real trailers | dropped for hand labels |
| 2026-10-28 | YouTube search for dead links, title filter only | 2nd hit for *Godfather Part II* is the *Godfather* anniversary trailer | added the 25% alignment check |
| 2026-10-28 | Committing `film_shots.parquet` | 65 MB, over GitHub's 50 MB warning, regenerated in a minute | gitignored |
| 2026-10-28 | Trailer View with only the curves from trailer shots to film position | with 70+ shots the curves cross into a tangle; order is hard to read | kept, plus an order scatter (trailer vs film position) |
| 2026-10-28 | First full RQ1 model run (293 films, LightGBM, grouped 5-fold CV) | full AP 0.050 vs random 0.024, readable 0.045; length-adjusted AP 0.046 (see `results.md`) | yes, as the baseline to improve on |
