"""Does a bigger CLIP align better? Score one backbone on the hand-labeled films only.

Runs align_one on the labeled films with the backbone in TRAILERCUT_CLIP (embeddings must exist for
those films) and grades it exactly like label_eval: best-F1 cutoff, precision and recall there, top-1
accuracy. Also one concept check: within a film, how much "text card" tracks "night" (Spearman over
film shots, mean over films). B/32 reads dark frames as text; a sharper backbone should lower this.

Uses the repo's trailer shot lists (no shot detection), so labels stay valid.

  TRAILERCUT_CLIP=h14 python -m trailercut.backbone_trial
Writes data/features/backbone_trial_{tag}.json
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import concepts
from .align import align_one, load
from .label_eval import evaluate, load_gold, one_per_shot
from .paths import CLIP_TAG, FEATURES


def main():
    gold = one_per_shot(load_gold())
    ids = sorted(gold.imdb_id.unique())
    al = pd.concat([align_one(i, threshold=0.0).assign(imdb_id=i) for i in ids])
    ev, *_ = evaluate(gold, al)
    rho = []
    for i in ids:
        s = concepts.scores(load("film", i).numpy())
        rho.append(spearmanr(s["c_text_card"], s["c_night"])[0])
    ev.update(backbone=CLIP_TAG, films=ids, text_card_vs_night_rho=float(np.mean(rho)))
    (FEATURES / f"backbone_trial_{CLIP_TAG}.json").write_text(json.dumps(ev, indent=2))
    b = ev["best_f1"]
    print(f"{CLIP_TAG}: {ev['n']} labels, best F1 {b['f1']:.3f} at {b['threshold']:.3f} (P {b['precision']:.2f}, "
          f"R {b['recall']:.2f}); top-1 {ev['top1_acc_when_source']:.3f}; text card vs night rho {ev['text_card_vs_night_rho']:.2f}")


if __name__ == "__main__":
    main()
