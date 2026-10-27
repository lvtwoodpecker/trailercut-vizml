"""Readable visual features from CLIP: zero-shot shot scale + content concepts.

These stand in for the raw embedding when explaining the trailerness model, so a SHAP bar says
"close-up" or "explosion" instead of "embedding dim 417". The raw embedding stays a separate,
ablatable feature block.

  python -m trailercut.concepts   # writes data/cache/concept_text.npy
"""
import numpy as np

from .paths import CACHE

TEXT_EMB = CACHE / "concept_text.npy"

# MovieNet's scale taxonomy (ECS/CS/MS/FS/LS); scale is read as a softmax over these five
SHOT_SCALE = {
    "scale_ecu": "an extreme close-up shot of a face or object",
    "scale_cu": "a close-up shot of a person's face",
    "scale_ms": "a medium shot of a person from the waist up",
    "scale_fs": "a full shot of a person's whole body",
    "scale_ls": "a wide long shot of a landscape or a building",
}

CONTENT = {
    "c_explosion": "an explosion with fire and smoke",
    "c_gun": "a person firing a gun",
    "c_fight": "people fighting",
    "c_chase": "a car chase",
    "c_running": "a person running",
    "c_kiss": "two people kissing",
    "c_crying": "a person crying",
    "c_scream": "a person screaming in fear",
    "c_laugh": "a person laughing",
    "c_crowd": "a large crowd of people",
    "c_night": "a dark scene at night",
    "c_day_outdoor": "a bright sunny outdoor scene",
    "c_creature": "a monster or creature",
    "c_space": "a spaceship in outer space",
    "c_water": "the ocean or water",
    "c_city": "a city skyline",
    "c_interior": "people talking in a room",
    "c_text_card": "white text on a black screen",
    "c_logo": "a movie studio logo",
    "c_blood": "blood and violence",
}

ALL = {**SHOT_SCALE, **CONTENT}
NAMES = list(ALL)


def text_embeddings():
    if TEXT_EMB.exists():
        return np.load(TEXT_EMB)
    from .embed import encode_text, load_model
    model, _, tok = load_model()
    e = encode_text(model, tok, [f"a movie still of {p}" for p in ALL.values()])
    np.save(TEXT_EMB, e)
    return e


def scores(emb):
    """emb [n,3,d] keyframe embeddings -> dict of per-shot readable features."""
    txt = text_embeddings()
    sims = (emb.astype(np.float32) @ txt.T).mean(1)  # [n, n_concepts], averaged over keyframes
    out = {}
    k = len(SHOT_SCALE)
    logits = 100 * sims[:, :k]
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    for j, name in enumerate(SHOT_SCALE):
        out[name] = p[:, j]
    # 0 = extreme close-up ... 4 = long shot; expected value is a single ordinal "how wide"
    out["scale_width"] = p @ np.arange(k)
    for j, name in enumerate(CONTENT, start=k):
        out[name] = sims[:, j]
    return out


if __name__ == "__main__":
    print(text_embeddings().shape)
