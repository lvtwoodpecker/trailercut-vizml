# Running preprocessing on a Slurm cluster

Only the heavy steps run on the cluster: keyframe download, TransNetV2 shot detection and CLIP
embeddings. Everything after that (alignment features, clustering, figures, labeling) runs on a laptop
in minutes. The default backbone on the cluster is CLIP ViT-H/14, matching Argaw et al.

## 0. Set your account

Replace `CHANGE_ME` in both `.sbatch` files with your Slurm account, and check the scratch path in
`env.sh` (defaults to `$SCRATCH`, else `/scratch/$USER`). GPU partition flags vary by cluster; add
`#SBATCH --partition=...` or a GPU type (e.g. `--gres=gpu:a100:1`) if yours needs it.

## 1. Clone and set up (login node)

```bash
cd /scratch/$USER          # or wherever you keep code
git clone git@github.com:lvtwoodpecker/trailercut-vizml.git && cd trailercut-vizml
mkdir -p logs
bash hpc/setup.sh          # venv on scratch, MovieNet metadata, CLIP + TransNetV2 weights
```

## 2. Copy trailers up (from your laptop)

YouTube tends to block cluster IPs, so download locally and copy the videos up (~1.3 GB).

```bash
# laptop, from the repo root
rsync -av --progress data/trailers/ <user>@<cluster>:/scratch/<user>/trailercut/data/trailers/
```

## 3. Keyframes

```bash
sbatch hpc/fetch_keyframes.sbatch
```

Stores one uncompressed zip per film (~880 files instead of ~1.3M JPEGs). If compute nodes have
no internet, run the python line from the script on a login/transfer node inside `tmux` instead.

## 4. Shots + embeddings (GPU)

```bash
sbatch hpc/gpu_preprocess.sbatch                     # h14 by default
TRAILERCUT_CLIP=l14 sbatch hpc/gpu_preprocess.sbatch # or another backbone; caches don't collide
```

Writes `data/features/trailer_shots/` and `data/features/alignment/` in the repo checkout (commit
them from there or copy them down), and embeddings to `$TRAILERCUT_DATA/cache/clip/<tag>/`.

## 5. Copy results down (from your laptop)

```bash
R=<user>@<cluster>:/scratch/<user>/trailercut
rsync -av --progress $R/data/cache/clip/h14/        data/cache/clip/h14/
rsync -av --progress $R/data/cache/trailer_frames/  data/cache/trailer_frames/
rsync -av --progress $R/data/cache/trailer_audio/   data/cache/trailer_audio/
rsync -av $R/data/cache/*_h14.npy                   data/cache/
# trailer_shots + alignment CSVs: git pull if committed on the cluster, else rsync data/features/
```

Then locally:

```bash
export TRAILERCUT_CLIP=h14
python -m trailercut.film_features && python -m trailercut.trailer_features
python -m trailercut.cluster && python -m trailercut.figures
python -m trailercut.label_tool --n 10   # needs film keyframes locally for the sampled films
```

Shot boundaries from GPU TransNetV2 can differ slightly from CPU runs, so copy `trailer_frames`
and `trailer_audio` down together with `trailer_shots` instead of mixing them with local ones.
