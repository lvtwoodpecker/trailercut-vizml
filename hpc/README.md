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

## Backbone trial (is a bigger CLIP worth it?)

Embeds only the 9 hand-labeled films and their trailers, then grades alignment against the labels
the same way `label_eval` does (`trailercut.backbone_trial`). B/32 on a laptop scores best F1 0.893
(precision 0.87, recall 0.92), and "text card" tracks "night" at rho 0.52 within a film. It reuses
the repo's trailer shot lists, so don't run `trailer_prep` for this.

On NYU Torch: nothing heavy runs on the login node (setup and jobs go through `sbatch`), submit with
`--export=NONE`, copy data through the transfer node `dtn.torch.hpc.nyu.edu`, keep temp files on
scratch (`env.sh` sets `TMPDIR`), and remember scratch is purged after 60 days without access and
never backed up (code lives in git, results get copied down).

```bash
# 1. cluster, login node: clone the branch, then set up as a job
cd /scratch/$USER && git clone -b h14-trial git@github.com:lvtwoodpecker/trailercut-vizml.git
cd trailercut-vizml && mkdir -p logs
sbatch --export=NONE --account=<slurm account> hpc/setup.sbatch

# 2. laptop, repo root: keyframes of the labeled films as one uncompressed zip per film (what the
#    readers expect, and ~9 files instead of ~40k), plus their trailer frames (~25 MB)
IDS="tt0032138 tt0049730 tt0052357 tt0061722 tt0065214 tt0067116 tt0086879 tt0119528 tt1411238"
U=<netid>; R=$U@dtn.torch.hpc.nyu.edu:/scratch/$U/trailercut/data
mkdir -p /tmp/tc_upload && for i in $IDS; do (cd data/movienet/frames/$i && zip -0qr /tmp/tc_upload/$i.zip .); done
rsync -av --progress /tmp/tc_upload/ $R/movienet/frames/
for i in $IDS; do rsync -a data/cache/trailer_frames/$i/ $R/cache/trailer_frames/$i/; done

# 3. cluster: run and read
sbatch --export=NONE --account=<slurm account> hpc/backbone_trial.sbatch                    # h14
sbatch --export=TRAILERCUT_CLIP=l14 --account=<slurm account> hpc/backbone_trial.sbatch  # only that var, like NONE
squeue -u $USER; tail -f logs/tc-trial_*.out
cat data/features/backbone_trial_h14.json
```

Worth a full rerun if F1 clearly beats 0.893 (say above 0.92) or the text card / night overlap drops.
