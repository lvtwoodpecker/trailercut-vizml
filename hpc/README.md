# Running preprocessing on a Slurm cluster

Only the heavy steps run on the cluster: keyframe download, TransNetV2 shot detection and CLIP
embeddings. Everything after that (alignment features, clustering, figures, labeling) runs on a laptop
in minutes. The backbone is CLIP ViT-B/32 everywhere: H/14 (Argaw et al.) matched worse in the
backbone trial below (docs/design.md).

## 0. Rules on the cluster (NYU Torch)

From NYU's HPC docs and what has worked in our other Torch projects:

- **Work under `$SCRATCH`, never `$HOME`.** Clone the repo there; everything the pipeline needs
  (data, venv, caches, logs) then lives inside that one checkout (`env.sh`), all gitignored.
- **No work on the login node.** It is only for git, editing and submitting/watching jobs. Setup, installs, downloads and
  every Python run go through `sbatch` (or `srun --pty` for interactive debugging, which lands on a
  compute node). For a one-off check, write the script to a file on scratch and submit that, instead
  of a long `python -c "..."` with fragile quoting.
- **Submit with `sbatch --export=NONE`**, so your login shell's variables don't leak into the job;
  the jobs `source hpc/env.sh` themselves. To pass one variable, use `--export=VAR=value` (that
  also exports nothing else).
- **Give the Slurm account on the command line:** `sbatch --export=NONE --account=<account> ...`.
  Accounts are per project or course; `sacctmgr show assoc user=$USER format=account%30` lists yours.
  Don't borrow another project's.
- **Temp files: never the login node's `/tmp`** (it is not the compute node's). Inside a job,
  `env.sh` points `TMPDIR` at the node-local `$SLURM_TMPDIR`; PyTorch worker temp files on scratch's
  network filesystem fail to delete at exit ("Device or resource busy"). Outside a job, `.cache/tmp`.
- **Copy data through a transfer node**, not the login node. `dtn.torch.hpc.nyu.edu` lands on either
  `dtn011.hpc.nyu.edu` or `dtn012.hpc.nyu.edu`, which have different host keys, so ssh warns that the
  key "changed" when it switches. Pick one node by name and use it every time. Before accepting its
  key the first time, compare with what the cluster sees: on the login node,
  `ssh-keyscan -t ed25519 dtn012.hpc.nyu.edu 2>/dev/null | ssh-keygen -lf -`. Each new connection asks
  for a device login, so copy everything in one `rsync`.
- **Storage** (check with `myquota`):

  | where | for | limit | backed up | purged |
  |---|---|---|---|---|
  | `$HOME` | small config | 50 GB, 30k files | yes | no |
  | `$SCRATCH` | data, venv, caches, this repo's checkout | 5 TB, 5M files | **no** | **after 60 days without access** |
  | `$ARCHIVE` | finished data as one tar (login nodes only) | 2 TB, 20k files | yes | no |

  Code lives in git and results get copied down, so a purge only costs a re-download. Touching files
  to dodge the purge is a policy violation. No environments or model caches in `$HOME` (`env.sh`
  puts the venv, uv, Hugging Face and torch caches on scratch).
- **Few big files, not many small ones.** Keyframes stay one uncompressed zip per film, read in place.
- **Data rules.** HPC is approved for moderate-risk data only. Film keyframes, trailers and frames are
  copyrighted: they stay in your own scratch, never in a shared or public location.
- **GPU jobs** ask for `--gres=gpu:1 --constraint="l40s|h200" --requeue`; logs go to
  `logs/<job>_<id>.out` and `.err`.
- **Watching jobs:** `squeue -u $USER`, `tail -f logs/<job>_*.out`, and after a run
  `sacct -j <id> --format=JobID,Elapsed,MaxRSS,State` or `seff <id>` to right-size time and memory.
  Chain steps with `sbatch --parsable` and `--dependency=afterok:<id>`.

Layout on the cluster: one checkout anywhere under `$SCRATCH` (the examples use
`$SCRATCH/trailercut-vizml`; a project folder like `$SCRATCH/<project>/trailercut-vizml` works the
same), with `data/` (MovieNet, keyframes, trailers, embeddings), `.venv/`, `.cache/` (model weights,
package caches, temp files) and `logs/` inside it, the same layout as on a laptop. `env.sh` finds all
of it from the checkout, so only the rsync targets need the real path: run `pwd` in the checkout on
the cluster and use that.

## 1. Clone and set up

```bash
# login node (git only)
cd $SCRATCH
git clone https://github.com/lvtwoodpecker/trailercut-vizml.git && cd trailercut-vizml
mkdir -p logs
sbatch --export=NONE --account=<account> hpc/setup.sbatch   # venv on scratch, MovieNet metadata, CLIP + TransNetV2 weights
```

`setup.sbatch` runs `setup.sh` as a job and needs internet on the compute node. If it fails on
downloads, ask before falling back to anything on the login node.

## 2. Copy trailers up (from your laptop)

YouTube tends to block cluster IPs, so download locally and copy the videos up (~1.3 GB).

```bash
# laptop, from the repo root
rsync -av --progress data/trailers/ <netid>@dtn012.hpc.nyu.edu:/scratch/<netid>/trailercut-vizml/data/trailers/
```

## 3. Keyframes

```bash
sbatch --export=NONE --account=<account> hpc/fetch_keyframes.sbatch
```

Stores one uncompressed zip per film (~880 files instead of ~1.3M JPEGs). It needs internet on the
compute node; if that fails, ask HPC support or run the download on the transfer node.

## 4. Shots + embeddings (GPU)

```bash
sbatch --export=NONE --account=<account> hpc/gpu_preprocess.sbatch               # b32 by default
sbatch --export=TRAILERCUT_CLIP=l14 --account=<account> hpc/gpu_preprocess.sbatch # another backbone; caches don't collide
```

Writes `data/features/trailer_shots/` and `data/features/alignment/` in the repo checkout (commit
them from there or copy them down), and embeddings to `$TRAILERCUT_DATA/cache/clip/<tag>/`.

## 5. Copy results down (from your laptop)

```bash
R=<netid>@dtn012.hpc.nyu.edu:/scratch/<netid>/trailercut-vizml
rsync -av --progress $R/data/cache/clip/b32/        data/cache/clip/b32/
rsync -av --progress $R/data/cache/trailer_frames/  data/cache/trailer_frames/
rsync -av --progress $R/data/cache/trailer_audio/   data/cache/trailer_audio/
rsync -av $R/data/cache/*_b32.npy                   data/cache/
# trailer_shots + alignment CSVs: git pull if committed on the cluster, else rsync data/features/
```

Then locally:

```bash
export TRAILERCUT_CLIP=b32
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
cd $SCRATCH && git clone https://github.com/lvtwoodpecker/trailercut-vizml.git
cd trailercut-vizml && mkdir -p logs
sbatch --export=NONE --account=<slurm account> hpc/setup.sbatch

# 2. laptop, repo root: stage keyframes of the labeled films as one uncompressed zip per film (what
#    the readers expect, ~9 files instead of ~40k) plus their trailer frames (~25 MB), then copy it
#    all in one rsync (one login). The array form works in both bash and zsh.
IDS=(tt0032138 tt0049730 tt0052357 tt0061722 tt0065214 tt0067116 tt0086879 tt0119528 tt1411238)
U=/tmp/tc_upload; rm -rf $U && mkdir -p $U/movienet/frames $U/cache/trailer_frames
for i in "${IDS[@]}"; do
  (cd data/movienet/frames/$i && zip -0qr $U/movienet/frames/$i.zip .)
  cp -R data/cache/trailer_frames/$i $U/cache/trailer_frames/
done
rsync -av --progress $U/ <netid>@<dtn node>:<checkout path from pwd on the cluster>/data/
rm -rf $U

# 3. cluster: run and read
sbatch --export=TRAILERCUT_CLIP=h14 --account=<slurm account> hpc/backbone_trial.sbatch  # backbone to test
sbatch --export=TRAILERCUT_CLIP=l14 --account=<slurm account> hpc/backbone_trial.sbatch  # only that var, like NONE
squeue -u $USER; tail -f logs/tc-trial_*.out
cat data/features/backbone_trial_h14.json
```

Worth a full rerun if F1 clearly beats 0.893 (say above 0.92) or the text card / night overlap drops.
Result (2026-11-11): H/14 scored 0.672, B/32 on the cluster 0.891, so B/32 stays (docs/results.md).

## More experiments (branch `hpc-experiments`)

All three reuse the backbone trial's upload (the labeled films' keyframes and trailer frames), except
Whisper, which needs the trailer audio.

**DINOv2 for alignment.** A self-supervised image model trained to tell images apart, which is the
part CLIP gets wrong (look-alike shots). Alignment only: it has no text side, so no concept scores.

```bash
sbatch --export=TRAILERCUT_CLIP=dinov2 --account=<account> hpc/backbone_trial.sbatch
```

**Whisper large-v3 transcripts.** The laptop used Whisper small. Output goes to
`data/cache/trailer_asr_large-v3/`, next to the small ones. Transcripts are copyrighted text: they
stay on scratch and on your laptop (or a private Drive), never in git.

```bash
# laptop: trailer audio up (~5.9 GB, one rsync)
rsync -av --progress data/cache/trailer_audio/ <netid>@dtn012.hpc.nyu.edu:<checkout>/data/cache/trailer_audio/
# cluster
sbatch --export=NONE --account=<account> hpc/asr.sbatch
# laptop: transcripts down
rsync -av <netid>@dtn012.hpc.nyu.edu:<checkout>/data/cache/trailer_asr_large-v3/ data/cache/trailer_asr_large-v3/
```

**A vision-language model as a same-shot verifier** (`vlm_rerank.py`, `hpc/vlm_pair.sbatch`). For each
labeled trailer shot and each of CLIP's 5 candidates, Qwen2.5-VL answers "same shot?" (P(yes) from the
logits), with 1 or 3 frames per shot and high-res trailer frames. Graded on what CLIP gets wrong (rescues,
rejects of footage not in the film) and blended with CLIP's score, settings picked leave-one-film-out. A
first version (all 6 frames in one prompt, pick a number, 7B) was right 54% of the time vs 94% for CLIP;
this one about 70% (docs/results.md), so CLIP stays.

```bash
# laptop: high-res trailer frames of the labeled films (~65 MB), then up
python -m trailercut.vlm_rerank frames
rsync -av data/cache/trailer_frames_hr/ <netid>@dtn012.hpc.nyu.edu:<checkout>/data/cache/trailer_frames_hr/
# cluster: one job per setting (pilot of 60 shots, then all 630)
sbatch --export=VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct,VLM_FRAMES=1 --account=<account> hpc/vlm_pair.sbatch
sbatch --export=VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct,VLM_FRAMES=3 --account=<account> hpc/vlm_pair.sbatch
sbatch --export=VLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct,VLM_FRAMES=3 --constraint=h200 --account=<account> hpc/vlm_pair.sbatch
```


## Rulers that don't come from B/32 (branch `blind-labels`)

The hand labels were picked among B/32's candidates with its order and scores shown, so they may lean
toward B/32. `trailercut.ruler` grades each backbone three other ways too (docs/design.md): the old
labels with near misses counted, the trailer's dialogue found in the film's subtitles, and a synthetic
test where film keyframes made to look like trailer frames have to find their own shot. Reuses the
backbone trial's embeddings and keyframes; one job per backbone, each grades everything there so far.

```bash
git pull && git checkout blind-labels
sbatch --export=TRAILERCUT_CLIP=b32 --account=<account> hpc/ruler.sbatch      # control: same numbers as the laptop?
sbatch --export=TRAILERCUT_CLIP=l14 --account=<account> hpc/ruler.sbatch
sbatch --export=TRAILERCUT_CLIP=h14 --account=<account> hpc/ruler.sbatch
sbatch --export=TRAILERCUT_CLIP=dinov2 --account=<account> hpc/ruler.sbatch
# after all four, grade them together (a short job, not on the login node)
sbatch --export=NONE --account=<account> --wrap "source hpc/env.sh && \$PY -m trailercut.ruler grade" \
  --job-name=tc-grade --time=00:10:00 --mem=8GB --output=logs/%x_%j.out
# laptop: results down
rsync -av <netid>@dtn012.hpc.nyu.edu:<checkout>/data/features/ruler/ data/features/ruler/
```
