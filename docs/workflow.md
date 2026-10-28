# Workflow

How we work in this repo.

## Setup

```bash
uv venv --python 3.12 && uv pip install -e .     # laptop
bash hpc/setup.sh                                # cluster, see hpc/README.md
```

## Repo layout

| path | what |
|---|---|
| `src/trailercut/` | one module per pipeline step, each runnable with `python -m trailercut.<step>` |
| `scripts/run_all.sh` | every step in order |
| `hpc/` | Slurm jobs for the heavy steps |
| `data/ids/` | film list, trailer availability and recovery logs |
| `data/features/` | everything the analysis and views read |
| `labels/` | labeling tool, sample, gold labels |
| `figures/` | generated figures |
| `docs/` | these docs |

## Adding a pipeline step

1. New module `src/trailercut/<step>.py` with a top docstring: what it does, outputs, CLI usage.
2. Skip work that's already done, so reruns are cheap.
3. Paths from `trailercut.paths` only; never hardcode `data/`.
4. Add it to `scripts/run_all.sh`, `docs/pipeline.md` and the README pipeline table.
5. New output files go in `docs/data_dictionary.md`, and either the committed list in the README or
   `.gitignore`.

## What gets committed

Commit: code, IDs and logs, feature tables under ~20 MB, JSON reports, figures, gold labels.
Don't commit: videos, keyframes, embeddings, audio, anything that rebuilds in a few minutes and is
over ~50 MB. Check `git status` for large files before committing (`du -h` on anything new).

## Commits

- Small commits, one idea each, present tense ("Add ...", "Stop tracking ...").
- Pipeline outputs get their own commit ("Run pipeline on ..."), separate from code changes.

## Docs

- Every code change gets a docs pass: `design.md` for new decisions and why, `pipeline.md` and
  `data_dictionary.md` for new steps or files, `results.md` for new numbers, `experiments.md` for
  things tried, `TODO.md` for status.
- Numbers in docs come from committed outputs, never from memory.
- No dashes for ranges, write "0.78 to 0.93".

## Dividing the work

- Cedric: data pipeline, RQ1 model, Trailer View.
- Estee: clustering, Overview, case studies.
- Shared: labeling, report, presentation.

The interface between the two halves is `data/features/` (see `data_dictionary.md`). Change a column
there and tell the other person.
