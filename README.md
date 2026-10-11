# TRCS evaluation suite

Code, analysis plan and per-clip results for a small, self-published evaluation of six open speech-recognition models on the synthetic TRCS benchmark. It asks two things: what do the models write when there is no speech, and how fast does error grow when speech passes through telephone codecs and noise.

**Status: the full run is in progress.** The analysis plan ([PLAN.md](PLAN.md)) was pushed before the run started. Results, when they exist, are in `results/` with SHA-256 digests. Nothing here is peer reviewed, and no one outside the project has reproduced anything yet ([REPRODUCTIONS.md](REPRODUCTIONS.md)).

- **Data:** [RoamingPigs/trcs-tactical-audio](https://huggingface.co/datasets/RoamingPigs/trcs-tactical-audio). The speech is synthetic text to speech and the channel is simulated.
- **Write-up and corrections:** [roamingpigs.org](https://roamingpigs.org/). Interactive explorers: [roamingpigs.ai](https://roamingpigs.ai/).
- **Run it yourself:** [REPRODUCING.md](REPRODUCING.md). **Report what you got:** open a "Reproduction report" issue. Rules in [REPRODUCTION-RULES.md](REPRODUCTION-RULES.md).

## Layout

| Path | What |
|---|---|
| `PLAN.md` | What was decided before the run, and what the run cannot show |
| `trcs_eval/` | Normaliser and word error counts, bootstrap statistics, gates, model runner |
| `run_e2.py`, `run_e1.py`, `run_speech_cost.py` | The three runs |
| `analyze.py` | Every pre-specified analysis, nothing else |
| `verify_results.py` | Re-hash and re-derive, or compare your run with mine |
| `MODELS.md` | The six checkpoints and their revisions |
| `expected/tolerances.json` | What counts as agreement, set before any outside run |
| `results/` | Per-clip results, analysis, report, digests |

## Licences

Code: Apache-2.0 (`LICENSE`). Results files and analysis: CC BY 4.0. Model weights keep their own licences.

## Cite

See `CITATION.cff`. There is no DOI yet.

## Earlier pilot

An earlier pilot used 5 utterances and different audio, and had decoding problems described in `PLAN.md`. It is superseded by this run.
