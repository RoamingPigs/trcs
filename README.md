# TRCS evaluation suite

Code, analysis plan and per-clip results for a small, self-published evaluation of six open speech-recognition models on the synthetic TRCS benchmark. It asks two things: what do the models write when there is no speech, and how fast does error grow when speech passes through telephone codecs and noise.

**Status: the full run is finished and published in `results/`.** The analysis plan ([PLAN.md](PLAN.md)) was pushed on 2026-10-11 at 22:51:28 UTC, before the run started, as commit `7af45c9`. Results carry SHA-256 digests; `python3 verify_results.py --published results` checks them and rebuilds the analysis from the raw per-clip files. Nothing here is peer reviewed, and no one outside the project has reproduced anything yet ([REPRODUCTIONS.md](REPRODUCTIONS.md)).

Headline numbers are in [results/REPORT.md](results/REPORT.md). Where my written expectations in the plan turned out wrong, [results/EXPECTATIONS.md](results/EXPECTATIONS.md) says so.

**Deviations from the plan:** see `results/RUN-INFO.json`. In short: E1 and the speech-cost run were started once, were far too slow because the VAD library reset torch to one thread, and were discarded and re-run from the start at 48 threads after a fix (commit `9f023c7`). E2 was not affected. No other change was made to code or analysis after the plan was pushed, apart from adding `check_matrix.py` and `make_sums.py`, which read results and change none.

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
| `check_matrix.py` | Completeness and consistency of the raw results against the dataset manifest |
| `make_sums.py` | Writes `results/SHA256SUMS` |
| `generator/` | The code that made the audio, and a check that regenerating it reproduces every released SHA-256 (added after the run) |
| `MODELS.md` | The six checkpoints and their revisions |
| `expected/tolerances.json` | What counts as agreement, set before any outside run |
| `results/` | Per-clip results, analysis, report, digests |

## Licences

Code: Apache-2.0 (`LICENSE`). Results files and analysis: CC BY 4.0. Model weights keep their own licences.

## Cite

See `CITATION.cff`. There is no DOI yet.

## Earlier pilot

An earlier pilot used 5 utterances and different audio, and had decoding problems described in `PLAN.md`. It is superseded by this run.
