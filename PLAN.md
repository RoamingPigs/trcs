# Analysis plan: TRCS full-grid run (E1 v2, E2 v2)

This plan is pushed to this repository before the full run starts. The push time is recorded by GitHub, and the commit that holds this file is named in every report that uses the results. It is a dated plan, not a registered one: a registry such as OSF would add a third party's timestamp, and this repository does not.

Author: Cisco Caceres. Self-published, not peer reviewed.

## Why a second run

The pilot (October 2026) had three flaws that this run removes.

1. It scored 5 utterances, not the 40 in the released dataset, and its audio was made with a different speaking rate (160 words per minute against 155). It did not score the released files.
2. It capped output at 48 new tokens and reported no confidence intervals.
3. It padded audio features to the longest clip in each batch. Before this plan was written I found that this makes the text depend on batch composition: with that padding, 12 to 13 of 24 test clips changed when the batch size changed, for each of three models. With the standard feature extraction (every clip zero-padded to 30 s) the same clips gave identical text across batch sizes and orders (24 of 24, three models). This run uses the standard extraction and checks invariance again, for every model.

## What was done before this plan

- A smoke test of the pipeline on 12 clips with the 39M and 809M models, for timing and crashes only.
- The batch-invariance test above, on 24 clips and three models.
- Per-clip timing of the two largest models at 24, 48 and 96 threads (48 is used).
- Recomputing one clip per codec from the generator, to confirm that the released audio is real codec output.

No error rates, hallucination rates or contrasts from the full data were looked at. The smoke-test transcripts were read to confirm that the pipeline works, and none of them informed the choices below.

## Fixed inputs

| Item | Value |
|---|---|
| Dataset | `RoamingPigs/trcs-tactical-audio`, revision `8814846748e31203aa48abaf44504d5a10531dc6`. Every file is checked against `provenance_manifest.json` before any run. |
| Models | Six, listed with exact revisions in `MODELS.md` and `trcs_eval/models.py`. |
| Library | Hugging Face `transformers` 5.16.1, `torch` 2.14.0 (CPU), float32. |
| Decoding | Greedy (`num_beams=1`, `do_sample=False`), `max_new_tokens=128`, no timestamps. `language="en"` and `task="transcribe"` for the two multilingual-config models; nothing set for the English-only models. Batch size 16. |
| Library defaults | In `transformers` the fallback and no-speech thresholds apply only to audio longer than 30 s, so they do not run on these clips. Results are not comparable with `openai-whisper` or `faster-whisper` defaults, which apply them. |
| Text normaliser | The English normaliser of the Whisper tokenizer at `openai/whisper-base.en` revision `911407f4214e0e1d82085af863093ec0b66f9cd6`, applied identically to the reference and to every output. It was developed alongside Whisper and may favour it. Raw-token scoring (lower-case words, as in the pilot) is reported next to it. |
| Hardware | Intel Xeon Platinum 8160, 48 threads, CPU only. |
| Seeds | Decoding is deterministic. The bootstrap seed is `20261011`. |

## E2: error rate on speech (960 clips, 6 models)

Unit of analysis: the utterance. The 40 scripted calls recur in every audio path and noise level, so I resample utterances with replacement and carry all 24 cells of each script together. 10,000 resamples, percentile 95% intervals. Error rate is corpus-level: total edits over total reference words, never a mean of per-clip rates. Differences are computed on the same resamples (paired).

### Pre-specified analyses

- **A1.** Error rate at each noise level (clean, 20, 10, 0 dB), pooled over the six models and six audio paths.
- **A2.** Error rate for each model on clean, uncompressed audio.
- **A3.** Paired difference between each of the five narrowband paths and the uncompressed path, on clean audio, pooled over models.
- **A4.** Paired difference between each noise level and clean, pooled over models and paths.
- **A5.** The full grid (model x path x noise) with intervals. Descriptive only: no claim is made from a single cell.
- Secondary, descriptive: the same A1 to A4 with raw-token scoring; the share of edits that are insertions; the rate at which expected call signs and channel numbers are missing; how many outputs hit the token cap; how many clips have more edits than reference words.

### Expectations stated in advance

- The error rate rises with noise, so every A4 difference is positive.
- Every narrowband path is worse than uncompressed on clean audio, but the five narrowband paths cannot be ranked against each other. I make no ranking claim.
- Larger models have lower clean error rates. With six models this is descriptive only.

A contrast is reported as an effect only when its interval excludes zero. Everything else is reported as undetermined at this sample size.

## E1: words written for non-speech clips (90 clips, 6 models, 3 arms)

A **hallucination** is a non-empty normalised transcript of a non-speech clip. No error rate is computed on these clips.

Clips with identical audio (identical SHA-256) are counted once, because greedy decoding gives identical output. That leaves 71 distinct non-speech clips per model: one digital-silence clip and ten seeds in each of seven other conditions.

Arms:

- **none:** the raw clip.
- **energy:** the pilot's energy gate (20 ms frames, 10 ms hop, -42 dBFS, 3-frame hangover, clip skipped if under 15% of frames are active, inactive frames zeroed).
- **neural:** Silero VAD (version recorded in `run-meta.json`, ONNX), used the same way: skip the clip if no speech is found, otherwise zero samples outside detected speech.

### Pre-specified analyses

- **E1-1.** Per arm and model: the share of distinct non-speech clips with any word, with a 95% Wilson interval.
- **E1-2.** Per arm and condition, pooled over models, with a bootstrap interval over clips.
- **E1-3.** The ten most frequent ungated outputs.
- **E1-4.** Among ungated hallucinations: how many hit the token cap (a proxy for looping), and the mean words per output.
- **Speech cost.** Each gate is also applied to 80 speech clips (uncompressed, clean and 0 dB). I report the error rate with each gate, the number of speech clips a gate skips, and the paired difference from no gate. A gate that removes hallucination but also removes speech is not a fix.

### Expectations stated in advance

- The energy gate removes words on digital silence and quiet-room clips and not on noise or squelch, because it measures level only.
- The neural gate removes more than the energy gate, but not all: an earlier study measured Silero VAD at about 61% accuracy on environmental noise.
- Either gate costs something on speech at 0 dB.

## Checks

- Every dataset file matches the manifest before a run starts.
- Batch invariance: for each model, 24 seeded clips decoded one at a time must give the same text as the batched run. The count is written to `results/e2/batch-invariance.json`. A mismatch is reported, not hidden.
- `verify_results.py` re-hashes every result file and re-derives `analysis.json` from the raw per-clip results.

## What this run cannot show

- Anything about real radio or real telephone speech. The speech is synthetic text to speech and the channel is simulated. Noise is added after the codec, and the SNR is measured over the whole clip including pauses.
- Anything about models outside this family, or about other decoding libraries and settings.
- Anything about other hardware. CPU and GPU results can differ.
- A benefit of any gate beyond these clips.

## Claims I will not make

"Preregistered", "validated", "verified", "reproduced" (until someone outside the project has done it), any ranking of codecs without a paired interval, and any statement about a product.

## Reproduction criteria

Set before any outside run, in `expected/tolerances.json`. Summary: on the same revisions and software, at least 95% of per-clip transcripts should be identical to those published, and every published headline rate should lie inside its published interval. A different CPU or library version may flip rare tokens, so exact identity is not required.
