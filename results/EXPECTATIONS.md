# Expectations written in advance, and what happened

From the "Expectations stated in advance" sections of PLAN.md. Numbers are from `analysis.json`.

| Expectation | Outcome |
|---|---|
| Error rises with noise: every noise-minus-clean difference is positive | Held. All three differences are positive with intervals that exclude zero. |
| Every narrowband path is worse than uncompressed on clean audio | Held. All five differences are positive with intervals that exclude zero. |
| The five narrowband paths cannot be ranked against each other | Not tested. The plan did not specify codec-against-codec contrasts, so no ranking is claimed. |
| Larger models have lower clean error | Partly. The smallest model is worst and the largest best, but the 166M distilled model is worse than the 74M model. Six models, descriptive only. |
| The energy gate clears digital silence and the quiet room and not noise or squelch | Partly wrong. It cleared both silences and also radio squelch bursts. It left all five noise conditions at 100%. |
| The neural gate clears more than the energy gate, but not everything | Wrong. It skipped every non-speech clip (0 of 71 distinct clips produced text, for every model) and skipped none of the 80 speech clips. |
| Either gate costs something on speech at 0 dB | Half wrong. The energy gate changed word error by 0.0 points (95% interval -0.04 to +0.04), and 471 of 480 transcripts were identical to ungated. The synthetic pauses are already digital silence and noisy clips never fall below its threshold. The neural gate raised word error by +8.6 points (95% interval +1.6 to +16.0). |

The energy-gate speech result is zero by construction on this audio and says nothing about real recordings. The neural-gate result is on generated noise (white, pink, brown, engine, babble, squelch), which may be easier to separate from speech than real noise.
