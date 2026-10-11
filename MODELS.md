# Models

Six open speech models, each pinned to an exact Hugging Face revision. The websites describe them by size and kind. The repository names them so the runs can be repeated.

| Label used on the websites | Checkpoint | Revision |
|---|---|---|
| 39M English-only | `openai/whisper-tiny.en` | `87c7102498dcde7456f24cfd30239ca606ed9063` |
| 74M English-only | `openai/whisper-base.en` | `911407f4214e0e1d82085af863093ec0b66f9cd6` |
| 166M English-only (distilled) | `distil-whisper/distil-small.en` | `9e4a67ca4569c30be43a3fe7fba1621e504f0093` |
| 244M English-only | `openai/whisper-small.en` | `e8727524f962ee844a7319d92be39ac1bd25655a` |
| 756M multilingual (distilled) | `distil-whisper/distil-large-v3` | `ff05a2254c6f9542535fd778dae85b39a08cae31` |
| 809M multilingual | `openai/whisper-large-v3-turbo` | `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` |

Text normaliser: the English normaliser of the tokenizer at `openai/whisper-base.en` revision `911407f4214e0e1d82085af863093ec0b66f9cd6`.

Multilingual-config checkpoints are run with `language="en"` and `task="transcribe"`. English-only checkpoints get neither.
