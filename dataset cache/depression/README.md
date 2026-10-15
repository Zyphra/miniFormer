# Depression dataset cache

NeuroAI's depression benchmark exported for use without NeuroAI or EDF loading.
Preserves eyes-open, eyes-closed, and P300 recordings, subject splits, and labels (healthy=0, MDD=1).

| File | Purpose |
| --- | --- |
| [export.py](export.py) | Read EDFs through NeuroAI and save the dataset once. |
| [test_export.py](test_export.py) | Compare every recording, window, split, and label against NeuroAI. |
| `data/cache.json` | Subjects, conditions, labels, splits, window boundaries, and provenance. |
| `data/*.pt` | Complete EEG, five-second window views, and example IDs; readable with PyTorch. |

EEG is 19 channels at 256 Hz, float64 volts, before model preprocessing. Window views share the recording's storage.
Export and verification use the NeuroAI environment; [tokenization](../../tokenization/README.md) consumes this cache.
