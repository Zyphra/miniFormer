# Depression dataset cache

From the repo root, run `export.py` in the existing NeuroAI environment. It reuses NeuroAI's
configuration assembly, study loader, subject splitter, segmenter, and label
encoder. It reads the EDFs through NeuroAI and saves the decoded EEG in PyTorch
files. An existing output folder is an error; a new folder is published only
after the full export succeeds. Run with assertions enabled (without Python's
`-O` flag).

```sh
NEURALBENCH_CONFIG=/data/groups/bci/jonhuml/workspace/neuroai/.neuralbench_config.json \
MNE_DONTWRITE_HOME=true \
/data/groups/bci/jonhuml/workspace/neuroai/neuralbench-repo/.venv/bin/python "dataset cache/depression/export.py" \
  --neuroai /data/groups/bci/jonhuml/workspace/neuroai \
  --data-root /data/groups/bci/datasets/neuralbench/datasets/Mumtaz2018Machine
```

Generated files live in `data/` beside this exporter and test. Each `.pt` file
contains three tensors, readable with PyTorch alone:

- `eeg`: the complete recording, shaped `(19, samples)`, including its tail.
- `windows`: NeuroAI's five-second examples, shaped `(examples, 19, 1280)`.
- `example_ids`: the stable IDs of those examples, shaped `(examples,)`.

Signals retain MNE's float64 values in **volts**, at **256 Hz**, before any
model-specific filtering, referencing, resampling, scaling, or float32 cast.
`windows` is a view of `eeg`; PyTorch stores their shared signal data only once.
Both tensors share storage after loading too; do not modify one in place while
expecting the other to remain unchanged. The full cache is approximately 2.87 GB.
Neither NeuroAI nor the EDF files are needed to consume it.

For example, inference code can load a file and form batches without split or
label metadata:

```python
import torch

recording = torch.load("dataset cache/depression/data/000.pt", map_location="cpu", weights_only=True)
for eeg, example_ids in zip(recording["windows"].split(32), recording["example_ids"].split(32)):
    # Apply the tokenizer's window-level preprocessing, then tokenize eeg.
    # Save example_ids alongside the resulting tokens.
    pass
```

These are unprocessed windows, not checkpoint-ready inputs. When a tokenizer
requires filtering or resampling of the full recording, start from `eeg`, apply
that preprocessing, then select the cached time intervals at the new sample
rate before batching. The native window views are not valid after resampling.
Tokenizer-specific preprocessing and inference adapters are not part of this
exporter.

The accompanying `cache.json` is readable with Python's standard `json` module.
`recordings` preserves every column of NeuroAI's event
table, including `filepath` (its loader recipe), subject, condition, diagnosis,
and split. Each recording also stores the actual EDF path, content hash, sample
count, integer label, source channel inventory and EEG selection indices, plus
its exported `signal_file` and SHA-256 hash. EDF paths and loader recipes are
provenance only; consumers read the `.pt` files.
Shared EEG metadata includes channel order, native 256 Hz sampling, signal units
in volts, and head coordinates in meters. These are native coordinates, before
NeuroAI's coordinate normalization.

`windows` preserves NeuroAI's example order. `recording_id` indexes `recordings`;
the label and split come from that recording. Window boundaries are stored both
in seconds and as native sample indices, with an **exclusive** `stop_sample`.
All conditions are included: eyes closed, eyes open, and P300. The expected counts
are 8,756 train / 3,094 validation / 2,833 test windows; healthy is 0 and MDD is 1.

The cache also records exclusions, the resolved data configuration, code hashes,
repository revision, and dependency versions. The configuration's EEG extractor
settings are benchmark provenance; the exporter does not run that extractor.
Generate a new cache explicitly when inputs or benchmark definitions change.
Classifier training joins token `example_ids` to these windows and uses their
cached labels and split assignments. Tokenizer inference does not need them.

Run `python -m pytest "dataset cache/depression/test_export.py"` with the same
NeuroAI environment to compare every cached recording, window, split, and label against native
`Data.prepare()`. It also verifies EDF and tensor-file hashes, channel order,
units, sample rate, coordinates, and exact equality of every saved signal sample
against NeuroAI's native EDF reader. This includes every window and the unused
recording tails. The metadata check uses a static placeholder brain input to
avoid running model-specific preprocessing.
