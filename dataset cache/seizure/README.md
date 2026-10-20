# Seizure dataset cache

`export.py` exports NeuralBench's `seizure` task from the BIDS CHB-MIT dataset.
It uses the native subject split, five-second segments, and target extractor.
All 23 subjects remain separate across train/validation/test; original case 21
is already a second session of subject 01 in the BIDS source.

The format matches the depression cache: `data/cache.json` plus one `.pt` file
per recording, containing float64 EEG in volts, window views sharing its storage,
and example IDs. Seizure labels belong to individual windows: seizure is 0 and
background is 1, matching the native one-hot target order. Recording labels are
not used. The native class-balanced training sampler is saved in `data_config`;
validation and test retain every window at their natural prevalence.

All models receive the same 16 longitudinal bipolar derivations.
LaBraM's utility vocabulary lists all 16, but its checkpoint has embeddings for
only the first eight. For the other eight, the adapter follows NeuralBench's
first-electrode position fallback (for example, Fp1-F3 uses Fp1's position ID).
The waveform remains bipolar, and the exact IDs are recorded in run metadata.
Fz-Cz and Cz-Pz are omitted. Legacy temporal electrode names
are renamed to their equivalent modern names without changing signal values.
Coordinates use the first electrode of each pair in the standard_1005 montage,
in head coordinates and meters. Signals are not filtered or normalized during
export; each tokenizer applies its own preprocessing later.

The exporter verifies every serialized signal/window against the native decoded
array, saves raw and exported file hashes, and publishes the cache only after
all recordings finish. Four workers export independent recordings concurrently.
JSON metadata is checked before the bulk export and after each recording.
An interrupted export retains `.seizure-staging/` so a retry can validate and reuse
finished tensors. The launch script and log stay beside the exporter; archive an
existing log before retrying because the launch script refuses to overwrite it.

```sh
bash 'dataset cache/seizure/export.sh'
python 'dataset cache/seizure/verify.py'
python -m tests.test_seizure
```

The export requires the NeuroAI environment. Training uses the existing EEG
environment and can tokenize this cache in memory without writing token files.
