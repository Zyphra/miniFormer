# miniFormer

Compare DAC, TFM, and LaBraM EEG tokens using shared classifiers: cached EEG + fixed splits → tokenization → classifier training → results.

| Directory | Contents |
| --- | --- |
| [dataset cache/](<dataset cache/depression/README.md>) | EEG tensors, NeuroAI splits, metadata, and export checks. |
| [tokenization/](tokenization/README.md) | Pretrained adapters, in-memory inference, and optional token export. |
| [token_cache/](token_cache/) | Generated tokens, codebooks, and provenance. |
| [models/](models/) | Transformer and bag-of-tokens classifiers. |
| [training/](training/README.md) | Shared classifier runner for cached or live tokens, fixed splits, and comparison results. |
| [synthetic_data/](synthetic_data/sinusoid_amplitude/README.md) | Controlled EEG datasets and classifier experiments. |
| [results/](results/synthetic_runs/sinusoid_amplitude/arch_ablations/README.md) | Experiment runners, metrics, plots, and saved configurations. |
| [tests/](tests/) | Dataset, tokenization, classifier, and experiment checks. |
