"""Export NeuralBench's CHB-MIT signals, window labels, and subject split once."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter


# Standard longitudinal bipolar montage, shared by all three tokenizers.
CHANNELS = """Fp1-F7 F7-T7 T7-P7 P7-O1 Fp2-F8 F8-T8 T8-P8 P8-O2
Fp1-F3 F3-C3 C3-P3 P3-O1 Fp2-F4 F4-C4 C4-P4 P4-O2""".split()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_channel(name):
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    return "-".join(aliases.get(part, part).replace("FP", "Fp").replace("Z", "z")
                    for part in name.split("-"))


def build_data(neuroai, data_root, scratch):
    from exca import ConfDict
    from neuralbench.data import Data
    from neuralbench.experiment_config import prepare_task_configs
    from neuralbench.registry import load_yaml_config

    base = Path(neuroai) / "neuralbench-repo/neuralbench"
    configs = prepare_task_configs(
        config=ConfDict(load_yaml_config(base / "defaults/config.yaml")),
        grid=ConfDict(load_yaml_config(base / "defaults/grid.yaml")),
        device="eeg", task_name="seizure", models=[None], use_task_grid=True,
        debug=False, force=False, prepare=False, download=False, quiet=True,
    )
    assert all(c["data"] == configs[0]["data"] for c in configs)
    config = configs[0]["data"]
    config["study.source.path"] = str(Path(data_root).resolve())
    config["study.source.infra.folder"] = str(scratch)
    config["study.source.infra_timelines"] = {"cluster": None, "folder": str(scratch)}
    data = Data(**config)
    assert (data.start, data.duration, data.stride, data.stride_drop_incomplete) == (0, 5, 5, True)
    assert data.sampler.name == "ClassificationSampler"
    return data


def native_windows(data, events):
    """Use native segmentation and target extraction, including seizure boundaries."""
    import neuralset as ns

    dataset = ns.Segmenter(start=data.start, duration=data.duration, stride=data.stride,
        stride_drop_incomplete=data.stride_drop_incomplete, trigger_query="type == 'Eeg'",
        extractors={"target": data.target}).apply(events)
    dataset.prepare()
    assert data.target._label_to_ind == {"seiz": 0, "__missing__": -1}
    return dataset


def save_recording(recording_id, row, examples, annotations, output, data_root):
    """Save and verify one recording; retain finished tensors for a retry."""
    import numpy as np
    import torch
    from mne.io.constants import FIFF
    from neuralset.events.etypes import Eeg

    started = perf_counter()
    raw = Eeg.from_dict(row).read()
    renames = {name: canonical_channel(name) for name in raw.ch_names}
    raw.rename_channels(renames)
    picks = [raw.ch_names.index(name) for name in CHANNELS]
    assert raw.info["sfreq"] == row["frequency"] == 256 and raw.first_samp == 0
    assert raw.n_times == row["duration"] * 256 and row["start"] == row["offset"] == 0
    assert all(raw.info["chs"][p]["unit"] == FIFF.FIFF_UNIT_V for p in picks)
    assert all(raw.get_channel_types()[p] == "eeg" for p in picks)
    raw_path = Path(raw.filenames[0]).resolve()
    assert raw_path.is_relative_to(data_root)
    signal = raw.get_data(picks=picks)
    assert signal.dtype == np.float64 and np.isfinite(signal).all()
    eeg = torch.from_numpy(signal)
    assert [w["start_sample"] for w in examples] == list(range(0, raw.n_times - 1279, 1280))
    signal_windows = eeg.unfold(1, 1280, 1280).permute(1, 0, 2)
    ids = torch.tensor([w["example_id"] for w in examples], dtype=torch.int64)
    filename = f"{recording_id:03d}.pt"
    if not (output / filename).exists():
        temporary = output / (filename + ".tmp")
        torch.save({"eeg": eeg, "windows": signal_windows, "example_ids": ids}, temporary)
        temporary.replace(output / filename)
    saved = torch.load(output / filename, weights_only=True)
    assert torch.equal(saved["eeg"], eeg) and torch.equal(saved["windows"], signal_windows)
    assert torch.equal(saved["example_ids"], ids)
    assert saved["windows"].untyped_storage().data_ptr() == saved["eeg"].untyped_storage().data_ptr()
    row.update(recording_id=recording_id, raw_path=str(raw_path), raw_sha256=sha256(raw_path),
        raw_size_bytes=raw_path.stat().st_size, n_samples=int(raw.n_times),
        signal_file=filename, signal_sha256=sha256(output / filename),
        source_channel_indices=picks, source_channel_names_after_neuroai_loading=raw.ch_names,
        channel_renames=renames, seizures=annotations)
    json.dumps(row, allow_nan=False)
    raw.close()
    del saved, eeg, signal, signal_windows
    return recording_id, len(examples), perf_counter() - started


def build_cache(neuroai, data_root, output):
    import mne
    import neuralset as ns
    import torch
    from neuralbench.data import Data, ClassificationSampler
    from neuralfetch.studies.dan2023bids import Dan2023Bids
    from neuralset.events.etypes import Eeg

    torch.set_num_threads(4)
    mne.set_log_level("ERROR")
    neuroai, data_root = Path(neuroai).resolve(), Path(data_root).resolve()
    sources = {}
    for obj in (Data, ClassificationSampler, Dan2023Bids, ns.Segmenter, ns.Segment,
                ns.events.transforms.SklearnSplit, ns.extractors.LabelEncoder, Eeg):
        module = obj.__module__
        relative = Path(module.split(".")[0] + "-repo", *module.split(".")).with_suffix(".py")
        assert sha256(inspect.getfile(obj)) == sha256(neuroai / relative), relative
        sources[str(relative)] = sha256(neuroai / relative)
    for relative in ("defaults/config.yaml", "defaults/grid.yaml",
                     "tasks/eeg/seizure/config.yaml", "tasks/eeg/seizure/grid.yaml"):
        path = neuroai / "neuralbench-repo/neuralbench" / relative
        sources[str(path.relative_to(neuroai))] = sha256(path)

    with TemporaryDirectory(prefix="miniformer-seizure-metadata-") as scratch:
        data = build_data(neuroai, data_root, scratch)
        print("Reading native CHB-MIT metadata and subject split", flush=True)
        events = data.study.run()
        assert events.subject.nunique() == 23
        assert events.groupby("subject").split.nunique().eq(1).all()
        assert events.type.value_counts().to_dict() == {"Eeg": 686, "Seizure": 198}
        recordings = json.loads(events[events.type == "Eeg"].to_json(orient="records"))
        recording_ids = {row["timeline"]: i for i, row in enumerate(recordings)}
        seizures = events[events.type == "Seizure"]
        print("Extracting native five-second windows and labels", flush=True)
        dataset = native_windows(data, events)
        windows, by_recording = [], [[] for _ in recordings]
        started = perf_counter()
        for example_id, segment in enumerate(dataset.segments):
            target = data.target(**segment._to_extractor())
            assert target.shape == (2,) and target.sum() > 0
            recording_id = recording_ids[segment.timeline]
            start = int(segment.start * 256)
            assert start == segment.start * 256
            window = dict(example_id=example_id, recording_id=recording_id,
                start=float(segment.start), duration=float(segment.duration),
                start_sample=start, stop_sample=start + 1280, label=int(target.argmax()))
            windows.append(window)
            by_recording[recording_id].append(window)
            if (example_id + 1) % 100000 == 0:
                print(f"Labeled {example_id + 1:,} windows in {perf_counter() - started:.1f}s", flush=True)
        assert len(windows) == 707683
        del dataset

        montage = mne.channels.make_standard_montage("standard_1005")
        transform = mne.channels.compute_native_head_t(montage)
        positions = montage.get_positions()["ch_pos"]
        coords = [mne.transforms.apply_trans(transform, positions[c.split("-")[0]]).tolist()
                  for c in CHANNELS]
        signal_metadata = dict(frequency_hz=256, signal_unit="V", dtype="float64",
            processing_stage="before_model_preprocessing", channel_names=CHANNELS,
            channel_positions_m=coords, coordinate_frame="head",
            position_source="standard_1005, first electrode of each bipolar derivation",
            channel_selection="16 bipolar derivations supported by all three tokenizers; omit Fz-Cz and Cz-Pz")

        counts = {split: Counter(w["label"] for w in windows
                  if recordings[w["recording_id"]]["split"] == split) for split in ("train", "val", "test")}
        cache = dict(schema_version=2, task="seizure", data_root=str(data_root),
            label_mapping={"seizure": 0, "background": 1}, signal_metadata=signal_metadata,
            recordings=recordings, windows=windows, split_window_counts={s: sum(c.values()) for s, c in counts.items()},
            split_label_counts=counts, data_config=data.model_dump(mode="json"),
            provenance=dict(neuroai_revision=subprocess.check_output(
                ["git", "-C", str(neuroai), "rev-parse", "HEAD"], text=True).strip(),
                source_sha256=sources, exporter_sha256=sha256(__file__),
                versions={name: version(name) for name in ("neuralbench", "neuralset", "neuralfetch",
                    "exca", "mne", "numpy", "pandas", "torch", "scikit-learn", "pydantic")}))

        # Validate the full JSON structure before starting the large signal export.
        json.dumps(cache, allow_nan=False)
        annotations = {timeline: json.loads(group[["start", "duration", "state"]].to_json(orient="records"))
                       for timeline, group in seizures.groupby("timeline")}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(save_recording, i, row, by_recording[i],
                       annotations.get(row["timeline"], []), output, data_root)
                       for i, row in enumerate(recordings)]
            for completed, future in enumerate(as_completed(futures), 1):
                recording_id, n_windows, elapsed = future.result()
                print(f"Saved {completed}/686: recording {recording_id}, {n_windows:,} windows in {elapsed:.1f}s", flush=True)
        return cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neuroai", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    assert not args.output.exists(), f"Cache already exists: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.parent / ".seizure-staging"
    staging.mkdir(exist_ok=True)
    cache = build_cache(args.neuroai, args.data_root, staging)
    (staging / "cache.json").write_text(json.dumps(cache, indent=2, allow_nan=False) + "\n")
    assert not args.output.exists()
    staging.rename(args.output)
    print(f"Export complete: {args.output}", flush=True)
    print(cache["split_window_counts"], cache["split_label_counts"], flush=True)
