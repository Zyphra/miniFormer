"""Export NeuroAI's depression EEG tensors, examples, and metadata once."""

import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_cache(neuroai, data_root, output):
    # Only exporting requires NeuroAI. Consumers need just torch and json.
    import mne
    import numpy as np
    import neuralset as ns
    import torch
    from exca import ConfDict
    from mne.io.constants import FIFF
    from neuralbench.data import Data
    from neuralbench.experiment_config import prepare_task_configs
    from neuralbench.registry import load_yaml_config
    from neuralfetch.studies.mumtaz2018machine import Mumtaz2018Machine
    from neuralset.base import Frequency
    from neuralset.events.etypes import Eeg
    from neuralset.events.utils import SpecialLoader

    neuroai, data_root = Path(neuroai).resolve(), Path(data_root).resolve()
    assert (data_root / "download").is_dir(), data_root
    sources = {}
    for obj in (Data, prepare_task_configs, load_yaml_config, Mumtaz2018Machine,
                ns.Chain, ns.Segmenter, ns.Segment, ns.events.transforms.SklearnSplit,
                ns.extractors.LabelEncoder, Eeg, SpecialLoader):
        package = obj.__module__.split(".")[0]
        relative = Path(package + "-repo", *obj.__module__.split(".")).with_suffix(".py")
        expected = sha256(neuroai / relative)
        assert sha256(inspect.getfile(obj)) == expected, f"Installed source differs: {relative}"
        sources[str(relative)] = expected

    base = neuroai / "neuralbench-repo/neuralbench"
    for relative in ("defaults/config.yaml", "defaults/grid.yaml",
                     "tasks/eeg/depression_diagnosis/config.yaml",
                     "tasks/eeg/depression_diagnosis/grid.yaml"):
        path = base / relative
        sources[str(path.relative_to(neuroai))] = sha256(path)
    configs = prepare_task_configs(
        config=ConfDict(load_yaml_config(base / "defaults/config.yaml")),
        grid=ConfDict(load_yaml_config(base / "defaults/grid.yaml")),
        device="eeg", task_name="depression_diagnosis", models=[None],
        use_task_grid=True, debug=False, force=False, prepare=False,
        download=False, quiet=True,
    )
    config = configs[0]
    assert all(c["data"] == config["data"] for c in configs)
    config["data.study.source.path"] = str(data_root)
    data_config = Data(**config["data"]).model_dump(mode="json")

    # Fresh metadata cache; model-specific EEG preprocessing happens downstream.
    with TemporaryDirectory(prefix="neuroai-depression-") as scratch:
        config["data.study.source.infra.folder"] = scratch
        config["data.study.source.infra_timelines"] = {"cluster": None, "folder": scratch}
        data = Data(**config["data"])
        events = data.study.run()
        assert len(events) == events.timeline.nunique() == 180
        assert events.subject.nunique() == 64
        assert events.groupby("subject").split.nunique().eq(1).all()
        assert events.groupby("subject").diagnosis.nunique().eq(1).all()
        assert events.task_code.value_counts().to_dict() == {"EC": 58, "EO": 61, "TASK": 61}
        assert data.trigger_event_type == "Eeg"
        assert (data.start, data.duration, data.stride, data.stride_drop_incomplete) == (0, 5, 5, True)
        dataset = ns.Segmenter(
            start=data.start, duration=data.duration, stride=data.stride,
            stride_drop_incomplete=data.stride_drop_incomplete,
            trigger_query="type == 'Eeg'", extractors={"target": data.target},
        ).apply(events)
        dataset.prepare()

        recordings = events.to_dict(orient="records")
        recording_ids = {r["timeline"]: i for i, r in enumerate(recordings)}
        windows = []
        frequency = Frequency(256)
        for example_id, segment in enumerate(dataset.segments):
            recording_id = recording_ids[segment.timeline]
            start = frequency.to_ind(segment.start)
            stop = start + frequency.to_ind(segment.duration)
            assert start == segment.start * 256 and stop - start == 1280
            assert 0 <= start < stop <= recordings[recording_id]["duration"] * 256
            windows.append(dict(example_id=example_id, recording_id=recording_id,
                                start=float(segment.start), duration=float(segment.duration),
                                start_sample=start, stop_sample=stop))
        assert len(windows) == len({(w["recording_id"], w["start_sample"]) for w in windows}) == 14683

        signal_metadata = None
        for recording_id, row in enumerate(recordings):
            event = Eeg.from_dict(row)
            raw = event.read()
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            assert len(raw.filenames) == 1 and len(picks) == 19
            assert raw.info["sfreq"] == row["frequency"] == 256 and raw.first_samp == 0
            assert raw.n_times == row["duration"] * 256 and row["start"] == row["offset"] == 0
            channels = [raw.info["chs"][i] for i in picks]
            assert all(c["unit"] == FIFF.FIFF_UNIT_V for c in channels)
            assert all(c["coord_frame"] == FIFF.FIFFV_COORD_HEAD for c in channels)
            positions = np.array([c["loc"][:3] for c in channels])
            assert np.isfinite(positions).all()
            header = {
                "frequency_hz": 256, "signal_unit": "V",
                "dtype": "float64", "processing_stage": "before_model_preprocessing",
                "channel_names": [raw.ch_names[i] for i in picks],
                "channel_positions_m": positions.tolist(), "coordinate_frame": "head",
            }
            if recording_id == 0:
                signal_metadata = header
            assert header == signal_metadata, f"Recording metadata differs: {row['timeline']}"
            target = data.target.get_static(event).tolist()
            assert target in ([1, 0], [0, 1]), target
            raw_path = Path(raw.filenames[0]).resolve()
            assert raw_path.is_relative_to(data_root)
            row.update(recording_id=recording_id, label=int(np.argmax(target)),
                       raw_path=str(raw_path), raw_sha256=sha256(raw_path),
                       raw_size_bytes=raw_path.stat().st_size, n_samples=int(raw.n_times),
                       source_channel_indices=picks.tolist(),
                       source_channel_names_after_neuroai_loading=raw.ch_names,
                       source_channel_types=raw.get_channel_types(),
                       recorded_highpass_hz=float(raw.info["highpass"]),
                       recorded_lowpass_hz=float(raw.info["lowpass"]), bad_channels=raw.info["bads"])

            signal = raw.get_data(picks=picks)
            assert signal.dtype == np.float64 and signal.shape == (19, row["n_samples"])
            assert np.isfinite(signal).all(), row["timeline"]
            eeg = torch.from_numpy(signal)
            examples = [w for w in windows if w["recording_id"] == recording_id]
            # Native segmentation must match these regular tensor views exactly.
            assert [w["start_sample"] for w in examples] == list(range(0, raw.n_times - 1279, 1280))
            signal_windows = eeg.unfold(1, 1280, 1280).permute(1, 0, 2)
            assert signal_windows.shape == (len(examples), 19, 1280)
            signal_file = f"{recording_id:03d}.pt"
            torch.save({"eeg": eeg, "windows": signal_windows,
                        "example_ids": torch.tensor([w["example_id"] for w in examples], dtype=torch.int64)},
                       output / signal_file)
            row.update(signal_file=signal_file, signal_sha256=sha256(output / signal_file))
            raw.close()

        labels = {r["diagnosis"]: r["label"] for r in recordings}
        assert labels == {"healthy": 0, "major_depressive_disorder": 1}
        assert all(r["label"] == labels[r["diagnosis"]] for r in recordings)
        assert len({r["raw_path"] for r in recordings}) == 180
        counts = {split: sum(recordings[w["recording_id"]]["split"] == split for w in windows)
                  for split in ("train", "val", "test")}
        assert counts == {"train": 8756, "val": 3094, "test": 2833}, counts

    return {
        "schema_version": 2, "task": "depression_diagnosis", "data_root": str(data_root),
        "label_mapping": labels, "signal_metadata": signal_metadata,
        "recordings": recordings, "windows": windows, "split_window_counts": counts,
        "excluded_recordings": list(Mumtaz2018Machine._MISSING),
        "data_config": data_config,
        "provenance": {
            "neuroai_revision": subprocess.check_output(
                ["git", "-C", str(neuroai), "rev-parse", "HEAD"], text=True).strip(),
            "neuroai_modified_files": subprocess.check_output(
                ["git", "-C", str(neuroai), "diff", "HEAD", "--name-only"], text=True).splitlines(),
            "source_sha256": sources, "exporter_sha256": sha256(__file__),
            "versions": {name: version(name) for name in (
                "neuralbench", "neuralset", "neuralfetch", "exca", "mne",
                "numpy", "pandas", "torch", "scikit-learn", "pydantic")},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neuroai", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "data")
    args = parser.parse_args()
    assert not args.output.exists(), f"Cache already exists: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after every tensor and the metadata have been written.
    with TemporaryDirectory(prefix=".depression-", dir=args.output.parent) as scratch:
        staging = Path(scratch) / "depression"
        staging.mkdir()
        cache = build_cache(args.neuroai, args.data_root, staging)
        encoded = json.dumps(cache, indent=2, allow_nan=False) + "\n"
        (staging / "cache.json").write_text(encoded)
        assert not args.output.exists(), f"Cache already exists: {args.output}"
        staging.rename(args.output)
    size = sum(p.stat().st_size for p in args.output.iterdir())
    print(f"Saved {len(cache['recordings'])} recordings and {len(cache['windows'])} windows "
          f"to {args.output} ({size:,} bytes including EEG)")
    print(cache["split_window_counts"])


if __name__ == "__main__":
    main()
