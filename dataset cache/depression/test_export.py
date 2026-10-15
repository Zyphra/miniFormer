"""Integration check against NeuroAI; run after exporting in its environment."""

import json
from pathlib import Path


def test_cache_matches_native_pipeline(tmp_path):
    import mne
    import numpy as np
    import neuralset as ns
    import torch
    from mne.io.constants import FIFF
    from neuralbench.data import Data
    from neuralset.events.etypes import Eeg
    from export import sha256

    path = Path(__file__).parent / "data/cache.json"
    cache = json.loads(path.read_text())
    assert cache["schema_version"] == 2
    assert cache["provenance"]["exporter_sha256"] == sha256(Path(__file__).with_name("export.py"))
    data = Data(**cache["data_config"])
    source = data.study.steps["source"]
    source.infra.folder = tmp_path
    source.infra_timelines.folder = tmp_path
    source.infra_timelines.cluster = None
    data.num_workers = 0

    # Exercise native Data.prepare() without filtering/caching EEG. This static
    # extractor changes only the unused brain input, preserving events/targets.
    data.neuro = ns.extractors.EventField(
        event_types="Eeg", event_field="frequency", aggregation="trigger")
    events = data.study.run()
    recordings = cache["recordings"]
    assert [{key: row[key] for key in events.columns} for row in recordings] == events.to_dict("records")
    metadata = cache["signal_metadata"]
    for i, row in enumerate(recordings):
        assert row["recording_id"] == i
        assert row["raw_sha256"] == sha256(row["raw_path"])
        assert row["label"] == cache["label_mapping"][row["diagnosis"]]
        signal_path = path.parent / row["signal_file"]
        assert row["signal_sha256"] == sha256(signal_path)
        payload = torch.load(signal_path, map_location="cpu", weights_only=True)
        assert set(payload) == {"eeg", "windows", "example_ids"}
        eeg, windows, ids = payload["eeg"], payload["windows"], payload["example_ids"]
        assert eeg.dtype == windows.dtype == torch.float64 and metadata["dtype"] == "float64"
        assert eeg.shape == (19, row["n_samples"])
        assert ids.dtype == torch.int64
        assert windows.untyped_storage().data_ptr() == eeg.untyped_storage().data_ptr()

        # Compare every decoded sample at the same pre-filtering stage, including
        # the recording tail that is not part of any benchmark window.
        raw = Eeg.from_dict(events.iloc[i]).read()
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        assert picks.tolist() == row["source_channel_indices"]
        assert [raw.ch_names[p] for p in picks] == metadata["channel_names"]
        assert raw.info["sfreq"] == metadata["frequency_hz"] == 256
        assert metadata["signal_unit"] == "V"
        assert all(raw.info["chs"][p]["unit"] == FIFF.FIFF_UNIT_V for p in picks)
        assert metadata["coordinate_frame"] == "head"
        assert all(raw.info["chs"][p]["coord_frame"] == FIFF.FIFFV_COORD_HEAD for p in picks)
        np.testing.assert_array_equal(
            [raw.info["chs"][p]["loc"][:3] for p in picks], metadata["channel_positions_m"])
        native = raw.get_data(picks=picks)
        np.testing.assert_array_equal(eeg.numpy(), native)
        examples = [w for w in cache["windows"] if w["recording_id"] == i]
        assert ids.tolist() == [w["example_id"] for w in examples]
        assert windows.shape == (len(examples), 19, 1280)
        for signal, example in zip(windows, examples, strict=True):
            np.testing.assert_array_equal(
                signal.numpy(), native[:, example["start_sample"]:example["stop_sample"]])
        raw.close()

    expected = {}
    for split, loader in data.prepare().items():
        batch = loader.dataset.load_all()
        labels = batch.data["target"].argmax(dim=-1).tolist()
        for segment, label in zip(batch.segments, labels, strict=True):
            key = (segment.timeline, segment.start, segment.duration)
            assert key not in expected
            expected[key] = (split, label)

    actual = {}
    for i, window in enumerate(cache["windows"]):
        assert window["example_id"] == i
        recording = recordings[window["recording_id"]]
        key = (recording["timeline"], window["start"], window["duration"])
        assert key not in actual
        actual[key] = (recording["split"], recording["label"])
        assert window["start_sample"] == window["start"] * recording["frequency"]
        assert window["stop_sample"] == (window["start"] + window["duration"]) * recording["frequency"]
        assert 0 <= window["start_sample"] < window["stop_sample"] <= recording["n_samples"]
    assert actual == expected
