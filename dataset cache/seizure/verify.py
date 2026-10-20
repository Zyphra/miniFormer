"""Check every cached window label against the original BIDS seizure annotations."""

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path


def verify(folder):
    data = json.loads((folder / "cache.json").read_text())
    recordings, windows = data["recordings"], data["windows"]
    assert len(recordings) == 686 and len(windows) == 707683
    assert data["label_mapping"] == {"seizure": 0, "background": 1}
    assert data["data_config"]["sampler"] == {"name": "ClassificationSampler"}
    assert data["provenance"]["exporter_sha256"] == hashlib.sha256(
        Path(__file__).with_name("export.py").read_bytes()).hexdigest()
    subjects = {split: {r["subject"] for r in recordings if r["split"] == split}
                for split in ("train", "val", "test")}
    assert sum(map(len, subjects.values())) == len(set.union(*subjects.values())) == 23
    counts = {split: Counter() for split in subjects}
    by_recording = [[] for _ in recordings]
    for example_id, window in enumerate(windows):
        assert window["example_id"] == example_id
        by_recording[window["recording_id"]].append(window)
    n_seizures = 0
    for recording_id, (row, selected) in enumerate(zip(recordings, by_recording)):
        assert row["recording_id"] == recording_id
        raw_path = Path(row["raw_path"])
        assert raw_path.stat().st_size == row["raw_size_bytes"]
        assert (folder / row["signal_file"]).stat().st_size >= row["n_samples"] * 16 * 8
        events = raw_path.with_name(raw_path.name.replace("_eeg.edf", "_events.tsv"))
        with events.open() as stream:
            seizures = [(float(r["onset"]), float(r["duration"]))
                        for r in csv.DictReader(stream, delimiter="\t") if r["eventType"] == "sz"]
        assert sorted(seizures) == sorted((s["start"], s["duration"]) for s in row["seizures"])
        n_seizures += len(seizures)
        assert len(selected) == row["n_samples"] // 1280
        for index, window in enumerate(selected):
            assert window["start"] == index * 5 and window["duration"] == 5
            assert window["start_sample"] == index * 1280
            assert window["stop_sample"] == (index + 1) * 1280
            overlap = any(start < window["start"] + 5 and start + duration > window["start"]
                          for start, duration in seizures)
            assert window["label"] == (0 if overlap else 1), (raw_path, window)
            counts[row["split"]][window["label"]] += 1
    assert n_seizures == 198
    assert sum(c[0] for c in counts.values()) == 2480
    for split, count in counts.items():
        assert all(count[label] > 0 for label in (0, 1))
        assert {str(k): v for k, v in count.items()} == data["split_label_counts"][split]
        assert sum(count.values()) == data["split_window_counts"][split]
        print(f"{split}: {len(subjects[split])} subjects, {sum(count.values()):,} windows, "
              f"{count[0]:,} seizure, {count[1]:,} background", flush=True)
    print("PASS: all 707,683 labels match BIDS annotations; subjects do not cross splits", flush=True)


if __name__ == "__main__":
    verify(Path(__file__).parent / "data")
