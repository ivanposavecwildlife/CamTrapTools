import json
import os
import struct
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from camtraptools import detect, ingest, video


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def mp4(recorded: datetime | None, moov_last: bool = True, version: int = 0) -> bytes:
    seconds = int((recorded - video.QUICKTIME_EPOCH).total_seconds()) if recorded else 0
    if version == 1:
        header = bytes([1, 0, 0, 0]) + struct.pack(">QQ", seconds, seconds) + bytes(100)
    else:
        header = bytes([0, 0, 0, 0]) + struct.pack(">II", seconds, seconds) + bytes(88)
    movie = box(b"moov", box(b"mvhd", header))
    parts = [box(b"ftyp", b"isom" + bytes(4)), box(b"mdat", bytes(5000))]
    return b"".join([*parts, movie] if moov_last else [parts[0], movie, parts[1]])


@pytest.mark.parametrize(("moov_last", "version"), [(True, 0), (False, 0), (True, 1)])
def test_quicktime_time_reads_mvhd_anywhere(tmp_path, moov_last, version):
    path = tmp_path / "clip.MP4"
    path.write_bytes(mp4(datetime(2024, 3, 1, 8, 30), moov_last, version))

    assert video.quicktime_time(path) == datetime(2024, 3, 1, 8, 30)


def test_avi_time_reads_idit(tmp_path):
    path = tmp_path / "clip.AVI"
    text = b"MON MAR 03 10:22:33 2024\n\0"
    path.write_bytes(
        b"RIFF" + struct.pack("<I", 100) + b"AVI LIST" + bytes(20) + b"IDIT" + struct.pack("<I", len(text)) + text
    )

    assert video.avi_time(path) == datetime(2024, 3, 3, 10, 22, 33)


def test_read_time_falls_back_to_the_file_date(tmp_path):
    path = tmp_path / "clip.MP4"
    path.write_bytes(mp4(None))
    stamp = datetime(2024, 5, 1, 12).timestamp()
    os.utime(path, (stamp, stamp))

    assert video.read_time(path) == (datetime(2024, 5, 1, 12), False)


def test_ingest_with_videos(make_image, tmp_path):
    make_image("cards/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 08:00:00")
    (tmp_path / "cards/camera_A/VID_0002.MP4").write_bytes(mp4(datetime(2024, 3, 1, 8, 1)))
    (tmp_path / "cards/camera_A/notes.txt").write_text("x")

    without = ingest.build_media(tmp_path / "cards")
    media = ingest.build_media(tmp_path / "cards", videos=True)

    assert without["fileName"].tolist() == ["IMG_0001.JPG"]
    assert media["fileMediatype"].tolist() == ["image/jpeg", "video/mp4"]
    assert media["timestamp"].iloc[1] == datetime(2024, 3, 1, 8, 1)
    assert ingest.skipped_files(tmp_path / "cards") == {".mp4": 1, ".txt": 1}
    assert ingest.skipped_files(tmp_path / "cards", videos=True) == {".txt": 1}


def detection(label, conf):
    return {"label": label, "conf": conf, "bbox": [0, 0, 1, 1]}


def test_best_frame_prefers_the_most_confident_animal():
    frames = [
        {"filepath": "f0", "detections": [detection("human", 0.99)]},
        {"filepath": "f1", "detections": [detection("animal", 0.6)]},
        {"filepath": "f2", "detections": [detection("animal", 0.9), detection("animal", 0.1)]},
        {"filepath": "f3", "failures": ["DETECTOR"]},
    ]

    assert detect.best_frame(frames)["filepath"] == "f2"
    assert detect.best_frame(frames[:1])["filepath"] == "f0"
    assert detect.best_frame([{"filepath": "f4", "detections": []}])["filepath"] == "f4"
    assert detect.best_frame(frames[3:]) is None


def test_detect_runs_on_video_frames(make_image, monkeypatch, tmp_path):
    make_image("cards/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 08:00:00")
    (tmp_path / "cards/camera_A/VID_0002.MP4").write_bytes(mp4(datetime(2024, 3, 1, 8, 1)))
    ingest.run(tmp_path / "cards", tmp_path / "package", videos=True)

    def sample_frames(path, out_dir, interval, max_frames):
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = [out_dir / f"frame_{number:03d}.jpg" for number in range(3)]
        for frame in frames:
            frame.write_bytes(b"jpeg")
        return frames

    class SpeciesNet:
        def __init__(self, model):
            pass

        def predict(self, filepaths, country, admin1_region, predictions_json, progress_bars):
            confidences = {"frame_000.jpg": 0.3, "frame_001.jpg": 0.95, "frame_002.jpg": 0.5, "IMG_0001.JPG": 0.8}
            predictions = [
                {"filepath": path, "detections": [detection("animal", confidences[Path(path).name])], "model_version": "x"}
                for path in filepaths
            ]
            Path(predictions_json).write_text(json.dumps({"predictions": predictions}))

    module = types.ModuleType("speciesnet")
    module.SpeciesNet = SpeciesNet
    monkeypatch.setitem(sys.modules, "speciesnet", module)
    monkeypatch.setattr(video, "sample_frames", sample_frames)

    results = json.loads(detect.run(tmp_path / "package", tmp_path / "cards").read_text())["predictions"]

    assert [prediction["filepath"] for prediction in results] == ["camera_A/IMG_0001.JPG", "camera_A/VID_0002.MP4"]
    clip = results[1]
    assert clip["frame"] == "frames/camera_A/VID_0002.MP4/frame_001.jpg"
    assert clip["frames"] == 3
    assert clip["detections"][0]["conf"] == 0.95


def test_sample_frames_from_a_real_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for shade in range(50):
        writer.write(numpy.full((48, 64, 3), shade * 5, dtype=numpy.uint8))
    writer.release()

    frames = video.sample_frames(path, tmp_path / "frames", interval=1.0, max_frames=10)

    assert [frame.name for frame in frames] == [f"frame_{number:03d}.jpg" for number in range(5)]
    assert video.sample_frames(path, tmp_path / "few", interval=0.1, max_frames=4)[-1].name == "frame_003.jpg"
    assert pd.Series([frame.stat().st_size for frame in frames]).gt(0).all()
