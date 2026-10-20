"""Camera-trap videos: recording times, and frames for detection.

Trail cameras store when a clip was recorded inside the file: MP4 and MOV files in
the ``mvhd`` header (seconds since 1904), AVI files in an ``IDIT`` chunk (text such as
``MON MAR 03 10:22:33 2024``). Like EXIF times in images, these are the camera's
local clock time. When a file has no such time, its modification date is used.
"""

import struct
from datetime import datetime, timedelta
from pathlib import Path

VIDEO_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo"}
QUICKTIME_EPOCH = datetime(1904, 1, 1)
AVI_TIME_FORMATS = ["%a %b %d %H:%M:%S %Y", "%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]
HEADER_BYTES = 1_000_000
DEFAULT_FRAME_INTERVAL = 1.0
DEFAULT_MAX_FRAMES = 10
FRAMES_FOLDER = "frames"


def is_video(path: Path) -> bool:
    """Return True for MP4, MOV and AVI files, skipping hidden files."""
    return path.suffix.lower() in VIDEO_TYPES and not path.name.startswith(".")


def media_type(path: Path) -> str:
    """Return the media type of a video file, e.g. ``video/mp4``."""
    return VIDEO_TYPES[path.suffix.lower()]


def boxes(data: bytes, start: int = 0, end: int | None = None):
    """Yield the type, data start and end of each MP4/QuickTime box in ``data``."""
    end = len(data) if end is None else end
    position = start
    while position + 8 <= end:
        size, kind = struct.unpack(">I4s", data[position : position + 8])
        header = 8
        if size == 1 and position + 16 <= end:
            size = struct.unpack(">Q", data[position + 8 : position + 16])[0]
            header = 16
        elif size == 0:
            size = end - position
        if size < header:
            return
        yield kind, position + header, min(position + size, end)
        position += size


def top_level_box(path: Path, wanted: bytes) -> bytes | None:
    """Return the contents of the first top-level box of type ``wanted`` in an MP4 or
    MOV file, stepping from box to box by their sizes, so it is found whether it comes
    before or after the video data."""
    with path.open("rb") as file:
        size_of_file = path.stat().st_size
        position = 0
        while position + 8 <= size_of_file:
            file.seek(position)
            header = file.read(16)
            size, kind = struct.unpack(">I4s", header[:8])
            offset = 8
            if size == 1:
                size, offset = struct.unpack(">Q", header[8:16])[0], 16
            elif size == 0:
                size = size_of_file - position
            if size < offset:
                return None
            if kind == wanted:
                file.seek(position + offset)
                return file.read(min(size - offset, HEADER_BYTES))
            position += size
    return None


def quicktime_time(path: Path) -> datetime | None:
    """Return the creation time in the ``mvhd`` header of an MP4 or MOV file."""
    movie = top_level_box(path, b"moov")
    if movie is None:
        return None
    for kind, start, _ in boxes(movie):
        if kind == b"mvhd":
            version = movie[start]
            if version == 1:
                seconds = struct.unpack(">Q", movie[start + 4 : start + 12])[0]
            else:
                seconds = struct.unpack(">I", movie[start + 4 : start + 8])[0]
            return QUICKTIME_EPOCH + timedelta(seconds=seconds) if seconds else None
    return None


def avi_time(path: Path) -> datetime | None:
    """Return the recording time in the ``IDIT`` chunk of an AVI file."""
    with path.open("rb") as file:
        data = file.read(HEADER_BYTES)
    position = data.find(b"IDIT")
    if position < 0:
        return None
    length = struct.unpack("<I", data[position + 4 : position + 8])[0]
    text = data[position + 8 : position + 8 + length].split(b"\0")[0].decode("ascii", errors="ignore").strip()
    for time_format in AVI_TIME_FORMATS:
        try:
            return datetime.strptime(text.title() if "%a" in time_format else text, time_format)
        except ValueError:
            continue
    return None


def read_time(path: Path) -> tuple[datetime | None, bool]:
    """Return when a video was recorded and whether that time was stored in the file
    (``False`` means the file's modification date was used instead)."""
    try:
        recorded = avi_time(path) if path.suffix.lower() == ".avi" else quicktime_time(path)
    except (OSError, struct.error):
        recorded = None
    if recorded is not None:
        return recorded, True
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).replace(microsecond=0), False
    except OSError:
        return None, False


def sample_frames(
    path: Path,
    out_dir: Path,
    interval: float = DEFAULT_FRAME_INTERVAL,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> list[Path]:
    """Save one frame every ``interval`` seconds of a video (at most ``max_frames``,
    spread over the clip) as JPEG images in ``out_dir`` and return their paths.
    Needs OpenCV, which the ``detect`` extra installs."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        rate = capture.get(cv2.CAP_PROP_FPS) or 0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return []
        step = max(round(rate * interval), 1) if rate > 0 else max(total // max_frames, 1)
        positions = list(range(0, total, step))
        if len(positions) > max_frames:
            positions = [positions[round(index * (len(positions) - 1) / (max_frames - 1))] for index in range(max_frames)]
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for number, position in enumerate(positions):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                continue
            target = out_dir / f"frame_{number:03d}.jpg"
            cv2.imwrite(str(target), frame)
            saved.append(target)
        return saved
    finally:
        capture.release()
