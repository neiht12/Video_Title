from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

class VideoIOError(RuntimeError):
    "Base class for video I/O errors."

@dataclass (frozen=True, slots=True)
class VideoInfo:
    "Metadata information about a video file."

    path: Path
    width: int
    height: int
    fps: Fraction
    duration_ms: int | None
    frame_count: int | None
    video_codec: str | None
    audio_codec: str | None
    has_audio: bool
    is_variable_framerate: bool
    rotation: int | None

@dataclass ( slots=True)
class VideoFrame:
    "A single frame of a video."

    index: int
    timestamp_ms: int
    image: np.ndarray  # This should be a type that represents an image, e.g., numpy.ndarray or PIL.Image

def _require_executable(name: str) -> str:

        executable_path = shutil.which(name)
        if executable_path is None:
            raise VideoIOError(
                f"Required executable '{name}' not found in PATH."
                f" Please install '{name}' and ensure it is available in your system's PATH."
                )
        return executable_path

def _extract_rotation(video_stream: dict) -> int | None:
        "Extract the rotation metadata from the video stream."
        tags = video_stream.get("tags", {})
        rotate_str = tags.get("rotate")
        if rotate_str is not None:
            try:
                return int(round(float(rotate_str))) % 360
            except (TypeError, ValueError):
                pass
        for side_data in video_stream.get("side_data_list", []):
            if side_data.get("side_data_type") == "Display Matrix":
                rotation = side_data.get("rotation")
                if rotation is not None:
                    try:
                        return int(round(float(rotation))) % 360
                    except (TypeError, ValueError):
                        pass
        return 0

def probe_video(video_path: Path) -> VideoInfo:

        path = Path(video_path).expanduser().resolve()

        if not path.is_file():
            raise VideoIOError(f"Video file '{path}' does not exist.")

        ffprobe = _require_executable("ffprobe")
        