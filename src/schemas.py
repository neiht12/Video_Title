"""Canonical data types for the OCR pipeline's coordinates and timeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias

import numpy as np


PointXY: TypeAlias = tuple[float, float]
BBoxXYXY: TypeAlias = tuple[float, float, float, float]


def _valid_fraction(value: Fraction, name: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be fractions.Fraction")


def _validate_score(score: float) -> None:
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be finite and between 0 and 1")


def _validate_box(box: BBoxXYXY) -> None:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError("bbox_xyxy must contain four finite coordinates")
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
        raise ValueError("bbox_xyxy must be a nonnegative, nonempty half-open box")


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    index: int
    codec: str | None
    time_base: Fraction
    start_time: Fraction | None = None
    duration: Fraction | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("audio stream index must be nonnegative")
        _valid_fraction(self.time_base, "time_base")
        if self.time_base <= 0:
            raise ValueError("audio stream time_base must be positive")
        _valid_fraction(self.start_time, "start_time", allow_none=True)
        _valid_fraction(self.duration, "duration", allow_none=True)


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Metadata with dimensions after applying any display rotation."""

    width: int
    height: int
    codec: str | None
    reference_fps: Fraction | None
    video_time_base: Fraction
    audio_streams: tuple[AudioStreamInfo, ...] = ()
    start_time: Fraction | None = None
    duration: Fraction | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video width and height must be positive")
        _valid_fraction(self.reference_fps, "reference_fps", allow_none=True)
        if self.reference_fps is not None and self.reference_fps <= 0:
            raise ValueError("reference_fps must be positive when present")
        _valid_fraction(self.video_time_base, "video_time_base")
        if self.video_time_base <= 0:
            raise ValueError("video_time_base must be positive")
        _valid_fraction(self.start_time, "start_time", allow_none=True)
        _valid_fraction(self.duration, "duration", allow_none=True)
        if self.duration is not None and self.duration < 0:
            raise ValueError("video duration cannot be negative")


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One decoded presentation-order frame with its source PTS unchanged."""

    index: int
    source_pts: int
    time_base: Fraction
    time_sec: Fraction
    image_bgr: np.ndarray

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("frame index must be nonnegative")
        _valid_fraction(self.time_base, "time_base")
        _valid_fraction(self.time_sec, "time_sec")
        if self.time_base <= 0:
            raise ValueError("frame time_base must be positive")
        if self.time_sec != Fraction(self.source_pts) * self.time_base:
            raise ValueError("time_sec must equal source_pts multiplied by time_base")
        if not isinstance(self.image_bgr, np.ndarray) or self.image_bgr.ndim != 3 or self.image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be a three-channel BGR ndarray")


@dataclass(frozen=True, slots=True)
class TextRegion:
    """Text detector polygon in full-frame pixel coordinates."""

    polygon_xy: tuple[PointXY, ...]
    score: float

    def __post_init__(self) -> None:
        if len(self.polygon_xy) < 3:
            raise ValueError("polygon_xy must have at least three points")
        if any(len(point) != 2 or not all(math.isfinite(c) for c in point) for point in self.polygon_xy):
            raise ValueError("polygon_xy coordinates must be finite x,y pairs")
        if any(x < 0 or y < 0 for x, y in self.polygon_xy):
            raise ValueError("polygon_xy coordinates must be nonnegative full-frame positions")
        _validate_score(self.score)

    def validate_frame_bounds(self, width: int, height: int) -> None:
        if any(x > width or y > height for x, y in self.polygon_xy):
            raise ValueError(f"polygon_xy exceeds frame bounds {width}x{height}")


@dataclass(frozen=True, slots=True)
class LineDetection:
    bbox_xyxy: BBoxXYXY
    score: float
    regions: tuple[TextRegion, ...] = ()

    def __post_init__(self) -> None:
        _validate_box(self.bbox_xyxy)
        _validate_score(self.score)

    def validate_frame_bounds(self, width: int, height: int) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 > width or y2 > height:
            raise ValueError(f"bbox_xyxy {self.bbox_xyxy} exceeds frame bounds {width}x{height}")


@dataclass(frozen=True, slots=True)
class FrameObservation:
    frame_index: int
    time_sec: Fraction
    lines: tuple[LineDetection, ...]
    fingerprints: tuple[bytes | str, ...] = ()

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        _valid_fraction(self.time_sec, "time_sec")
        if self.fingerprints and len(self.fingerprints) != len(self.lines):
            raise ValueError("fingerprints must contain one value per line when provided")


@dataclass(frozen=True, slots=True)
class FrameBoxes:
    frame_index: int
    time_sec: Fraction
    event_id: int | str | None
    boxes_xyxy: tuple[BBoxXYXY, ...]

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        _valid_fraction(self.time_sec, "time_sec")
        for box in self.boxes_xyxy:
            _validate_box(box)

    def validate_frame_bounds(self, width: int, height: int) -> None:
        for box in self.boxes_xyxy:
            x1, y1, x2, y2 = box
            if x2 > width or y2 > height:
                raise ValueError(f"box {box} exceeds frame bounds {width}x{height}")


@dataclass(frozen=True, slots=True)
class SubtitleLine:
    bbox_xyxy: BBoxXYXY
    text: str | None = None

    def __post_init__(self) -> None:
        _validate_box(self.bbox_xyxy)


@dataclass(frozen=True, slots=True)
class SubtitleEvent:
    id: int | str
    start_frame: int
    end_frame_exclusive: int
    start_sec: Fraction
    end_sec: Fraction
    lines: tuple[SubtitleLine, ...]

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame_exclusive <= self.start_frame:
            raise ValueError("event frames must form a nonempty [start, end) interval")
        _valid_fraction(self.start_sec, "start_sec")
        _valid_fraction(self.end_sec, "end_sec")
        if self.end_sec <= self.start_sec:
            raise ValueError("event times must form a nonempty [start, end) interval")
