"""Offline, per-line subtitle tracking over ordered frame observations."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Sequence

import cv2
import numpy as np

from src.config import DEFAULT_ROI_BOTTOM
from src.detect import Detector
from src.lines import LineMergeConfig, merge_lines
from src.schemas import BBoxXYXY, FrameBoxes, FrameObservation, VideoFrame


@dataclass(frozen=True, slots=True)
class EventConfig:
    max_gap_frames: int = 2
    min_confirmed_frames: int = 3
    subtitle_top_ratio: float = 0.62
    subtitle_bottom_ratio: float = 0.80
    min_text_style: float = 0.04
    max_center_distance_heights: float = 0.75
    max_height_ratio: float = 1.7
    fingerprint_change_distance: float = 0.55
    change_confirm_frames: int = 2
    smooth_radius: int = 2
    smooth_safety_px: int = 2

    def __post_init__(self) -> None:
        if self.max_gap_frames < 0 or self.min_confirmed_frames < 1:
            raise ValueError("max_gap_frames must be nonnegative and min_confirmed_frames positive")
        if not 0 <= self.subtitle_top_ratio < self.subtitle_bottom_ratio <= 1:
            raise ValueError("subtitle band must be within the frame")
        if not 0 <= self.min_text_style <= 1:
            raise ValueError("min_text_style must be between 0 and 1")
        if self.max_center_distance_heights <= 0 or self.max_height_ratio < 1:
            raise ValueError("geometry limits must be positive")
        if not 0 < self.fingerprint_change_distance <= 1 or self.change_confirm_frames < 2:
            raise ValueError("fingerprint change settings are invalid")
        if self.smooth_radius < 0 or self.smooth_safety_px < 0:
            raise ValueError("smoothing settings must be nonnegative")


def _text_mask(frame_bgr: np.ndarray, box: BBoxXYXY) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
    right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
    if right <= left or bottom <= top:
        return np.zeros((1, 1), dtype=np.uint8)
    crop = frame_bgr[top:bottom, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    white = (hsv[:, :, 2] > 190) & (hsv[:, :, 1] < 80)
    near_dark = cv2.dilate((gray < 75).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return (white & near_dark).astype(np.uint8)


def line_signature(frame_bgr: np.ndarray, box: BBoxXYXY) -> tuple[float, bytes]:
    """Measure edged white text and fingerprint it in fixed frame coordinates.

    Normalizing each detected crop to a fixed size made a one-pixel box jitter
    move every stroke of a short line such as 田.  Keep the strokes at their
    source coordinates so a box change affects only pixels near its boundary.
    """

    mask = _text_mask(frame_bgr, box)
    style_score = float(mask.mean())
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
    right, bottom = min(width, math.ceil(x2)), min(height, math.ceil(y2))
    canvas = np.zeros((height, width), dtype=np.uint8)
    if right > left and bottom > top:
        canvas[top:bottom, left:right] = mask
    normalized = cv2.resize(
        canvas, ((width + 3) // 4, (height + 3) // 4), interpolation=cv2.INTER_AREA,
    ) > 0.2
    return style_score, np.packbits(normalized.reshape(-1)).tobytes()


def observe_frame(
    frame: VideoFrame,
    detector: Detector,
    *,
    roi_bottom: float = DEFAULT_ROI_BOTTOM,
    line_config: LineMergeConfig | None = None,
) -> FrameObservation:
    height, width = frame.image_bgr.shape[:2]
    regions = detector.detect(frame.image_bgr, roi_bottom)
    lines = tuple(merge_lines(regions, (width, height), line_config))
    signatures = [line_signature(frame.image_bgr, line.bbox_xyxy) for line in lines]
    return FrameObservation(
        frame_index=frame.index,
        time_sec=frame.time_sec,
        lines=lines,
        fingerprints=tuple(value[1] for value in signatures),
        source_pts=frame.source_pts,
        style_scores=tuple(value[0] for value in signatures),
    )


def fingerprint_distance(a: bytes | str | None, b: bytes | str | None) -> float:
    if a is None or b is None:
        return 0.0
    if isinstance(a, str) or isinstance(b, str):
        return 0.0 if a == b else 1.0
    if len(a) != len(b):
        return 1.0
    bits_a = np.unpackbits(np.frombuffer(a, dtype=np.uint8)).astype(bool)
    bits_b = np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(bool)
    union = int(np.logical_or(bits_a, bits_b).sum())
    if not union:
        return 0.0
    intersection = int(np.logical_and(bits_a, bits_b).sum())
    return 1.0 - intersection / union


def _geometry_cost(a: BBoxXYXY, b: BBoxXYXY, config: EventConfig) -> float | None:
    ah, bh = a[3] - a[1], b[3] - b[1]
    if max(ah, bh) / min(ah, bh) > config.max_height_ratio:
        return None
    dy = abs((a[1] + a[3] - b[1] - b[3]) / 2) / max(ah, bh)
    if dy > config.max_center_distance_heights:
        return None
    y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / min(ah, bh)
    if y_overlap < 0.25:
        return None
    x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    x_union = max(a[2], b[2]) - min(a[0], b[0])
    x_iou = x_overlap / x_union if x_union else 0.0
    aw, bw = a[2] - a[0], b[2] - b[0]
    dx = abs((a[0] + a[2] - b[0] - b[2]) / 2) / max(aw, bw)
    if x_iou < 0.10 and dx > 0.35:
        return None
    return dy + 0.3 * (1 - x_iou) + 0.2 * (1 - y_overlap) + 0.2 * dx


@dataclass(slots=True)
class _Detection:
    frame_pos: int
    line_index: int
    box: BBoxXYXY
    fingerprint: bytes | str | None


@dataclass(slots=True)
class _Track:
    id: int
    detections: list[_Detection] = field(default_factory=list)

    @property
    def last(self) -> _Detection:
        return self.detections[-1]


@dataclass(frozen=True, slots=True)
class LineEvent:
    id: int
    start_frame: int
    end_frame_exclusive: int
    start_sec: Fraction
    end_sec: Fraction
    detected_frames: int
    filled_frames: int
    median_box: BBoxXYXY


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    frame_index: int
    source_pts: int
    time_sec: Fraction
    raw_boxes: tuple[BBoxXYXY, ...]
    tracked_boxes: tuple[BBoxXYXY, ...]
    draw_boxes: tuple[BBoxXYXY, ...]
    event_ids: tuple[int, ...]
    rejected: tuple[tuple[int, str], ...]

    def as_frame_boxes(self) -> FrameBoxes:
        event_id: int | str | None
        if not self.event_ids:
            event_id = None
        elif len(self.event_ids) == 1:
            event_id = self.event_ids[0]
        else:
            event_id = "+".join(str(value) for value in self.event_ids)
        return FrameBoxes(
            frame_index=self.frame_index, time_sec=self.time_sec,
            event_id=event_id, boxes_xyxy=self.draw_boxes, source_pts=self.source_pts,
        )


@dataclass(frozen=True, slots=True)
class TrackingResult:
    frames: tuple[TrackedFrame, ...]
    events: tuple[LineEvent, ...]

    def frame_boxes(self) -> tuple[FrameBoxes, ...]:
        return tuple(frame.as_frame_boxes() for frame in self.frames)


def _split_on_content(detections: list[_Detection], config: EventConfig) -> list[list[_Detection]]:
    if not detections:
        return []
    segments: list[list[_Detection]] = []
    start = 0
    while start < len(detections):
        refs = [item.fingerprint for item in detections[start:start + 3]
                if item.fingerprint is not None]
        if not refs:
            segments.append(detections[start:])
            break
        reference = min(refs, key=lambda fp: sum(fingerprint_distance(fp, other) for other in refs))
        pending: list[int] = []
        split_at: int | None = None
        for index in range(start + 1, len(detections)):
            current = detections[index].fingerprint
            if current is None or fingerprint_distance(reference, current) < config.fingerprint_change_distance:
                pending.clear()
                continue
            pending.append(index)
            if len(pending) >= config.change_confirm_frames:
                recent = pending[-config.change_confirm_frames:]
                if all(recent[i] == recent[0] + i for i in range(len(recent))) and all(
                    fingerprint_distance(detections[recent[0]].fingerprint,
                                         detections[other].fingerprint) < config.fingerprint_change_distance
                    for other in recent[1:]
                ):
                    split_at = recent[0]
                    break
        if split_at is None:
            segments.append(detections[start:])
            break
        segments.append(detections[start:split_at])
        start = split_at
    return segments


def _smooth_box(
    detections: list[_Detection], index: int, width: int, height: int, config: EventConfig,
) -> BBoxXYXY:
    current = detections[index].box
    near = [item.box for item in detections
            if abs(item.frame_pos - detections[index].frame_pos) <= config.smooth_radius]
    median = [statistics.median(box[axis] for box in near) for axis in range(4)]
    margin = config.smooth_safety_px
    return (
        max(0, min(current[0], median[0] - margin)),
        max(0, min(current[1], median[1] - margin)),
        min(width, max(current[2], median[2] + margin)),
        min(height, max(current[3], median[3] + margin)),
    )


def track_observations(
    observations: Sequence[FrameObservation],
    frame_size: tuple[int, int],
    config: EventConfig | None = None,
) -> TrackingResult:
    """Track accepted lines, split confirmed text changes, then fill internal gaps."""

    settings = config or EventConfig()
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("frame_size must be positive width,height")
    if not observations:
        return TrackingResult((), ())
    for pos, obs in enumerate(observations):
        if obs.source_pts is None:
            raise ValueError(f"source_pts missing at frame {obs.frame_index}")
        if pos and (obs.frame_index != observations[pos - 1].frame_index + 1
                    or obs.time_sec <= observations[pos - 1].time_sec):
            raise ValueError("observations must have contiguous frame indices and increasing PTS time")
        for line in obs.lines:
            line.validate_frame_bounds(width, height)

    rejected: list[list[tuple[int, str]]] = [[] for _ in observations]
    tracks: list[_Track] = []
    next_track_id = 1
    for pos, obs in enumerate(observations):
        candidates: list[_Detection] = []
        for index, line in enumerate(obs.lines):
            box = line.bbox_xyxy
            if box[1] < height * settings.subtitle_top_ratio or box[3] > height * settings.subtitle_bottom_ratio:
                rejected[pos].append((index, "outside_subtitle_band"))
                continue
            style = obs.style_scores[index] if obs.style_scores else 1.0
            if style < settings.min_text_style:
                rejected[pos].append((index, "weak_text_style"))
                continue
            fingerprint = obs.fingerprints[index] if obs.fingerprints else None
            candidates.append(_Detection(pos, index, box, fingerprint))

        costs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            gap = obs.frame_index - observations[track.last.frame_pos].frame_index - 1
            if gap > settings.max_gap_frames:
                continue
            for candidate_index, candidate in enumerate(candidates):
                cost = _geometry_cost(track.last.box, candidate.box, settings)
                if cost is not None:
                    costs.append((cost + 0.05 * gap, track_index, candidate_index))
        used_tracks: set[int] = set()
        used_candidates: set[int] = set()
        for _, track_index, candidate_index in sorted(costs):
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            tracks[track_index].detections.append(candidates[candidate_index])
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)
        for index, candidate in enumerate(candidates):
            if index not in used_candidates:
                tracks.append(_Track(next_track_id, [candidate]))
                next_track_id += 1

    tracked: list[list[tuple[int, BBoxXYXY]]] = [[] for _ in observations]
    drawn: list[list[tuple[int, BBoxXYXY]]] = [[] for _ in observations]
    events: list[LineEvent] = []
    next_event_id = 1
    deltas = [observations[i].time_sec - observations[i - 1].time_sec
              for i in range(1, len(observations))]
    final_delta = statistics.median(deltas) if deltas else Fraction(1, 30)
    for track in tracks:
        for segment in _split_on_content(track.detections, settings):
            if len(segment) < settings.min_confirmed_frames:
                for item in segment:
                    rejected[item.frame_pos].append((item.line_index, "transient_track"))
                continue
            event_id = next_event_id
            next_event_id += 1
            smooth = [_smooth_box(segment, index, width, height, settings)
                      for index in range(len(segment))]
            for item, box in zip(segment, smooth):
                tracked[item.frame_pos].append((event_id, item.box))
                drawn[item.frame_pos].append((event_id, box))
            filled_frames = 0
            for index in range(1, len(segment)):
                previous, current = segment[index - 1], segment[index]
                gap = current.frame_pos - previous.frame_pos - 1
                if not 0 < gap <= settings.max_gap_frames:
                    continue
                for offset in range(1, gap + 1):
                    ratio = offset / (gap + 1)
                    pos = previous.frame_pos + offset
                    raw_interpolated = tuple(
                        previous.box[axis] * (1 - ratio) + current.box[axis] * ratio
                        for axis in range(4)
                    )
                    smooth_interpolated = tuple(
                        smooth[index - 1][axis] * (1 - ratio) + smooth[index][axis] * ratio
                        for axis in range(4)
                    )
                    tracked[pos].append((event_id, raw_interpolated))
                    drawn[pos].append((event_id, smooth_interpolated))
                    filled_frames += 1
            first_pos, last_pos = segment[0].frame_pos, segment[-1].frame_pos
            end_pos = last_pos + 1
            end_sec = (observations[end_pos].time_sec if end_pos < len(observations)
                       else observations[last_pos].time_sec + final_delta)
            median_box = tuple(statistics.median(item.box[axis] for item in segment)
                               for axis in range(4))
            events.append(LineEvent(
                id=event_id,
                start_frame=observations[first_pos].frame_index,
                end_frame_exclusive=observations[last_pos].frame_index + 1,
                start_sec=observations[first_pos].time_sec,
                end_sec=end_sec,
                detected_frames=len(segment),
                filled_frames=filled_frames,
                median_box=median_box,
            ))

    frames: list[TrackedFrame] = []
    for pos, obs in enumerate(observations):
        ordered_tracked = sorted(tracked[pos], key=lambda item: (item[1][1], item[1][0]))
        ordered_drawn = sorted(drawn[pos], key=lambda item: (item[1][1], item[1][0]))
        frames.append(TrackedFrame(
            frame_index=obs.frame_index,
            source_pts=obs.source_pts,  # type: ignore[arg-type]
            time_sec=obs.time_sec,
            raw_boxes=tuple(line.bbox_xyxy for line in obs.lines),
            tracked_boxes=tuple(box for _, box in ordered_tracked),
            draw_boxes=tuple(box for _, box in ordered_drawn),
            event_ids=tuple(event_id for event_id, _ in ordered_drawn),
            rejected=tuple(sorted(rejected[pos])),
        ))
    return TrackingResult(tuple(frames), tuple(events))


def build_events(
    observations: Sequence[FrameObservation],
    frame_size: tuple[int, int],
    config: EventConfig | None = None,
) -> tuple[tuple[LineEvent, ...], tuple[FrameBoxes, ...]]:
    """Return the per-line event list and the shared draw/export timeline."""

    result = track_observations(observations, frame_size, config)
    return result.events, result.frame_boxes()
