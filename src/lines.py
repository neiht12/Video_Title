"""Merge text polygons into separate subtitle-line boxes within frame bounds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.schemas import LineDetection, TextRegion


@dataclass(frozen=True, slots=True)
class LineMergeConfig:
    min_score: float = 0.5
    padding_x: int = 3
    padding_y: int = 2
    max_gap_height_ratio: float = 2.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_score) or not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be finite and between 0 and 1")
        if self.padding_x < 0 or self.padding_y < 0:
            raise ValueError("line padding must be nonnegative")
        if not math.isfinite(self.max_gap_height_ratio) or self.max_gap_height_ratio < 0:
            raise ValueError("max_gap_height_ratio must be finite and nonnegative")


def _bounds(region: TextRegion) -> tuple[float, float, float, float]:
    xs = [point[0] for point in region.polygon_xy]
    ys = [point[1] for point in region.polygon_xy]
    return min(xs), min(ys), max(xs), max(ys)


def _same_row(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> bool:
    ah, bh = a[3] - a[1], b[3] - b[1]
    overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    center_delta = abs((a[1] + a[3] - b[1] - b[3]) / 2)
    baseline_delta = abs(a[3] - b[3])
    return (overlap >= 0.5 * min(ah, bh) and center_delta <= 0.45 * max(ah, bh)
            and baseline_delta <= 0.5 * max(ah, bh))


def _horizontal_gap(a: tuple[float, float, float, float],
                    b: tuple[float, float, float, float]) -> float:
    return max(0.0, max(a[0] - b[2], b[0] - a[2]))


def merge_lines(
    regions: Sequence[TextRegion],
    frame_size: tuple[int, int],
    config: LineMergeConfig | None = None,
) -> list[LineDetection]:
    """Keep one or more valid polygons per line, including narrow one-character lines."""

    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("frame_size must be positive width,height")
    settings = config or LineMergeConfig()
    valid: list[tuple[TextRegion, tuple[float, float, float, float]]] = []
    for region in regions:
        if region.score < settings.min_score:
            continue
        try:
            region.validate_frame_bounds(width, height)
        except ValueError:
            continue
        bounds = _bounds(region)
        if bounds[0] < bounds[2] and bounds[1] < bounds[3]:
            valid.append((region, bounds))
    valid.sort(key=lambda item: ((item[1][1] + item[1][3]) / 2, item[1][0]))

    groups: list[list[tuple[TextRegion, tuple[float, float, float, float]]]] = []
    for region, bounds in valid:
        candidates: list[tuple[float, int]] = []
        for index, group in enumerate(groups):
            if not all(_same_row(bounds, other_bounds) for _, other_bounds in group):
                continue
            group_bounds = (
                min(item[1][0] for item in group), min(item[1][1] for item in group),
                max(item[1][2] for item in group), max(item[1][3] for item in group),
            )
            max_height = max(bounds[3] - bounds[1], group_bounds[3] - group_bounds[1])
            gap = _horizontal_gap(bounds, group_bounds)
            if gap <= max(12.0, settings.max_gap_height_ratio * max_height):
                candidates.append((gap, index))
        if candidates:
            groups[min(candidates)[1]].append((region, bounds))
        else:
            groups.append([(region, bounds)])

    lines: list[LineDetection] = []
    for group in groups:
        group.sort(key=lambda item: item[1][0])
        x1 = max(0, math.floor(min(item[1][0] for item in group)) - settings.padding_x)
        y1 = max(0, math.floor(min(item[1][1] for item in group)) - settings.padding_y)
        x2 = min(width, math.ceil(max(item[1][2] for item in group)) + settings.padding_x)
        y2 = min(height, math.ceil(max(item[1][3] for item in group)) + settings.padding_y)
        if x1 >= x2 or y1 >= y2:
            continue
        line = LineDetection(
            bbox_xyxy=(x1, y1, x2, y2),
            score=sum(item[0].score for item in group) / len(group),
            regions=tuple(item[0] for item in group),
        )
        line.validate_frame_bounds(width, height)
        lines.append(line)
    return sorted(lines, key=lambda line: (line.bbox_xyxy[1], line.bbox_xyxy[0]))
