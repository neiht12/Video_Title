from __future__ import annotations

import unittest

import numpy as np

from src.detect import DetectionError, Detector
from src.lines import LineMergeConfig, merge_lines
from src.schemas import TextRegion


def region(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> TextRegion:
    return TextRegion(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), score)


class FakePredictor:
    def __init__(self, polygons: list[list[list[float]]], scores: list[float]):
        self.polygons = polygons
        self.scores = scores
        self.input_shape: tuple[int, ...] | None = None

    def predict(self, *, input: np.ndarray) -> list[dict]:
        self.input_shape = input.shape
        return [{"dt_polys": self.polygons, "dt_scores": self.scores}]


class DetectTests(unittest.TestCase):
    def test_roi_crop_and_offset_are_applied_once(self) -> None:
        fake = FakePredictor([[[10, 2], [30, 2], [30, 12], [10, 12]]], [0.87])
        detector = Detector(predictor=fake)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        detected = detector.detect(frame)
        self.assertEqual(fake.input_shape, (45, 200, 3))
        self.assertEqual(detected[0].polygon_xy,
                         ((10.0, 57.0), (30.0, 57.0), (30.0, 67.0), (10.0, 67.0)))
        self.assertEqual(detected[0].score, 0.87)

    def test_roi_polygons_are_clamped_to_frame_boundaries(self) -> None:
        fake = FakePredictor(
            [[[-10, -4], [205, -4], [205, 50], [-10, 50]],
             [[220, 4], [230, 4], [230, 10], [220, 10]]],
            [0.8, 0.9],
        )
        regions = Detector(predictor=fake).detect(np.zeros((100, 200, 3), np.uint8))
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].polygon_xy,
                         ((0.0, 55.0), (200.0, 55.0), (200.0, 100.0), (0.0, 100.0)))
        regions[0].validate_frame_bounds(200, 100)

    def test_malformed_paddle_result_raises_clear_error(self) -> None:
        fake = FakePredictor([[[0, 0], [10, 0], [10, 10], [0, 10]]], [])
        with self.assertRaisesRegex(DetectionError, "different lengths"):
            Detector(predictor=fake).detect(np.zeros((100, 200, 3), np.uint8))

    def test_malformed_polygon_is_skipped(self) -> None:
        fake = FakePredictor(
            [[["bad", 0], [10, 0], [10, 10]], [[2, 2], [8, 2], [8, 8], [2, 8]]],
            [0.9, 0.8],
        )
        regions = Detector(predictor=fake).detect(np.zeros((100, 200, 3), np.uint8))
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].score, 0.8)


class MergeLinesTests(unittest.TestCase):
    def test_regions_on_same_row_merge_and_two_rows_stay_separate(self) -> None:
        regions = [
            region(12, 100, 45, 125),
            region(51, 101, 90, 126),
            region(30, 155, 70, 180),
        ]
        lines = merge_lines(regions, (200, 200))
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].bbox_xyxy, (9, 98, 93, 128))
        self.assertEqual(len(lines[0].regions), 2)
        self.assertEqual(lines[1].bbox_xyxy, (27, 153, 73, 182))
        self.assertEqual(len(lines[1].regions), 1)

    def test_one_character_narrow_line_is_kept(self) -> None:
        lines = merge_lines([region(96, 165, 105, 188)], (200, 200))
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].bbox_xyxy, (93, 163, 108, 190))

    def test_padding_clamps_to_frame_and_weak_or_invalid_regions_drop(self) -> None:
        lines = merge_lines(
            [region(0, 0, 8, 10), region(188, 190, 200, 200),
             region(20, 20, 40, 40, score=0.1), region(195, 5, 205, 15)],
            (200, 200), LineMergeConfig(padding_x=5, padding_y=6),
        )
        self.assertEqual([line.bbox_xyxy for line in lines],
                         [(0, 0, 13, 16), (183, 184, 200, 200)])
        for line in lines:
            line.validate_frame_bounds(200, 200)

    def test_far_apart_regions_on_same_row_do_not_merge(self) -> None:
        lines = merge_lines([region(10, 100, 30, 120), region(170, 101, 190, 121)], (200, 200))
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
