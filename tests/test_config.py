from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.config import (
    ConfigError,
    DeviceSpec,
    build_config,
    make_argument_parser,
    parse_device,
    parse_roi_bottom,
    roi_y0,
    validate_device_available,
)
from src.schemas import LineDetection, VideoFrame


class RoiConfigTests(unittest.TestCase):
    def test_roi_limits_and_floor_coordinates(self) -> None:
        self.assertEqual(parse_roi_bottom("0.25"), 0.25)
        self.assertEqual(parse_roi_bottom("0.45"), 0.45)
        self.assertEqual(roi_y0(1280, 0.25), 960)
        self.assertEqual(roi_y0(1280, 0.45), 704)

    def test_roi_rejects_out_of_range_and_non_finite_values(self) -> None:
        for value in (0.249, 0.451, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                parse_roi_bottom(value)


class SchemaTests(unittest.TestCase):
    def test_video_frame_time_uses_exact_source_pts_and_time_base(self) -> None:
        frame = VideoFrame(
            index=0,
            source_pts=512,
            time_base=Fraction(1, 15360),
            time_sec=Fraction(1, 30),
            image_bgr=np.zeros((10, 20, 3), dtype=np.uint8),
        )
        self.assertEqual(frame.time_sec, Fraction(1, 30))
        with self.assertRaisesRegex(ValueError, "source_pts multiplied by time_base"):
            VideoFrame(0, 512, Fraction(1, 15360), Fraction(0), frame.image_bgr)

    def test_line_box_is_half_open_and_must_fit_the_frame(self) -> None:
        line = LineDetection((0, 0, 20, 10), 0.9)
        line.validate_frame_bounds(20, 10)
        with self.assertRaisesRegex(ValueError, "exceeds frame bounds"):
            LineDetection((0, 0, 21, 10), 0.9).validate_frame_bounds(20, 10)


class CliConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.input_path = self.root / "data" / "AI Engineer test.mp4"

    def args(self, *extra: str):
        return make_argument_parser().parse_args(
            [
                "--input", str(self.input_path),
                "--output", str(self.root / "outputs" / "boxed.mp4"),
                "--json", str(self.root / "outputs" / "boxes.json"),
                *extra,
            ]
        )

    def test_default_roi_device_and_mkldnn_setting(self) -> None:
        config = build_config(self.args())
        self.assertEqual(config.roi_bottom, 0.45)
        self.assertEqual(config.device, DeviceSpec("cpu"))
        self.assertFalse(config.enable_mkldnn)

    def test_missing_input_is_rejected(self) -> None:
        missing = self.root / "outputs" / "file-that-does-not-exist.mp4"
        args = make_argument_parser().parse_args(
            ["--input", str(missing), "--output", str(self.root / "outputs" / "out.mp4"),
             "--json", str(self.root / "outputs" / "out.json")]
        )
        with self.assertRaisesRegex(ConfigError, "Input file does not exist"):
            build_config(args)

    def test_device_syntax_validation(self) -> None:
        self.assertEqual(parse_device("cpu"), DeviceSpec("cpu"))
        self.assertEqual(parse_device("gpu"), DeviceSpec("gpu", 0))
        self.assertEqual(parse_device("gpu:2").paddle_name, "gpu:2")
        for value in ("cuda", "gpu:-1", "gpu:x", "gpu:", "gpu:1:2"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                parse_device(value)

    def test_gpu_unavailable_fails_without_cpu_fallback(self) -> None:
        fake_paddle = SimpleNamespace(is_compiled_with_cuda=lambda: False)
        with patch.dict(sys.modules, {"paddle": fake_paddle}):
            with self.assertRaisesRegex(ConfigError, "no CUDA support") as error:
                validate_device_available("gpu:0")
        self.assertIn("No CPU fallback was used", str(error.exception))

    def test_output_cannot_overwrite_input(self) -> None:
        args = make_argument_parser().parse_args(
            ["--input", str(self.input_path), "--output", str(self.input_path),
             "--json", str(self.root / "outputs" / "out.json")]
        )
        with self.assertRaisesRegex(ConfigError, "must not refer to the input file"):
            build_config(args)


if __name__ == "__main__":
    unittest.main()
