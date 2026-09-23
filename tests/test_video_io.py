from __future__ import annotations

import unittest
from fractions import Fraction
from itertools import islice
from pathlib import Path

import numpy as np

from src.video_io import _apply_rotation, _extract_rotation, iter_frames, probe_video


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_VIDEO = REPO_ROOT / "data" / "AI Engineer test.mp4"


class VideoIOTests(unittest.TestCase):
    """Use the supplied MP4 as a read-only fixture; no temporary directories are created."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.info = probe_video(SOURCE_VIDEO)

    def test_probe_reports_rotated_dimensions_time_base_and_audio(self) -> None:
        self.assertEqual((self.info.width, self.info.height), (720, 1280))
        self.assertEqual(self.info.video_time_base, Fraction(1, 15360))
        self.assertEqual(self.info.reference_fps, Fraction(30, 1))
        self.assertEqual(self.info.rotation, 0)
        self.assertEqual(self.info.frame_count, 3733)
        self.assertTrue(self.info.has_audio)
        self.assertEqual(self.info.audio_streams[0].codec, "aac")

    def test_frames_keep_source_pts_and_exact_time(self) -> None:
        frames = iter_frames(SOURCE_VIDEO)
        try:
            first, second = islice(frames, 2)
        finally:
            frames.close()
        self.assertEqual((first.index, first.source_pts, first.time_sec), (0, 0, Fraction(0)))
        self.assertEqual((second.index, second.source_pts, second.time_sec), (1, 512, Fraction(1, 30)))
        self.assertEqual(first.image_bgr.shape, (1280, 720, 3))

    def test_display_rotation_is_applied_counterclockwise_once(self) -> None:
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        image[0, 0] = (1, 2, 3)
        rotated = _apply_rotation(image, _extract_rotation({"side_data_list": [{
            "side_data_type": "Display Matrix", "rotation": 90,
        }]}))
        self.assertEqual(rotated.shape, (3, 2, 3))
        self.assertTupleEqual(tuple(rotated[2, 0]), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
