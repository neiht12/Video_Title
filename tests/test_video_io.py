from __future__ import annotations

import json
import subprocess
import unittest
import uuid
from fractions import Fraction
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np

from src.schemas import FrameBoxes
from src.video_io import (
    VideoIOError, _apply_rotation, _extract_rotation, _require_executable,
    export_annotated_segment, iter_frames, probe_video,
)


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


class ExportTests(unittest.TestCase):
    """All generated fixtures come from the one supplied MP4 and are removed afterward."""

    @classmethod
    def setUpClass(cls) -> None:
        output_dir = REPO_ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        prefix = f".test_video_io_{uuid.uuid4().hex}"
        cls.av_source = output_dir / f"{prefix}_av.mp4"
        cls.silent_source = output_dir / f"{prefix}_silent.mp4"
        cls.short_audio_source = output_dir / f"{prefix}_short_audio.mp4"
        cls.av_clip = output_dir / f"{prefix}_av_clip.mp4"
        cls.silent_clip = output_dir / f"{prefix}_silent_clip.mp4"
        cls.silent_segment_clip = output_dir / f"{prefix}_silent_segment_clip.mp4"
        cls.full_clip = output_dir / f"{prefix}_full_clip.mp4"
        cls.short_audio_clip = output_dir / f"{prefix}_short_audio_clip.mp4"
        cls.fallback_clip = output_dir / f"{prefix}_fallback_clip.mp4"
        cls.timeline_clip = output_dir / f"{prefix}_timeline_clip.mp4"
        cls.bad_clip = output_dir / f"{prefix}_bad_clip.mp4"
        cls.all_paths = [
            cls.av_source, cls.silent_source, cls.short_audio_source, cls.av_clip,
            cls.silent_clip, cls.silent_segment_clip, cls.full_clip, cls.short_audio_clip,
            cls.fallback_clip, cls.timeline_clip,
            cls.bad_clip,
        ]
        cls.addClassCleanup(cls._cleanup)
        cls.ffmpeg = _require_executable("ffmpeg")
        cls.ffprobe = _require_executable("ffprobe")
        cls._ffmpeg([
            "-ss", "12", "-i", str(SOURCE_VIDEO), "-t", "1", "-map", "0:v:0",
            "-map", "0:a:0", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "30", "-c:a", "aac", "-b:a", "64k", str(cls.av_source),
        ])
        cls._ffmpeg(["-i", str(cls.av_source), "-c:v", "copy", "-an", str(cls.silent_source)])
        cls._ffmpeg([
            "-i", str(cls.av_source), "-filter_complex",
            "[0:a:0]atrim=start=0:end=0.4,asetpts=PTS-STARTPTS[a]",
            "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
            str(cls.short_audio_source),
        ])

    @classmethod
    def _cleanup(cls) -> None:
        for path in cls.all_paths:
            path.unlink(missing_ok=True)
        if any(path.exists() for path in cls.all_paths):
            raise AssertionError("Video I/O test fixtures were not removed")

    @classmethod
    def _ffmpeg(cls, args: list[str]) -> None:
        result = subprocess.run(
            [cls.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise AssertionError(result.stderr)

    @classmethod
    def _probe(cls, path: Path) -> list[dict]:
        result = subprocess.run(
            [cls.ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)["streams"]

    def _assert_decodes(self, path: Path, video_frames: int, audio_streams: int) -> None:
        streams = self._probe(path)
        video = [stream for stream in streams if stream["codec_type"] == "video"]
        audio = [stream for stream in streams if stream["codec_type"] == "audio"]
        self.assertEqual(len(video), 1)
        self.assertEqual(len(audio), audio_streams)
        self.assertEqual(int(video[0]["nb_frames"]), video_frames)
        self.assertEqual((video[0]["width"], video[0]["height"]), (720, 1280))
        for stream in audio:
            self.assertEqual(stream["codec_name"], "aac")
        result = subprocess.run(
            [self.ffmpeg, "-v", "error", "-xerror", "-i", str(path),
             "-map", "0", "-f", "null", "NUL"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with av.open(str(path)) as container:
            self.assertEqual(sum(1 for _ in container.decode(video=0)), video_frames)
        self.assertFalse(list(path.parent.glob(f".{path.stem}.*.stage.mp4")))
        self.assertFalse(list(path.parent.glob(f".{path.stem}.*.mux.mp4")))

    def test_precise_clip_reencodes_audio_and_decodes(self) -> None:
        result = export_annotated_segment(
            self.av_source, self.av_clip, start_sec=Fraction(1, 5),
            duration_sec=Fraction(3, 5), boxes_xyxy=[(8, 8, 48, 48)],
        )
        self.assertEqual(result.frame_count, 18)
        self.assertIn("segment-accurate", result.audio_mode)
        self._assert_decodes(self.av_clip, 18, 1)

    def test_silent_source_exports_without_audio(self) -> None:
        self.assertFalse(probe_video(self.silent_source).has_audio)
        result = export_annotated_segment(
            self.silent_source, self.silent_clip, boxes_xyxy=[],
        )
        self.assertEqual(result.audio_mode, "none (source has no audio)")
        self._assert_decodes(self.silent_clip, 30, 0)
        segment = export_annotated_segment(
            self.silent_source, self.silent_segment_clip,
            start_sec=Fraction(1, 5), duration_sec=Fraction(3, 5), boxes_xyxy=[],
        )
        self.assertEqual(segment.audio_mode, "none (source has no audio)")
        self._assert_decodes(self.silent_segment_clip, 18, 0)

    def test_full_video_remux_preserves_aac_packets(self) -> None:
        result = export_annotated_segment(
            self.av_source, self.full_clip, boxes_xyxy=[],
        )
        self.assertEqual(result.audio_mode, "AAC remux (stream copy)")
        self._assert_decodes(self.full_clip, 30, 1)
        def packets(path: Path) -> list[bytes]:
            with av.open(str(path)) as container:
                return [bytes(packet) for packet in container.demux(audio=0) if packet.size]
        self.assertEqual(packets(self.av_source), packets(self.full_clip))

    def test_short_audio_does_not_truncate_video(self) -> None:
        source_streams = self._probe(self.short_audio_source)
        self.assertLess(
            float(next(s for s in source_streams if s["codec_type"] == "audio")["duration"]),
            float(next(s for s in source_streams if s["codec_type"] == "video")["duration"]),
        )
        result = export_annotated_segment(
            self.short_audio_source, self.short_audio_clip, boxes_xyxy=[],
        )
        self.assertEqual(result.audio_mode, "AAC remux (stream copy)")
        self._assert_decodes(self.short_audio_clip, 30, 1)
        output_streams = self._probe(self.short_audio_clip)
        self.assertGreater(
            float(next(s for s in output_streams if s["codec_type"] == "video")["duration"]),
            float(next(s for s in output_streams if s["codec_type"] == "audio")["duration"]),
        )

    def test_remux_failure_falls_back_to_aac_encode(self) -> None:
        with patch("src.video_io._remux_audio", side_effect=VideoIOError("remux unavailable")):
            result = export_annotated_segment(
                self.av_source, self.fallback_clip, boxes_xyxy=[],
            )
        self.assertEqual(result.audio_mode, "AAC re-encode (remux unavailable)")
        self._assert_decodes(self.fallback_clip, 30, 1)

    def test_timeline_draws_per_frame_and_checks_index_and_pts(self) -> None:
        iterator = iter_frames(self.silent_source)
        try:
            frames = list(islice(iterator, 2))
        finally:
            iterator.close()
        timeline = [
            FrameBoxes(frame_index=frame.index, time_sec=frame.time_sec,
                       source_pts=frame.source_pts, event_id=None,
                       boxes_xyxy=(() if frame.index == 0 else ((8, 8, 48, 48),)))
            for frame in frames
        ]
        export_annotated_segment(
            self.silent_source, self.timeline_clip, start_sec=0,
            duration_sec=Fraction(2, 30), frame_boxes=timeline,
        )
        self._assert_decodes(self.timeline_clip, 2, 0)
        with av.open(str(self.timeline_clip)) as container:
            decoded = list(container.decode(video=0))
        red_before = int(decoded[0].to_ndarray(format="bgr24")[8, 20, 2])
        red_after = int(decoded[1].to_ndarray(format="bgr24")[8, 20, 2])
        self.assertGreater(red_after, red_before + 50)
        for bad in (
            FrameBoxes(99, frames[0].time_sec, None, (), frames[0].source_pts),
            FrameBoxes(frames[0].index, frames[0].time_sec, None, (), 999),
        ):
            with self.assertRaisesRegex(VideoIOError, "Box timeline mismatch"):
                export_annotated_segment(
                    self.silent_source, self.bad_clip, duration_sec=Fraction(2, 30),
                    frame_boxes=[bad, timeline[1]],
                )
            self.assertFalse(self.bad_clip.exists())
            self.assertFalse(list(self.bad_clip.parent.glob(f".{self.bad_clip.stem}.*.stage.mp4")))
        with self.assertRaisesRegex(VideoIOError, "Box timeline ended before source frame 1"):
            export_annotated_segment(
                self.silent_source, self.bad_clip, duration_sec=Fraction(2, 30),
                frame_boxes=timeline[:1],
            )
        self.assertFalse(self.bad_clip.exists())


if __name__ == "__main__":
    unittest.main()
