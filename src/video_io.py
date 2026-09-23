"""Video probing, presentation-order decoding, and an audio-preserving smoke export."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Sequence

import av
import cv2
import numpy as np

from src.schemas import AudioStreamInfo, BBoxXYXY, FrameBoxes, LineDetection, VideoFrame, VideoInfo


class VideoIOError(RuntimeError):
    """Raised when probing, decoding, encoding, or muxing a video fails."""


@dataclass(frozen=True, slots=True)
class SegmentExport:
    output_path: Path
    frame_count: int
    first_source_pts: int
    first_source_time: Fraction
    end_source_time: Fraction
    audio_mode: str

    @property
    def duration(self) -> Fraction:
        return self.end_source_time - self.first_source_time


def _require_executable(name: str) -> str:
    configured_dir = os.environ.get("FFMPEG_BIN")
    if configured_dir:
        configured_path = Path(configured_dir).expanduser()
        candidate = configured_path / (f"{name}.exe" if configured_path.is_dir() else "")
        if candidate.is_file():
            return str(candidate.resolve())
    executable = shutil.which(name)
    if executable is None:
        raise VideoIOError(
            f"Required executable '{name}' was not found in PATH. "
            "Install FFmpeg and restart the terminal so PATH is refreshed."
        )
    return executable


def _extract_rotation(video_stream: dict) -> int:
    """Read FFprobe display-matrix rotation, falling back to the legacy tag."""

    rotation_value: object | None = None
    for side_data in video_stream.get("side_data_list", []):
        if side_data.get("side_data_type") == "Display Matrix" and side_data.get("rotation") is not None:
            rotation_value = side_data["rotation"]
            break
    if rotation_value is None:
        rotation_value = video_stream.get("tags", {}).get("rotate", 0)
    try:
        degrees = int(round(float(rotation_value))) % 360
    except (TypeError, ValueError) as exc:
        raise VideoIOError(f"Invalid video rotation metadata: {rotation_value!r}") from exc
    if degrees not in (0, 90, 180, 270):
        raise VideoIOError(
            f"Unsupported display rotation {rotation_value!r}; expected a multiple of 90 degrees."
        )
    return degrees


def _apply_rotation(image_bgr: np.ndarray, rotation: int) -> np.ndarray:
    # FFmpeg's display-matrix angle describes the counter-clockwise rotation
    # needed for display; OpenCV's matching operations are explicit here.
    if rotation == 90:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image_bgr, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    return image_bgr


def _fraction_or_none(value: object | None, time_base: Fraction) -> Fraction | None:
    if value is None:
        return None
    return Fraction(int(value)) * time_base


def _probe_json(path: Path) -> dict:
    ffprobe = _require_executable("ffprobe")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise VideoIOError(f"ffprobe failed for '{path}': {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoIOError(f"ffprobe returned invalid JSON for '{path}'.") from exc


def probe_video(video_path: str | Path) -> VideoInfo:
    """Read stream metadata. Width/height and decoded frames are rotation-corrected."""

    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise VideoIOError(f"Video file does not exist or is not a file: '{path}'")

    probe = _probe_json(path)
    probe_streams = probe.get("streams", [])
    probe_video_streams = [stream for stream in probe_streams if stream.get("codec_type") == "video"]
    if not probe_video_streams:
        raise VideoIOError(f"No video stream found in '{path}'.")
    selected_probe_stream = probe_video_streams[0]
    stream_index = int(selected_probe_stream.get("index", 0))
    rotation = _extract_rotation(selected_probe_stream)

    try:
        with av.open(str(path), mode="r") as container:
            video_streams = [stream for stream in container.streams if stream.type == "video"]
            video_stream = next((stream for stream in video_streams if stream.index == stream_index), None)
            if video_stream is None and video_streams:
                video_stream = video_streams[0]
            if video_stream is None:
                raise VideoIOError(f"No decodable video stream found in '{path}'.")
            if video_stream.time_base is None:
                raise VideoIOError(f"Video stream in '{path}' has no time base.")

            video_time_base = Fraction(video_stream.time_base)
            raw_width = int(video_stream.codec_context.width)
            raw_height = int(video_stream.codec_context.height)
            width, height = (raw_height, raw_width) if rotation in (90, 270) else (raw_width, raw_height)
            average_rate = video_stream.average_rate
            reference_fps = Fraction(average_rate) if average_rate is not None and average_rate > 0 else None
            start_time = _fraction_or_none(video_stream.start_time, video_time_base)
            duration = _fraction_or_none(video_stream.duration, video_time_base)
            if duration is None and container.duration is not None:
                duration = Fraction(int(container.duration), 1_000_000)

            audio_streams: list[AudioStreamInfo] = []
            for audio_stream in (stream for stream in container.streams if stream.type == "audio"):
                audio_time_base = audio_stream.time_base
                if audio_time_base is None:
                    sample_rate = int(audio_stream.codec_context.sample_rate or 0)
                    if sample_rate <= 0:
                        continue
                    audio_time_base = Fraction(1, sample_rate)
                audio_time_base = Fraction(audio_time_base)
                audio_streams.append(
                    AudioStreamInfo(
                        index=int(audio_stream.index),
                        codec=audio_stream.codec_context.name,
                        time_base=audio_time_base,
                        start_time=_fraction_or_none(audio_stream.start_time, audio_time_base),
                        duration=_fraction_or_none(audio_stream.duration, audio_time_base),
                    )
                )

            frame_count = int(video_stream.frames) if video_stream.frames else None
            return VideoInfo(
                width=width,
                height=height,
                codec=video_stream.codec_context.name,
                reference_fps=reference_fps,
                video_time_base=video_time_base,
                audio_streams=tuple(audio_streams),
                start_time=start_time,
                duration=duration,
                path=path,
                rotation=rotation,
                frame_count=frame_count,
            )
    except VideoIOError:
        raise
    except Exception as exc:
        raise VideoIOError(f"Could not open video '{path}': {exc}") from exc


def _iter_frames(video_path: Path, info: VideoInfo) -> Iterator[VideoFrame]:
    try:
        with av.open(str(video_path), mode="r") as container:
            video_streams = [stream for stream in container.streams if stream.type == "video"]
            if not video_streams:
                raise VideoIOError(f"No decodable video stream found in '{video_path}'.")
            stream = video_streams[0]
            previous_time: Fraction | None = None
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None:
                    raise VideoIOError(
                        f"Decoded frame {index} in '{video_path}' has no PTS. "
                        "This reader requires source PTS and does not synthesize timestamps."
                    )
                time_base = Fraction(frame.time_base or stream.time_base)
                source_pts = int(frame.pts)
                time_sec = Fraction(source_pts) * time_base
                if previous_time is not None and time_sec <= previous_time:
                    raise VideoIOError(
                        f"Video PTS is not strictly increasing at decoded frame {index}: "
                        f"{time_sec} follows {previous_time}."
                    )
                previous_time = time_sec

                image_bgr = _apply_rotation(frame.to_ndarray(format="bgr24"), info.rotation)
                if image_bgr.shape[1] != info.width or image_bgr.shape[0] != info.height:
                    raise VideoIOError(
                        f"Decoded frame {index} is {image_bgr.shape[1]}x{image_bgr.shape[0]}, "
                        f"but probed display dimensions are {info.width}x{info.height}."
                    )
                yield VideoFrame(
                    index=index,
                    source_pts=source_pts,
                    time_base=time_base,
                    time_sec=time_sec,
                    image_bgr=image_bgr,
                )
    except VideoIOError:
        raise
    except Exception as exc:
        raise VideoIOError(f"Could not decode frames from '{video_path}': {exc}") from exc


def iter_frames(video_path: str | Path) -> Iterator[VideoFrame]:
    """Decode frames sequentially in presentation order without inventing PTS."""

    path = Path(video_path).expanduser().resolve()
    info = probe_video(path)
    yield from _iter_frames(path, info)


def _as_fraction(value: int | float | str | Fraction, name: str) -> Fraction:
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise VideoIOError(f"{name} must be a finite number of seconds.") from exc
    if not math.isfinite(float(result)):
        raise VideoIOError(f"{name} must be a finite number of seconds.")
    return result


def _format_seconds(value: Fraction) -> str:
    return f"{float(value):.12f}"


def _validate_boxes(boxes_xyxy: Sequence[BBoxXYXY], info: VideoInfo) -> tuple[BBoxXYXY, ...]:
    try:
        boxes = tuple(tuple(float(value) for value in box) for box in boxes_xyxy)
    except (TypeError, ValueError) as exc:
        raise VideoIOError("Each box must contain four finite numeric coordinates.") from exc
    for box in boxes:
        if len(box) != 4:
            raise VideoIOError("Each box must contain x1,y1,x2,y2.")
        try:
            LineDetection(box, score=1.0).validate_frame_bounds(info.width, info.height)
        except (TypeError, ValueError) as exc:
            raise VideoIOError(f"Invalid box {box} for frame {info.width}x{info.height}: {exc}") from exc
    return boxes


def _encode_video_stage(
    input_path: Path,
    stage_path: Path,
    info: VideoInfo,
    requested_start: Fraction,
    requested_end: Fraction | None,
    boxes: tuple[BBoxXYXY, ...] | None,
    frame_boxes: Sequence[FrameBoxes] | None,
    box_color_bgr: tuple[int, int, int],
    box_thickness: int,
) -> tuple[int, int, Fraction]:
    output_container = None
    output_stream = None
    frame_count = 0
    first_source_pts: int | None = None
    first_source_time: Fraction | None = None
    encoder_time_base = info.video_time_base
    reference_fps = info.reference_fps or Fraction(30, 1)

    try:
        for source_frame in _iter_frames(input_path, info):
            if source_frame.time_sec < requested_start:
                continue
            if requested_end is not None and source_frame.time_sec >= requested_end:
                break
            if frame_boxes is not None:
                if frame_count >= len(frame_boxes):
                    raise VideoIOError(
                        f"Box timeline ended before source frame {source_frame.index} "
                        f"(PTS {source_frame.source_pts})."
                    )
                expected = frame_boxes[frame_count]
                if (expected.frame_index != source_frame.index
                        or expected.source_pts != source_frame.source_pts
                        or expected.time_sec != source_frame.time_sec):
                    raise VideoIOError(
                        f"Box timeline mismatch at entry {frame_count}: expected frame_index="
                        f"{source_frame.index}, PTS={source_frame.source_pts}, "
                        f"time={source_frame.time_sec}; got frame_index={expected.frame_index}, "
                        f"PTS={expected.source_pts}, time={expected.time_sec}."
                    )
                current_boxes = _validate_boxes(expected.boxes_xyxy, info)
            else:
                assert boxes is not None
                current_boxes = boxes
            if output_container is None:
                first_source_pts = source_frame.source_pts
                first_source_time = source_frame.time_sec
                try:
                    output_container = av.open(str(stage_path), mode="w", format="mp4")
                    output_stream = output_container.add_stream("libx264", rate=reference_fps)
                    output_stream.width = info.width
                    output_stream.height = info.height
                    output_stream.pix_fmt = "yuv420p"
                    output_stream.time_base = encoder_time_base
                    output_stream.codec_context.time_base = encoder_time_base
                    output_stream.options = {"preset": "veryfast", "crf": "20"}
                except Exception as exc:
                    raise VideoIOError(f"Could not initialize the H.264 MP4 encoder: {exc}") from exc

            assert output_container is not None and output_stream is not None
            image_bgr = source_frame.image_bgr.copy()
            for x1, y1, x2, y2 in current_boxes:
                cv2.rectangle(
                    image_bgr,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)) - 1, int(round(y2)) - 1),
                    box_color_bgr,
                    thickness=box_thickness,
                    lineType=cv2.LINE_AA,
                )

            output_frame = av.VideoFrame.from_ndarray(image_bgr, format="bgr24")
            relative_time = source_frame.time_sec - first_source_time
            relative_pts = relative_time / encoder_time_base
            output_frame.pts = int(relative_pts + Fraction(1, 2))
            output_frame.time_base = encoder_time_base
            try:
                for packet in output_stream.encode(output_frame):
                    output_container.mux(packet)
            except Exception as exc:
                raise VideoIOError(f"H.264 encoding failed at source frame {source_frame.index}: {exc}") from exc
            frame_count += 1

        if output_container is None or output_stream is None or first_source_pts is None or first_source_time is None:
            raise VideoIOError("The requested segment contains no video frames.")
        if frame_boxes is not None and frame_count != len(frame_boxes):
            raise VideoIOError(
                f"Box timeline has {len(frame_boxes)} entries but the selected video has "
                f"{frame_count} frames."
            )
        try:
            for packet in output_stream.encode(None):
                output_container.mux(packet)
        except Exception as exc:
            raise VideoIOError(f"Could not flush the H.264 encoder: {exc}") from exc
        output_container.close()
        output_container = None
        return frame_count, first_source_pts, first_source_time
    finally:
        if output_container is not None:
            output_container.close()


def _run_mux(command: list[str], description: str) -> None:
    result = subprocess.run(
        command, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise VideoIOError(f"{description}: {result.stderr.strip()}")


def _remux_audio(
    ffmpeg: str,
    stage_path: Path,
    input_path: Path,
    mux_stage_path: Path,
    first_source_time: Fraction,
) -> None:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(stage_path)]
    if first_source_time:
        command.extend(["-itsoffset", _format_seconds(-first_source_time)])
    command.extend([
        "-i", str(input_path), "-map", "0:v:0", "-map", "1:a", "-c:v", "copy",
        "-c:a", "copy", "-movflags", "+faststart", str(mux_stage_path),
    ])
    _run_mux(command, "Could not remux source audio")


def _mux_audio(
    ffmpeg: str,
    stage_path: Path,
    input_path: Path,
    mux_stage_path: Path,
    audio_count: int,
    first_source_time: Fraction,
    requested_end: Fraction | None,
) -> None:
    filters: list[str] = []
    output_labels: list[str] = []
    start_text = _format_seconds(first_source_time)
    end_text = f":end={_format_seconds(requested_end)}" if requested_end is not None else ""
    for index in range(audio_count):
        label = f"a{index}"
        filters.append(
            f"[1:a:{index}]atrim=start={start_text}{end_text},"
            f"asetpts=PTS-({start_text})/TB[{label}]"
        )
        output_labels.append(label)

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(stage_path), "-i", str(input_path)]
    command.extend(["-filter_complex", ";".join(filters), "-map", "0:v:0"])
    for label in output_labels:
        command.extend(["-map", f"[{label}]"])
    command.extend(
        [
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "160k",
            "-movflags", "+faststart",
            str(mux_stage_path),
        ]
    )
    _run_mux(command, "Could not encode source audio as AAC")


def export_annotated_segment(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start_sec: int | float | str | Fraction = 0,
    duration_sec: int | float | str | Fraction | None = None,
    boxes_xyxy: Sequence[BBoxXYXY] | None = None,
    frame_boxes: Sequence[FrameBoxes] | None = None,
    box_color_bgr: tuple[int, int, int] = (0, 0, 255),
    box_thickness: int = 3,
) -> SegmentExport:
    """Export a segment or full video with fixed boxes or a checked per-frame timeline.

    Supply exactly one of boxes_xyxy and frame_boxes. A timeline has one entry for
    every selected frame, including frames with no boxes, and carries original PTS.
    Omitting duration_sec with start_sec=0 selects the whole video.
    """

    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve(strict=False)
    if not source.is_file():
        raise VideoIOError(f"Video file does not exist or is not a file: '{source}'")
    if source == output:
        raise VideoIOError("Output path must not overwrite the input video.")
    if output.suffix.lower() != ".mp4":
        raise VideoIOError("The output path must end in .mp4.")

    start_offset = _as_fraction(start_sec, "start_sec")
    duration = _as_fraction(duration_sec, "duration_sec") if duration_sec is not None else None
    if start_offset < 0 or (duration is not None and duration <= 0):
        raise VideoIOError("start_sec must be nonnegative and duration_sec must be positive.")
    if duration is None and start_offset != 0:
        raise VideoIOError("duration_sec is required when start_sec is nonzero.")
    if (boxes_xyxy is None) == (frame_boxes is None):
        raise VideoIOError("Provide exactly one of boxes_xyxy or frame_boxes.")
    if box_thickness <= 0:
        raise VideoIOError("box_thickness must be positive.")
    if len(box_color_bgr) != 3 or any(value < 0 or value > 255 for value in box_color_bgr):
        raise VideoIOError("box_color_bgr must contain three values in the range 0..255.")

    info = probe_video(source)
    boxes = _validate_boxes(boxes_xyxy, info) if boxes_xyxy is not None else None
    source_start = info.start_time if info.start_time is not None else Fraction(0)
    requested_start = source_start + start_offset
    requested_end = requested_start + duration if duration is not None else None
    if (requested_end is not None and info.duration is not None
            and requested_end > source_start + info.duration):
        raise VideoIOError(
            f"Requested segment ends at {float(requested_end - source_start):.3f}s, "
            f"past the video duration {float(info.duration):.3f}s."
        )

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VideoIOError(f"Could not create output directory '{output.parent}': {exc}") from exc
    stage_id = uuid.uuid4().hex
    stage_path = output.with_name(f".{output.stem}.{stage_id}.stage.mp4")
    mux_stage_path = output.with_name(f".{output.stem}.{stage_id}.mux.mp4")
    audio_mode = "none (source has no audio)"
    try:
        frame_count, first_source_pts, first_source_time = _encode_video_stage(
            source,
            stage_path,
            info,
            requested_start,
            requested_end,
            boxes,
            frame_boxes,
            box_color_bgr,
            box_thickness,
        )
        if not info.audio_streams:
            stage_path.replace(output)
        else:
            ffmpeg = _require_executable("ffmpeg")
            if duration is None and all(stream.codec == "aac" for stream in info.audio_streams):
                try:
                    _remux_audio(ffmpeg, stage_path, source, mux_stage_path, first_source_time)
                    audio_mode = "AAC remux (stream copy)"
                except VideoIOError:
                    _mux_audio(ffmpeg, stage_path, source, mux_stage_path,
                               len(info.audio_streams), first_source_time, None)
                    audio_mode = "AAC re-encode (remux unavailable)"
            else:
                _mux_audio(ffmpeg, stage_path, source, mux_stage_path,
                           len(info.audio_streams), first_source_time, requested_end)
                audio_mode = ("AAC re-encode (segment-accurate trim)" if duration is not None
                              else "AAC re-encode (remux unavailable)")
            mux_stage_path.replace(output)
    finally:
        for temporary_path in (stage_path, mux_stage_path):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                raise VideoIOError(f"Could not remove intermediate video file '{temporary_path}': {exc}") from exc

    return SegmentExport(
        output_path=output,
        frame_count=frame_count,
        first_source_pts=first_source_pts,
        first_source_time=first_source_time,
        end_source_time=(requested_end if requested_end is not None else
                         source_start + info.duration if info.duration is not None else
                         first_source_time + Fraction(frame_count, info.reference_fps or 30)),
        audio_mode=audio_mode,
    )


def _parse_box(value: str) -> BBoxXYXY:
    try:
        coordinates = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2 integers") from exc
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2 integers")
    return coordinates  # type: ignore[return-value]


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a fixed-box video/audio smoke clip (milestone 2).")
    parser.add_argument("--input", type=Path, default=Path("data/AI Engineer test.mp4"))
    parser.add_argument("--output", type=Path, default=Path("outputs/m2_smoke_clip.mp4"))
    parser.add_argument("--start-sec", type=float, default=12.0, help="segment start relative to video start")
    parser.add_argument("--duration-sec", type=float, default=8.0)
    parser.add_argument("--box", type=_parse_box, default=(70, 836, 650, 912), help="fixed full-frame box x1,y1,x2,y2")
    args = parser.parse_args(argv)
    try:
        result = export_annotated_segment(
            args.input,
            args.output,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            boxes_xyxy=[args.box],
        )
    except VideoIOError as exc:
        print(f"video_io: error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {result.output_path} | frames={result.frame_count} "
        f"| source_pts_start={result.first_source_pts} "
        f"| source_start={float(result.first_source_time):.6f}s "
        f"| duration={float(result.duration):.6f}s | audio={result.audio_mode}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
