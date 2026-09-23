"""Milestone-4 audit: raw lines, tracked lines, draw timeline, and short MP4s."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from src.detect import DEFAULT_MODEL_NAME, DetectionError, Detector
from src.events import EventConfig, TrackedFrame, line_signature, observe_frame, track_observations
from src.schemas import FrameObservation, LineDetection
from src.video_io import VideoIOError, export_annotated_segment, iter_frames


@dataclass(frozen=True, slots=True)
class DebugWindow:
    name: str
    start: Fraction
    end: Fraction


WINDOWS = (
    DebugWindow("short_tian_2p5_3p5", Fraction(5, 2), Fraction(7, 2)),
    DebugWindow("change_19p5_22p5", Fraction(39, 2), Fraction(45, 2)),
    DebugWindow("hand_23p5_24p5", Fraction(47, 2), Fraction(49, 2)),
    DebugWindow("figure_122p5_124p3", Fraction(245, 2), Fraction(1243, 10)),
)
ANCHORS = {"short_tian_2p5_3p5": Fraction(3),
           "change_19p5_22p5": Fraction(21),
           "hand_23p5_24p5": Fraction(24),
           "figure_122p5_124p3": Fraction(123)}


def _panel(image: np.ndarray, boxes: tuple, color: tuple[int, int, int], title: str) -> np.ndarray:
    canvas = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (round(x1), round(y1)), (round(x2) - 1, round(y2) - 1),
                      color, 3, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(canvas, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(canvas, (480, 853), interpolation=cv2.INTER_AREA)


def _comparison_image(image: np.ndarray, frame: TrackedFrame, path: Path) -> None:
    panels = [
        _panel(image, frame.raw_boxes, (0, 255, 0), "RAW"),
        _panel(image, frame.tracked_boxes, (0, 255, 255), "TRACKED"),
        _panel(image, frame.draw_boxes, (0, 0, 255), "DRAW"),
    ]
    if not cv2.imwrite(str(path), np.hstack(panels)):
        raise OSError(f"Could not write comparison image {path}")


def run_debug(
    input_path: str | Path = "data/AI Engineer test.mp4",
    output_dir: str | Path = "outputs/m4_debug",
    *,
    model_dir: str | Path = "models/PP-OCRv6_medium_det",
    config: EventConfig | None = None,
    reuse_raw: bool = False,
) -> Path:
    settings = config or EventConfig()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(input_path).expanduser().resolve()
    observations: dict[str, list[FrameObservation]] = {window.name: [] for window in WINDOWS}
    anchor_images: dict[str, np.ndarray] = {}
    cached: dict[int, dict] = {}
    if reuse_raw:
        cache_path = output / "comparison.json"
        if not cache_path.is_file():
            raise VideoIOError(f"Raw comparison cache missing: {cache_path}")
        prior = json.loads(cache_path.read_text(encoding="utf-8"))
        if Path(prior["source"]).resolve() != source:
            raise VideoIOError("Raw comparison cache belongs to a different source video")
        for window in prior["windows"]:
            for row in window["frames"]:
                cached[row["frame_index"]] = row
    detector = None if reuse_raw else Detector(
        device="cpu", model_name=DEFAULT_MODEL_NAME, model_dir=model_dir,
    )
    reader = iter_frames(source)
    frame_size: tuple[int, int] | None = None
    analyzed = 0
    try:
        for frame in reader:
            if frame.time_sec >= max(window.end for window in WINDOWS):
                break
            for window in WINDOWS:
                if window.start <= frame.time_sec < window.end:
                    if reuse_raw:
                        row = cached.get(frame.index)
                        if row is None or row["source_pts"] != frame.source_pts:
                            raise VideoIOError(
                                f"Raw comparison timeline mismatch at frame {frame.index}, "
                                f"PTS {frame.source_pts}"
                            )
                        lines = tuple(LineDetection(tuple(box), 1.0)
                                      for box in row["raw_boxes"])
                        signatures = [line_signature(frame.image_bgr, line.bbox_xyxy)
                                      for line in lines]
                        observation = FrameObservation(
                            frame_index=frame.index, time_sec=frame.time_sec,
                            lines=lines, source_pts=frame.source_pts,
                            fingerprints=tuple(value[1] for value in signatures),
                            style_scores=tuple(value[0] for value in signatures),
                        )
                    else:
                        observation = observe_frame(frame, detector)
                    observations[window.name].append(observation)
                    frame_size = (frame.image_bgr.shape[1], frame.image_bgr.shape[0])
                    analyzed += 1
                    if frame.time_sec == ANCHORS[window.name]:
                        anchor_images[window.name] = frame.image_bgr.copy()
                    if analyzed % 20 == 0:
                        print(f"Analyzed {analyzed} source frames (last PTS={frame.source_pts})", flush=True)
    finally:
        reader.close()
        if detector is not None:
            detector.close()
    if frame_size is None:
        raise VideoIOError("No frames found in the requested debug windows")

    report: dict = {"source": str(source), "model": DEFAULT_MODEL_NAME,
                    "device": "cpu", "enable_mkldnn": False,
                    "roi_bottom": 0.45, "reused_raw_boxes": reuse_raw,
                    "event_config": asdict(settings), "windows": []}
    for window in WINDOWS:
        observed = observations[window.name]
        if not observed:
            raise VideoIOError(f"No source frames in debug window {window.name}")
        result = track_observations(observed, frame_size, settings)
        if window.name in anchor_images:
            anchor = next((frame for frame in result.frames if frame.time_sec == ANCHORS[window.name]), None)
            if anchor is not None:
                _comparison_image(anchor_images[window.name], anchor,
                                  output / f"compare_{window.name}.png")
        clip_path = output / f"{window.name}.mp4"
        exported = export_annotated_segment(
            source, clip_path, start_sec=window.start,
            duration_sec=window.end - window.start,
            frame_boxes=result.frame_boxes(),
        )
        report["windows"].append({
            "name": window.name,
            "start_sec": float(window.start),
            "end_sec": float(window.end),
            "clip": clip_path.name,
            "exported_frames": exported.frame_count,
            "audio_mode": exported.audio_mode,
            "events": [{**asdict(event), "start_sec": float(event.start_sec),
                        "end_sec": float(event.end_sec)} for event in result.events],
            "frames": [{
                "frame_index": frame.frame_index,
                "source_pts": frame.source_pts,
                "timestamp_sec": float(frame.time_sec),
                "raw_boxes": frame.raw_boxes,
                "style_scores": observed[index].style_scores,
                "tracked_boxes": frame.tracked_boxes,
                "draw_boxes": frame.draw_boxes,
                "event_ids": frame.event_ids,
                "rejected": frame.rejected,
            } for index, frame in enumerate(result.frames)],
        })
        print(f"{window.name}: {len(observed)} frames, {len(result.events)} line events, "
              f"clip={clip_path}", flush=True)
    path = output / "comparison.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit milestone-4 events on source-video windows.")
    parser.add_argument("--input", type=Path, default=Path("data/AI Engineer test.mp4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m4_debug"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/PP-OCRv6_medium_det"))
    parser.add_argument("--reuse-raw", action="store_true",
                        help="Re-track previously saved raw boxes after validating source PTS.")
    args = parser.parse_args(argv)
    try:
        result = run_debug(args.input, args.output_dir, model_dir=args.model_dir,
                           reuse_raw=args.reuse_raw)
    except (DetectionError, VideoIOError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
