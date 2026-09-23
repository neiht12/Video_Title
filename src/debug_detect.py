"""Export milestone-3 detection overlays from selected source-video timestamps."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from src.config import DEFAULT_ROI_BOTTOM, ConfigError, parse_device, parse_roi_bottom, roi_y0
from src.detect import DEFAULT_MODEL_NAME, DetectionError, Detector
from src.lines import merge_lines
from src.schemas import LineDetection, TextRegion, VideoFrame
from src.video_io import VideoIOError, iter_frames


DEFAULT_SAMPLES = {
    "no_subtitle": Fraction(0),
    "short_tian": Fraction(3),
    "one_line": Fraction(15),
    "two_lines": Fraction(27),
}


def _draw(
    frame_bgr: np.ndarray,
    y0: int,
    regions: list[TextRegion],
    lines: list[LineDetection],
    label: str,
    frame: VideoFrame,
) -> tuple[np.ndarray, np.ndarray]:
    full = frame_bgr.copy()
    roi = frame_bgr[y0:].copy()
    cv2.line(full, (0, y0), (full.shape[1] - 1, y0), (0, 255, 255), 2)
    for region in regions:
        points = np.rint(np.asarray(region.polygon_xy)).astype(np.int32)
        cv2.polylines(full, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
        roi_points = points - np.array([0, y0], dtype=np.int32)
        cv2.polylines(roi, [roi_points], True, (0, 255, 0), 2, cv2.LINE_AA)
    for line in lines:
        x1, top, x2, bottom = map(int, line.bbox_xyxy)
        cv2.rectangle(full, (x1, top), (x2 - 1, bottom - 1), (0, 0, 255), 2)
        cv2.rectangle(roi, (x1, top - y0), (x2 - 1, bottom - y0 - 1), (0, 0, 255), 2)
    caption = f"{label} | frame={frame.index} PTS={frame.source_pts} t={float(frame.time_sec):.3f}s"
    cv2.putText(full, caption, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(full, caption, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(roi, caption, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(roi, caption, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return full, roi


def export_debug_frames(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    samples: dict[str, Fraction] | None = None,
    roi_bottom: float = DEFAULT_ROI_BOTTOM,
    device: str = "cpu",
    model_name: str = DEFAULT_MODEL_NAME,
    model_dir: str | Path | None = None,
) -> Path:
    """Save full and ROI overlays plus a timestamp/box manifest for four samples."""

    selected = samples or DEFAULT_SAMPLES
    if not selected or any(time < 0 for time in selected.values()):
        raise ValueError("samples must have nonnegative source timestamps")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    roi_bottom = parse_roi_bottom(roi_bottom)
    detector = Detector(device=device, model_name=model_name, model_dir=model_dir)
    pending = dict(selected)
    frames: dict[str, VideoFrame] = {}
    try:
        reader = iter_frames(input_path)
        try:
            for frame in reader:
                for label, target in list(pending.items()):
                    if frame.time_sec >= target:
                        frames[label] = frame
                        del pending[label]
                if not pending:
                    break
        finally:
            reader.close()
    except Exception:
        detector.close()
        raise
    if pending:
        detector.close()
        raise VideoIOError(f"No source frame at or after: {pending}")

    manifest: dict[str, object] = {
        "input": str(Path(input_path).resolve()),
        "model": model_name,
        "model_dir": str(model_dir) if model_dir is not None else None,
        "device": device,
        "enable_mkldnn": False,
        "roi_bottom": roi_bottom,
        "samples": [],
    }
    try:
        for label, frame in frames.items():
            height, width = frame.image_bgr.shape[:2]
            y0 = roi_y0(height, roi_bottom)
            regions = detector.detect(frame.image_bgr, roi_bottom)
            lines = merge_lines(regions, (width, height))
            full_image, roi_image = _draw(frame.image_bgr, y0, regions, lines, label, frame)
            stem = f"{label}_f{frame.index:06d}_pts{frame.source_pts}_t{float(frame.time_sec):09.3f}s"
            full_path = output / f"{stem}_full.png"
            roi_path = output / f"{stem}_roi.png"
            if not cv2.imwrite(str(full_path), full_image) or not cv2.imwrite(str(roi_path), roi_image):
                raise OSError(f"Could not write debug images for {label} to {output}")
            manifest["samples"].append({
                "label": label,
                "requested_time_sec": float(selected[label]),
                "frame_index": frame.index,
                "source_pts": frame.source_pts,
                "time_base": str(frame.time_base),
                "timestamp_sec": float(frame.time_sec),
                "roi_y0": y0,
                "raw_region_count": len(regions),
                "raw_regions": [
                    {"polygon_xy": region.polygon_xy, "score": region.score}
                    for region in regions
                ],
                "line_count": len(lines),
                "line_boxes_xyxy": [line.bbox_xyxy for line in lines],
                "line_scores": [line.score for line in lines],
                "full_image": full_path.name,
                "roi_image": roi_path.name,
            })
    finally:
        detector.close()
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _sample(value: str) -> tuple[str, Fraction]:
    try:
        label, seconds = value.split(":", 1)
        timestamp = Fraction(seconds)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError("sample must be LABEL:SECONDS") from exc
    if not label or timestamp < 0:
        raise argparse.ArgumentTypeError("sample must use a name and nonnegative seconds")
    return label, timestamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save raw polygons and merged line boxes (milestone 3).")
    parser.add_argument("--input", type=Path, default=Path("data/AI Engineer test.mp4"))
    parser.add_argument("--output-dir", type=Path, default=Path("debug_frames"))
    parser.add_argument("--roi-bottom", type=parse_roi_bottom, default=DEFAULT_ROI_BOTTOM)
    parser.add_argument("--device", type=parse_device, default=parse_device("cpu"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-dir", type=Path, help="local PaddleOCR inference model directory")
    parser.add_argument("--sample", type=_sample, action="append", help="LABEL:SECONDS; repeat to select frames")
    args = parser.parse_args(argv)
    try:
        samples = dict(args.sample) if args.sample else None
        result = export_debug_frames(
            args.input, args.output_dir, samples=samples, roi_bottom=args.roi_bottom,
            device=str(args.device), model_name=args.model_name, model_dir=args.model_dir,
        )
    except (ConfigError, DetectionError, VideoIOError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
