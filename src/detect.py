"""PaddleOCR text detection on the bottom ROI, returned in full-frame coordinates."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DEFAULT_ROI_BOTTOM, DeviceSpec, parse_device, roi_y0, validate_device_available
from src.schemas import TextRegion


DEFAULT_MODEL_NAME = "PP-OCRv6_medium_det"


class DetectionError(RuntimeError):
    """The detector could not initialize or returned an unusable result."""


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    pixel_threshold: float = 0.3
    box_threshold: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (("pixel_threshold", self.pixel_threshold),
                            ("box_threshold", self.box_threshold)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")


class Detector:
    """Initialize TextDetection once and predict one BGR ROI at a time."""

    def __init__(
        self,
        device: DeviceSpec | str = "cpu",
        model_name: str = DEFAULT_MODEL_NAME,
        thresholds: DetectionConfig | None = None,
        *,
        predictor: Any | None = None,
        model_dir: str | Path | None = None,
    ) -> None:
        self.device = parse_device(device)
        self.model_name = model_name
        self.thresholds = thresholds or DetectionConfig()
        if predictor is not None:
            self._predictor = predictor
            return

        validate_device_available(self.device)
        cache_dir = Path(__file__).resolve().parents[1] / "models" / "paddlex_cache"
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
        try:
            from paddleocr import TextDetection

            kwargs: dict[str, Any] = {
                "model_name": model_name,
                "device": self.device.paddle_name,
                "enable_mkldnn": False,
                "thresh": self.thresholds.pixel_threshold,
                "box_thresh": self.thresholds.box_threshold,
            }
            if model_dir is not None:
                kwargs["model_dir"] = str(Path(model_dir).expanduser().resolve())
            self._predictor = TextDetection(**kwargs)
        except Exception as exc:
            raise DetectionError(
                f"Could not initialize TextDetection {model_name} on {self.device} "
                f"(enable_mkldnn=False): {exc}"
            ) from exc

    def detect(
        self, frame_bgr: np.ndarray, roi_bottom: float = DEFAULT_ROI_BOTTOM,
    ) -> list[TextRegion]:
        """Return valid polygons with the ROI y-offset applied exactly once."""

        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3
                or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8):
            raise DetectionError("frame_bgr must be a uint8 BGR image with three channels")
        height, width = frame_bgr.shape[:2]
        y0 = roi_y0(height, roi_bottom)
        roi = np.ascontiguousarray(frame_bgr[y0:height, 0:width])
        try:
            results = self._predictor.predict(input=roi)
        except Exception as exc:
            raise DetectionError(f"TextDetection inference failed: {exc}") from exc
        if len(results) != 1:
            raise DetectionError(f"TextDetection returned {len(results)} results for one ROI")
        result = results[0]
        try:
            polygons = result["dt_polys"]
            scores = result["dt_scores"]
        except (KeyError, TypeError) as exc:
            raise DetectionError("TextDetection result lacks dt_polys or dt_scores") from exc
        if len(polygons) != len(scores):
            raise DetectionError("TextDetection dt_polys and dt_scores have different lengths")

        regions: list[TextRegion] = []
        for polygon, raw_score in zip(polygons, scores):
            try:
                points = np.asarray(polygon, dtype=np.float64)
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                    or not np.isfinite(points).all() or not math.isfinite(score)
                    or not 0 <= score <= 1):
                continue
            x = np.clip(points[:, 0], 0, width)
            y = np.clip(points[:, 1], 0, height - y0) + y0
            if x.max() <= x.min() or y.max() <= y.min():
                continue
            region = TextRegion(tuple((float(px), float(py)) for px, py in zip(x, y)), score)
            region.validate_frame_bounds(width, height)
            regions.append(region)
        return regions

    def close(self) -> None:
        close = getattr(self._predictor, "close", None)
        if close is not None:
            close()
