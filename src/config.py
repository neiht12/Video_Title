"""Command-line configuration and validation for the video OCR pipeline."""

from __future__ import annotations

import argparse
import importlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MIN_ROI_BOTTOM = 0.25
MAX_ROI_BOTTOM = 0.45
DEFAULT_ROI_BOTTOM = 0.45


class ConfigError(ValueError):
    """Raised when a CLI value or the selected execution device is invalid."""


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """A Paddle device selection; bare ``gpu`` refers to GPU index zero."""

    kind: str
    index: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "cpu" and self.index is None:
            return
        if self.kind == "gpu" and isinstance(self.index, int) and not isinstance(self.index, bool) and self.index >= 0:
            return
        raise ConfigError("device must be 'cpu' or a GPU with a nonnegative index")

    @property
    def paddle_name(self) -> str:
        if self.kind == "cpu":
            return "cpu"
        return f"gpu:{self.index}"

    def __str__(self) -> str:
        return self.paddle_name


@dataclass(frozen=True, slots=True)
class AppConfig:
    input_path: Path
    output_path: Path
    json_path: Path
    srt_path: Path | None
    roi_bottom: float = DEFAULT_ROI_BOTTOM
    device: DeviceSpec = DeviceSpec("cpu")
    # This was the verified CPU setting in the milestone 0 inference check.
    enable_mkldnn: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "roi_bottom", parse_roi_bottom(self.roi_bottom))
        if self.enable_mkldnn is not False:
            raise ConfigError("enable_mkldnn must remain False for this pipeline configuration")


def parse_roi_bottom(value: str | float) -> float:
    """Parse an ROI fraction measured upward from the bottom of the frame."""

    try:
        roi_bottom = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("--roi-bottom must be a number between 0.25 and 0.45") from exc
    if not math.isfinite(roi_bottom) or not MIN_ROI_BOTTOM <= roi_bottom <= MAX_ROI_BOTTOM:
        raise ConfigError("--roi-bottom must be between 0.25 and 0.45 (inclusive)")
    return roi_bottom


def roi_y0(height: int, roi_bottom: float = DEFAULT_ROI_BOTTOM) -> int:
    """Return the first ROI row using the plan's floor-based convention."""

    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ConfigError("frame height must be a positive integer")
    roi = parse_roi_bottom(roi_bottom)
    return math.floor(height * (1.0 - roi))


def parse_device(value: str | DeviceSpec) -> DeviceSpec:
    """Parse ``cpu``, ``gpu`` or ``gpu:<nonnegative index>``."""

    if isinstance(value, DeviceSpec):
        return value
    normalized = value.strip().lower()
    if normalized == "cpu":
        return DeviceSpec("cpu")
    if normalized == "gpu":
        return DeviceSpec("gpu", 0)
    if normalized.startswith("gpu:"):
        index_text = normalized[4:]
        if index_text.isdecimal():
            return DeviceSpec("gpu", int(index_text))
    raise ConfigError("--device must be 'cpu', 'gpu', or 'gpu:<nonnegative index>'")


def _canonical_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _resolved_path(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def build_config(args: argparse.Namespace) -> AppConfig:
    """Validate parsed arguments without creating outputs or opening video."""

    input_path = _resolved_path(args.input)
    if not input_path.exists():
        raise ConfigError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise ConfigError(f"Input path is not a file: {input_path}")

    output_path = _resolved_path(args.output)
    json_path = _resolved_path(args.json_output)
    srt_path = _resolved_path(args.srt) if args.srt is not None else None

    input_key = _canonical_path(input_path)
    output_paths = [("--output", output_path), ("--json", json_path)]
    if srt_path is not None:
        output_paths.append(("--srt", srt_path))
    for option, output in output_paths:
        if _canonical_path(output) == input_key:
            raise ConfigError(f"{option} must not refer to the input file: {output}")
    output_keys = [_canonical_path(path) for _, path in output_paths]
    if len(output_keys) != len(set(output_keys)):
        raise ConfigError("--output, --json, and --srt must use different paths")

    try:
        roi_bottom = parse_roi_bottom(args.roi_bottom)
        device = parse_device(args.device)
    except ConfigError:
        raise

    return AppConfig(
        input_path=input_path,
        output_path=output_path,
        json_path=json_path,
        srt_path=srt_path,
        roi_bottom=roi_bottom,
        device=device,
        enable_mkldnn=False,
    )


def validate_device_available(device: DeviceSpec | str) -> None:
    """Fail fast for an unavailable GPU; CPU selection never imports Paddle."""

    selected = parse_device(device)
    if selected.kind == "cpu":
        return

    try:
        paddle: Any = importlib.import_module("paddle")
    except Exception as exc:
        raise ConfigError(
            f"GPU {selected.index} was selected, but PaddlePaddle could not be loaded; "
            "install a CUDA-enabled PaddlePaddle build. No CPU fallback was used. "
            f"Details: {exc}"
        ) from exc

    try:
        if not paddle.is_compiled_with_cuda():
            raise ConfigError(
                f"GPU {selected.index} was selected, but this PaddlePaddle build has no CUDA support. "
                "Install a compatible CUDA-enabled build. No CPU fallback was used."
            )
        gpu_count = int(paddle.device.cuda.device_count())
        if selected.index is None or selected.index >= gpu_count:
            raise ConfigError(
                f"GPU {selected.index} was selected, but only {gpu_count} CUDA GPU(s) are available. "
                "No CPU fallback was used."
            )
        paddle.device.set_device(selected.paddle_name)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            f"GPU {selected.index} could not be initialized: {exc}. No CPU fallback was used."
        ) from exc


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate video OCR configuration (processing is added in milestone 2)."
    )
    parser.add_argument("--input", required=True, type=Path, help="source video file")
    parser.add_argument("--output", required=True, type=Path, help="annotated video output path")
    parser.add_argument("--json", dest="json_output", required=True, type=Path, help="JSON output path")
    parser.add_argument("--srt", type=Path, help="optional subtitle output path")
    parser.add_argument(
        "--roi-bottom",
        type=parse_roi_bottom,
        default=DEFAULT_ROI_BOTTOM,
        metavar="0.25..0.45",
        help="fraction of frame height included from the bottom (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        type=parse_device,
        default=DeviceSpec("cpu"),
        metavar="cpu|gpu[:index]",
        help="inference device; gpu means gpu:0 and never falls back to CPU (default: cpu)",
    )
    return parser
