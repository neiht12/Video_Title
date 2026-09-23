"""CLI entry point for the video-title OCR pipeline."""

from __future__ import annotations

import argparse
import sys

from src.config import ConfigError, build_config, make_argument_parser, validate_device_available


def main(argv: list[str] | None = None) -> int:
    parser = make_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = build_config(args)
        validate_device_available(config.device)
    except ConfigError as exc:
        parser.error(str(exc))

    print(
        "Configuration valid. "
        f"input={config.input_path}; output={config.output_path}; json={config.json_path}; "
        f"roi-bottom={config.roi_bottom:.2f}; device={config.device}. "
        "Video processing will be added in milestone 2."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
