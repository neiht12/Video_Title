# Video_title

Python project for video-title OCR.

## Prerequisites

- Python 3.12 (64-bit).
- FFmpeg command-line tools: both `ffmpeg` and `ffprobe` must be on `PATH`. On Windows, install them with:

```powershell
winget install --id Gyan.FFmpeg.Shared --exact --scope user
```

Restart the terminal after installation so it picks up the updated `PATH`.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import paddle; paddle.utils.run_check()"
python main.py --help
```

The pinned PaddlePaddle package is the CPU build. Install a GPU build only on a Windows/CUDA configuration supported by the [official PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/documentation/docs/install/pip/windows-pip_en.html).

For the pinned CPU stack, `TextDetection(device="cpu", enable_mkldnn=False)` is the verified inference configuration. The default MKL-DNN setting raised a PIR/oneDNN `NotImplementedError` during the milestone 0 frame check.

## CLI skeleton (milestone 1)

```powershell
python main.py --input "data\AI Engineer test.mp4" --output outputs\boxed.mp4 --json outputs\boxes.json --device cpu --roi-bottom 0.45
python -m unittest discover -s tests -v
```

The `main.py` CLI validates paths, ROI, and device selection; end-to-end OCR processing remains for later milestones. `--roi-bottom` accepts 0.25 through 0.45, inclusive. `--device` accepts `cpu`, `gpu`, or `gpu:<index>`; an unavailable GPU is an error and does not fall back to CPU.

## Video I/O smoke clip (milestone 2)

```powershell
python -m src.video_io --input "data\AI Engineer test.mp4" --output outputs\m2_smoke_clip.mp4 --start-sec 12 --duration-sec 8 --box "70,836,650,912"
python -m unittest discover -s tests -v
```

The smoke exporter reads frames with their source PTS/time base, applies display rotation once, writes H.264, and trims/re-encodes the selected audio segment as AAC so the segment boundaries remain aligned. FFmpeg and ffprobe must be available on `PATH`; if the current terminal has not picked up the install yet, set `FFMPEG_BIN` to FFmpeg's `bin` directory.
