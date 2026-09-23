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

The current CLI validates paths, ROI, and device selection; video reading and output writing are added in milestone 2. `--roi-bottom` accepts 0.25 through 0.45, inclusive. `--device` accepts `cpu`, `gpu`, or `gpu:<index>`; an unavailable GPU is an error and does not fall back to CPU.
