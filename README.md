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

`export_annotated_segment()` also accepts `start_sec=0, duration_sec=None` to export the whole video. Compatible AAC audio is remuxed (stream copied) for that case; exact segment cuts and remux failures use AAC encoding. A source without audio produces a silent MP4. `SegmentExport.audio_mode` reports the path used. The exporter does not use FFmpeg's `-shortest`, so an audio stream that ends first cannot truncate the video.

For boxes that change by frame, pass `frame_boxes` instead of `boxes_xyxy`. Supply one `FrameBoxes` per selected frame, including empty `boxes_xyxy` for frames without a box. Each entry carries the first read's `frame_index`, `source_pts`, and `time_sec`; the export's second read checks all three before drawing.

## Detection and line boxes (milestone 3)

```powershell
python -m src.debug_detect --input "data\AI Engineer test.mp4" --output-dir debug_frames --roi-bottom 0.45 --device cpu --model-name PP-OCRv6_medium_det --model-dir models\PP-OCRv6_medium_det
python -m unittest discover -s tests -v
```

The debug command reads the source at 0, 3, 15, and 27 seconds, runs PaddleOCR 3.7.0 `TextDetection` on the bottom 45% of each frame, and saves a full-frame image, an ROI image, and `debug_frames/manifest.json` with frame index, source PTS, timestamp, polygons, scores, and line boxes. Green outlines are raw polygons; red outlines are merged line boxes; yellow marks the ROI top. The CPU predictor uses `enable_mkldnn=False`. The default detection pixel/box thresholds are 0.3/0.5; line merging keeps scores at least 0.5 and adds 3 px horizontal and 2 px vertical padding after grouping. This milestone does not track lines between frames or create subtitle events.

The local `models/PP-OCRv6_medium_det` directory is ignored by Git. Omit `--model-dir` when running with network access to let PaddleOCR obtain the official model into `models/paddlex_cache`.

An additional source-video audit at 21, 24, and 123 seconds is in `debug_frames/audit/`. TextDetection also outlines a shirt detail at 21 seconds, a hand holding a stick at 24 seconds, and a foreground figure at 123 seconds. These are detection false positives at the current thresholds; line grouping alone cannot identify subtitle text or use persistence across frames.

## Per-line events (milestone 4)

```powershell
python -m src.debug_events --input "data\AI Engineer test.mp4" --output-dir outputs\m4_debug --model-dir models\PP-OCRv6_medium_det
python -m src.debug_events --input "data\AI Engineer test.mp4" --output-dir outputs\m4_debug --reuse-raw
python -m unittest discover -s tests -v
```

`src.events` matches lines one to one by position, overlap, and height; a white-stroke fingerprint in fixed frame coordinates confirms a changed sentence across two frames. Offline tracking fills only internal gaps of up to two frames within the same line event, then applies a local median with a safety margin around each observed box. Candidate filtering uses the sampled video's subtitle band (top at 62% and bottom at 80% of frame height) and a white-stroke-with-dark-neighbor fraction of at least 0.04; these values are configurable in `EventConfig`. A line needs three actual detections to become an event. The debug command writes short clips showing the boxes selected for drawing, comparison images at 3/21/24/123 seconds, and `comparison.json` with raw, tracked, and draw boxes for every scanned frame. `--reuse-raw` re-tracks the saved raw detections after checking each source frame index and PTS. It does not export the whole video or SRT.

In the 204 audited source frames, the short `田` stays a separate second line in frames 79-104. The 21-second shirt button was detected in 34 of 59 frames from 20.433-22.367 s (longest continuous run: 12 frames); all 34 were filtered by text style while the real subtitle remained. The hand/stick at 24 s was detected in 6 frames from 23.967-24.300 s (longest run: 3), and the foreground hand at 123 s in 6 frames from 123.000-123.500 s (longest run: 2); both were filtered by position. No false box was drawn in these sampled windows. The subtitle ending at 122.967 s has no drawn box at 123.000 s. A stationary, subtitle-styled sign in the same band can still become a false event; this milestone does not claim to distinguish its meaning.
