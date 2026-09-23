# Đặc tả architecture và plan triển khai — Đóng khung phụ đề tiếng Trung

## Mục tiêu và luồng dữ liệu

Nhận một MP4 từ `data/`, phát hiện phụ đề tiếng Trung hardcoded trong ROI phía dưới, vẽ box ổn định ôm từng dòng, giữ nguyên audio và xuất JSON có tọa độ cùng timestamp. SRT là tùy chọn.

Hệ thống được thiết kế để nhận MP4 H.264/H.265 ở độ phân giải phổ biến 720p–1080p. Vì chỉ có một video nguồn để nghiệm thu, kết quả chất lượng chỉ được xác nhận trên video đó; khả năng xử lý mọi MP4 là mục tiêu tương thích, chưa phải kết quả đã kiểm chứng.

```text
MP4 → frame + PTS gốc → ROI phía dưới → PaddleOCR detection → ghép dòng
    → subtitle event → làm mượt box → vẽ lại frame → MP4 + JSON (+ SRT)
```

## Architecture và giao diện module

| Module | Trách nhiệm |
| --- | --- |
| `video_io` | Dùng PyAV đọc thông tin MP4, giải mã frame theo PTS thực, áp rotation, mã hóa video đã vẽ và sao chép audio từ MP4 nguồn vào kết quả. Kiểm tra đồng bộ giữa video và audio sau khi xuất. |
| `detect` | Khởi tạo PaddleOCR `TextDetection` một lần trên thiết bị được chọn; nhận ảnh ROI và trả polygon cùng điểm tin cậy theo tọa độ toàn frame. |
| `lines` | Lọc detection yếu, gom vùng chữ có cùng hàng/baseline thành từng dòng, tạo box sát chữ với padding nhỏ và giới hạn box trong frame. Không loại một dòng chỉ vì box hẹp: ảnh mẫu có dòng dưới chỉ gồm ký tự `田`. |
| `events` | Ghép dòng qua các frame theo vị trí, IoU và thứ tự dòng; dùng dấu vân tay nét chữ để tách phụ đề mới tại cùng vị trí; chịu được mất detection ngắn và chốt box ổn định cho mỗi dòng trong từng event. |
| `render_export` | Vẽ box theo timeline, ghi JSON; khi bật `--srt`, nhận dạng chữ trên frame đại diện của mỗi event rồi ghi cue có nội dung. |
| `main` / `config` | Cung cấp CLI, cấu hình ROI/ngưỡng/thiết bị, kiểm tra đầu vào và phần cứng, điều phối hai lượt xử lý, báo lỗi rõ ràng và chạy benchmark CPU/GPU. |

### Cấu trúc dữ liệu và giao diện

- `VideoFrame(index, pts, image_bgr)` giữ PTS gốc và ảnh BGR; lưu kèm time base hoặc chuẩn hóa PTS theo cùng một mốc thời gian.
- `TextRegion(polygon, score)` là polygon phát hiện được và điểm tin cậy. Một polygon có thể bao một cụm chữ hoặc cả dòng, không mặc định là một token.
- `LineDetection(bbox_xyxy, score)` là box một dòng phụ đề sau khi ghép các vùng chữ khi cần.
- `FrameObservation(frame_index, time_sec, lines, fingerprints)` giữ detection và tín hiệu ảnh nhỏ gọn của từng frame.
- `FrameBoxes(frame_index, time_sec, event_id, boxes_xyxy)` là box sau làm mượt dùng chung cho video và JSON.
- `SubtitleEvent(id, start_frame, end_frame_exclusive, start_sec, end_sec, lines)` nhóm các dòng thuộc cùng một phụ đề theo thời gian.
- `detect(frame, roi) -> list[TextRegion]` phát hiện vùng chữ trong ROI và đổi tọa độ về toàn frame.
- `merge_lines(regions) -> list[LineDetection]` ghép polygon thành dòng.
- `build_events(observations) -> (list[SubtitleEvent], list[FrameBoxes])` theo dõi, tách event và làm mượt box.
- `render_frame(image, boxes) -> image` vẽ các box đang hoạt động tại PTS của frame.

JSON ghi metadata video và cấu hình, danh sách event, cùng box được vẽ theo từng frame. Tọa độ là pixel toàn frame dạng `xyxy` nửa mở; timestamp xuất theo mili giây; event dùng khoảng `[start_ms, end_ms)`. Tính toán thời gian nội bộ dựa trên PTS/time base, không suy timestamp từ FPS.

CLI nhận `--input`, `--output`, `--json`, `--roi-bottom` và `--device cpu|gpu[:index]`. `--roi-bottom` cho phép chọn 0.25–0.45 (25%–45% chiều cao tính từ mép dưới), mặc định **0.45**. Trong ảnh mẫu cao 671 px, chữ bắt đầu khoảng y=423: ROI 25% bắt đầu tại y=503 và bỏ sót chữ; ROI 40% bắt đầu tại y=402, còn ROI 45% bắt đầu tại y=369 và có thêm khoảng đệm. Mặc định thiết bị là `cpu`; cùng thiết bị được dùng cho detection và nhận dạng chữ khi bật SRT. Nếu người dùng chọn GPU nhưng bản PaddlePaddle không hỗ trợ CUDA, GPU index không hợp lệ hoặc model không khởi tạo được, chương trình dừng trước khi xử lý video và báo lỗi; không tự chuyển sang CPU.

## Hợp đồng triển khai để giao cho Luna

### 0. Khảo sát repo và môi trường trước khi viết code

- Liệt kê file thực tế, đọc `video_io.py` nếu có, xác định Python/OS/FFmpeg, codec của MP4, PTS/VFR, audio, rotation, CPU/GPU và phiên bản thư viện. Không giả định repo đã có code hoặc máy có GPU dùng được.
- Khởi tạo project tối thiểu nếu repo rỗng. Dự kiến `main.py`, `src/{config,schemas,video_io,detect,lines,events,render_export}.py`, `tests/`, `data/`, `outputs/`, `requirements.txt`, `README.md`. Có thể giữ tên file hiện hữu; báo rõ mọi thay đổi so với cấu trúc này. Không commit video/model vào repo.
- Chốt **một phiên bản PaddleOCR 3.x và PyAV cài được trên môi trường thực** rồi lưu phiên bản trong dependency file. Dùng API detection `TextDetection` của bản đã chọn; chạy một phép inference trên một frame ROI trước khi xây wrapper. Kiểm tra chính xác cấu trúc kết quả, điểm tin cậy và tọa độ; không dùng tham số `rec=False` của API cũ theo suy đoán. Bản detection có thể trả vùng chứa cả dòng, không ép tách thành token.
- Không tự động tải model trong lúc benchmark. Lần chạy đầu có thể tải model; ghi rõ model, nơi cache, phiên bản. Cài CPU và GPU theo hướng dẫn tương thích thiết bị thực; chọn `gpu:0` phải fail-fast nếu khởi tạo model lỗi.

### 1. Schema, tọa độ và thời gian: quy ước duy nhất

| Kiểu | Các trường tối thiểu và bất biến |
| --- | --- |
| `VideoInfo` | `width`, `height` **sau khi áp rotation**, codec, FPS tham khảo, `video_time_base`, audio streams, start time và duration nếu lấy được. FPS chỉ dùng để hiển thị/báo cáo, không tính timestamp. |
| `VideoFrame` | `index`, `source_pts`, `time_base`, `time_sec: Fraction`, `image_bgr`. `index` liên tục theo thứ tự hiển thị; PTS có thể không bắt đầu từ 0. Nếu nguồn thiếu PTS, báo rõ chính sách fallback và không tuyên bố giữ PTS gốc. |
| `TextRegion` | `polygon_xy` của toàn frame, `score ∈ [0,1]`. Nếu detection trả toạ độ ROI `(x,y)`, chuyển thành `(x, y+roi_y0)` **đúng một lần**. |
| `LineDetection` | `bbox_xyxy`, `score`, tùy chọn `regions`. `x2,y2` là biên nửa mở; luôn `0 ≤ x1 < x2 ≤ width`, `0 ≤ y1 < y2 ≤ height`. |
| `FrameObservation` | `frame_index`, `time_sec`, danh sách line, tín hiệu ảnh/fingerprint theo **từng line** nếu có. Lưu metadata nhỏ, không giữ toàn bộ frame BGR trong RAM. |
| `FrameBoxes` | `frame_index`, `time_sec`, `event_id|null`, `boxes_xyxy` **sau làm mượt**. Đây là dữ liệu chung duy nhất để vẽ MP4 và xuất JSON, tránh sai khác. |
| `SubtitleEvent` | `id`, `start_frame`, `end_frame_exclusive`, `start_sec`, `end_sec`, các line/box đại diện. Mỗi line có thể có khoảng tồn tại riêng khi phụ đề hai dòng thay đổi không đồng thời. |

- ROI: `roi_y0 = floor(height * (1 - roi_bottom))`, lấy `[roi_y0:height, 0:width]` trên frame đã xoay. Padding box cấu hình bằng pixel và chỉ thêm **sau** khi ghép dòng; clamp vào frame. Không lọc theo `min_width` tuyệt đối. Nếu muốn bỏ biển hiệu/logo trong ROI, dùng nhất quán vị trí, tính bền theo thời gian và trường hợp được đánh giá; không khẳng định detection đơn thuần biết tiếng Trung.
- Thời gian nội bộ dùng `Fraction(pts) * time_base`, có cùng trục thời gian với cả video/audio; timestamp JSON/SRT tính từ mốc bắt đầu trình chiếu của video được xuất. Khi xuất, giữ `source_pts` riêng và ghi `timestamp_ms` tương đối (số thực, làm tròn khi serialize). Phân biệt `frame_index` và timestamp vì VFR có thể có khoảng cách giữa frame không đều.
- Event dùng `[start_frame, end_frame_exclusive)` và `[start_ms, end_ms)`. `end` của event cuối lấy từ thời điểm kết thúc hiển thị frame cuối (duration từ nguồn nếu có; nếu không thì ước lượng từ các delta PTS hợp lệ và ghi rõ), không mặc nhiên lấy `last_pts` hoặc `frame_count/fps`.
- CLI ban đầu: `python main.py --input data/input.mp4 --output outputs/boxed.mp4 --json outputs/boxes.json --device cpu --roi-bottom 0.45`; thêm `--srt outputs/subtitles.srt` và cờ benchmark/đoạn thử sau khi chức năng cơ bản chạy. Báo lỗi path input thiếu, output trùng input, ROI ngoài khoảng, GPU không có và encoder không có.

### 2. `video_io`: hai lượt đọc, xử lý audio và rotation

1. `probe_video(path) -> VideoInfo` và `iter_frames(path) -> Iterator[VideoFrame]`: chỉ chọn video stream đích, decode tuần tự, tạo BGR theo thứ tự hiển thị; ghi nhận rotation/display matrix, áp đúng một lần rồi bỏ/chỉnh metadata xoay ở output để không xoay hai lần. Kiểm tra hướng bằng frame trước và sau.
2. Lượt 1 chỉ lưu `FrameObservation`/fingerprint ngắn để dựng timeline. Lượt 2 mở lại **cùng file**, đọc tuần tự, tra `FrameBoxes` theo `frame_index` **và đối chiếu PTS**, vẽ frame rồi encode. Nếu số frame/PTS giữa hai lượt không khớp, dừng với lỗi rõ ràng.
3. Mã hóa video output H.264 trong MP4 bằng encoder khả dụng; đặt PTS frame theo time base encoder sau khi đổi từ thời gian nguồn tương đối, đảm bảo PTS video theo thứ tự hiển thị và mux packet với time base phù hợp. Kiểm tra output VFR nếu nguồn VFR, không mặc định đặt `pts = index` cho mọi video. Khi encoder/muxer không đáp ứng được VFR nguồn, ghi giới hạn cụ thể và **không đánh dấu đạt yêu cầu giữ timestamp**.
4. Với audio: ưu tiên remux stream/audio packet có codec tương thích MP4, đổi time base đúng và bảo toàn offset tương đối audio/video. Nếu remux thất bại do codec/container, chuyển mã audio sang AAC và ghi rõ chế độ fallback; tuyệt đối không lặng lẽ bỏ audio. Nếu nguồn không có audio, output không cần audio. Có thể dùng FFmpeg riêng để ghép audio sau encode nếu PyAV mux đồng thời quá khó, nhưng vẫn phải giữ offset và kiểm tra đồng bộ.
5. Smoke test **trước OCR**: vẽ box cố định trên đoạn cắt từ MP4 nguồn; dùng `ffprobe` kiểm tra video/audio stream, số frame, thời lượng và start time; xem bằng mắt chỗ có tiếng để phát hiện lệch. Cắt đoạn thử từ nguồn, không cần video thứ hai.

### 3. `detect` và `lines`: thuật toán tối thiểu

1. `Detector(device, model_name, thresholds)` khởi tạo một lần; `detect(frame_bgr, roi_bottom) -> list[TextRegion]`. Chạy detection trên ROI BGR, kiểm tra output của đúng phiên bản cài đặt, lấy polygon + score và đổi sang toạ độ toàn frame. Thử `cpu` trước, `gpu:0` sau khi môi trường hỗ trợ. Khi GPU batch được hỗ trợ thật, giữ thứ tự frame tương ứng với kết quả model; không ép batch nếu RAM/VRAM không đủ.
2. `merge_lines(regions, frame_size, config)`: bỏ polygon lỗi/score quá thấp, tính vùng bao; nhóm các vùng có chồng lấp theo trục y hoặc baseline gần nhau và khoảng cách ngang hợp lý. Sắp từ trên xuống; cùng hàng từ trái sang phải; có thể giữ nguyên một vùng đã bao cả dòng. Tránh nối hai dòng khác hàng thành một box lớn. Chỉ tạo bbox cho line đã có ít nhất **một** detection hợp lệ, kể cả dòng một chữ `田`.
3. Cung cấp `debug_frames/` gồm ảnh ROI và ảnh toàn frame có polygon raw + box sau ghép ở đoạn một dòng, hai dòng, `田`, không có chữ. Nếu ảnh không có trường hợp đó, tìm mốc khác trong **cùng video** và ghi mốc thời gian.

### 4. `events`: quy tắc xử lý thời gian, chưa cần OCR toàn video

- Ghép line ở frame kế tiếp theo **thứ tự dòng + khoảng cách tâm chuẩn hóa theo chiều cao chữ + overlap/IoU**; khi nhiều ứng viên, ghép một-một theo chi phí nhỏ nhất trong ngưỡng. So chiều cao, tránh hoán đổi dòng trên/dưới. Ngưỡng đặt trong `config` và báo giá trị đã dùng; mặc định ban đầu chỉ là điểm khởi đầu, chỉnh dựa trên video.
- Lưu fingerprint nét chữ từ crop ROI của từng line đã căn chỉnh/kích thước chuẩn; đo thay đổi giữa các frame chất lượng đủ tốt. Dùng tín hiệu khác biệt qua **nhiều frame liên tiếp** và phép ghép hình học để phân biệt câu mới tại cùng vị trí với jitter/nhiễu. Nếu hai câu quá giống, đánh dấu tình huống chưa chắc chắn để kiểm tra thủ công; OCR theo event là phương án bổ sung, không cam kết fingerprint luôn đúng.
- Gap tối đa mặc định **2 frame**: chỉ nội suy/giữ box qua gap ngắn khi có detection của **cùng line/event** ở cả trước và sau; không giữ box sau detection cuối khi chữ biến mất. Sau khi đã nhìn được frame tiếp theo, chốt start/end của event; ở ranh giới hai câu, không nội suy box cũ sang câu mới. Frame có chữ nhưng detector mất liên tiếp quá 2 frame không được tự tạo box vô hạn.
- Trong một event ổn định, dùng median/robust statistics của box các frame tin cậy để chặn outlier và ổn định cạnh box, có thể cập nhật nhẹ nếu vị trí chữ di chuyển. Giữ box từng dòng riêng; giới hạn cạnh box để không cắt chữ quan sát được. Event và `FrameBoxes` được dựng **offline** sau lượt 1, nên có thể sử dụng cả frame trước/sau khi lấp gap.
- Test giả lập bao gồm: 2 dòng độc lập, chữ `田` một mình, jitter ± vài pixel, một/two frame rỗng giữa cùng câu, >2 frame rỗng, biến mất vĩnh viễn, câu khác ở cùng vị trí, biển hiệu đứng yên. Assert cụ thể event boundary, số box mỗi frame, độ ổn định và không box sau khi phụ đề biến mất.

### 5. Vẽ, JSON, SRT và tính nhất quán

- `render_frame(image_bgr, frame_boxes)` chỉ vẽ những box nằm trong frame đó; sử dụng đúng một `FrameBoxes` làm input cho vẽ và ghi JSON. Màu/nét vẽ cấu hình được, nhưng không dùng nét quá dày che chữ. Chỉ serialise số nhỏ (`int/float/str`), không nhét ảnh gốc vào JSON.
- JSON **có schema_version** và gồm `video` (thông tin nguồn/output), `config` (ROI, model, device, thresholds), `events` và `frames`. Mỗi `frames[i]` có `frame_index`, `source_pts`, `timestamp_ms`, `event_id`, `boxes_xyxy`; frame không có phụ đề vẫn có mảng `[]`. `events` có `start_ms`, `end_ms`, `line_boxes` đại diện và `text` nếu đã OCR. Ví dụ: `{"frame_index":12,"source_pts":12000,"timestamp_ms":400.0,"event_id":1,"boxes_xyxy":[[120,420,600,461],[305,475,330,507]]}`. Không coi tọa độ minh hoạ là nhãn của video thật.
- `--srt` chạy recognition chỉ trên frame đại diện đủ nét của **mỗi event**, crop từng dòng kèm padding, giữ thứ tự trên xuống; cue theo `[start_ms,end_ms)`. Event không OCR được ghi `unrecognized_events` trong JSON/log và bỏ cue rỗng. Khi không có `--srt`, không khởi tạo recognition model; khi chỉ chạy detection, không tự suy ra mọi vùng chữ là tiếng Trung.
- Xuất file đầu ra qua đường dẫn tạm rồi đổi tên khi thành công; nếu lỗi encode/OCR, báo lỗi và tránh để MP4 được coi là sản phẩm hoàn chỉnh. SRT tùy chọn thất bại phải được báo riêng, không thay đổi JSON/MP4 đã xác nhận.

### 6. Benchmark tái lập được

- Chọn một/tối đa vài đoạn từ cùng MP4 và cố định danh sách `frame_index`/PTS cho hai thiết bị. Chạy cùng model, ngưỡng, ROI, kích thước ảnh, stride và batch size; tách **startup/model load**, warm-up, detection thuần, encode và toàn pipeline. Không lấy tốc độ GPU trên batch khác đem so trực tiếp với CPU.
- Ghi Python, PaddlePaddle/PaddleOCR/PyAV, CPU/GPU/driver; đo wall-clock, frame count, FPS thực (`số frame / giây xử lý`), RAM/VRAM peak nếu đo được và chênh lệch số line/event/box giữa hai thiết bị. Nếu GPU không sẵn, báo `not run` và lý do, không thay bằng ước tính.

## Mốc giao việc cho Luna: làm và review tuần tự

Giữ cùng một repo và file plan. Mỗi mốc phải trả: file đã sửa, lệnh chạy từ thư mục gốc, output/log thực, test pass/fail, giới hạn còn lại. **Không tự báo mốc hoàn thành nếu thiếu bằng chứng tương ứng.** Sau khi người dùng review mốc hiện tại mới triển khai mốc sau; nếu không có MP4 ở `data/`, làm đến test giả lập, nêu chính xác phần bị chặn và chờ video nguồn.

| Mốc | Việc phải làm | Tiêu chí qua mốc |
| --- | --- | --- |
| 0. Khảo sát | Kiểm tra repo/môi trường và MP4; chốt dependency + `README` cài CPU/GPU. | Báo file hiện có, codec/PTS/audio/rotation, phiên bản thật; không ghi phần chưa kiểm tra là đã chạy. |
| 1. CLI/schema | `config`, `schemas`, validate input, ROI/device fail-fast, CLI skeleton. | Test ROI 0.25/0.45, input thiếu, device sai; lệnh `--help` chạy. |
| 2. Video I/O | Đọc → box cố định → MP4 + audio trong đoạn thử. | Clip xem được, PTS/thời lượng đúng, audio đồng bộ. **Sol review đoạn này** nếu có điều kiện. |
| 3. Detection/lines | ROI → TextDetection → polygons → line boxes; ảnh debug. | Một/hai dòng và dòng ngắn không bị loại; box trong biên, ghép dòng đúng trên ảnh được xem. |
| 4. Events | Tracking, nhận biết đổi câu, gap nội bộ, smoothing; timeline + clip quanh đổi câu. | Unit test tình huống giả lập, clip không chớp 1–2 frame, không kéo box sau khi câu kết thúc. |
| 5. Output | Lượt 2, JSON, MP4; `--srt` tùy chọn. | JSON trùng box MP4 ở frame đối chiếu, MP4/audio mở được, SRT không có cue rỗng. **Sol review tích hợp** nếu có điều kiện. |
| 6. Nghiệm thu | Chọn/nhãn 30–50 frame từ video nguồn, đối chiếu hình và JSON. | Báo đúng/bỏ sót/box nhầm, IoU, flicker, time boundary; chỉ kết luận trên video được cấp. |
| 7. Benchmark | CPU/GPU cùng input/cấu hình, log máy và phép đo. | Bảng thời gian/nguồn lực/tốc độ hoặc `not run` cho GPU thiếu môi trường. |

## Kiểm thử và tiêu chí nghiệm thu

- Dữ liệu giả lập bao gồm một hoặc hai dòng, vùng chữ rời trong cùng dòng, dòng rất ngắn như `田`, jitter tọa độ, mất detection ngắn, frame trống và câu mới xuất hiện cùng vị trí. Kiểm tra line grouping, event boundary, box ổn định và khoảng không có phụ đề.
- Kiểm tra `--device cpu`, `--device gpu:0`, GPU không khả dụng và GPU index sai; xác nhận model detection/recognition nhận đúng thiết bị. Chạy smoke test GPU khi môi trường GPU đã cài đặt.
- Trên video đích, xem mẫu frame đầu/giữa/cuối và quanh các lần đổi phụ đề: box ôm từng dòng, không cắt chữ, không nhảy trong event và không lưu lại khi phụ đề đã biến mất.
- Gán nhãn khoảng 30–50 frame/đoạn đại diện từ video nguồn, gồm một dòng, hai dòng, lúc đổi câu và lúc không có chữ. Báo cáo số dòng phát hiện đúng/bỏ sót, box nhầm, IoU với nhãn, frame mất box, mức dao động box và sai lệch thời điểm xuất hiện/biến mất. Chỉ kết luận theo dữ liệu thực sự đã kiểm tra.
- JSON có tọa độ trong biên frame, timestamp theo thứ tự hiển thị và khớp box trên video. Video xuất mở được, số frame/thời lượng phù hợp; nếu đầu vào có audio thì xác nhận stream audio và đồng bộ.
- Khi bật `--srt`, kiểm tra cue có timestamp và nội dung tiếng Trung đọc được; báo rõ các event không nhận dạng được thay vì tạo cue rỗng. Khi không bật SRT, detection chỉ xác định vùng chữ, chưa tự xác nhận ngôn ngữ của mọi box.
- Benchmark báo thời gian và FPS CPU/GPU dưới cùng cấu hình; nếu số event hoặc tọa độ khác đáng kể thì báo chênh lệch chất lượng bên cạnh tốc độ.
- Báo cáo cuối ghi đường dẫn sản phẩm, số frame/event, thiết bị dùng thực tế, kết quả kiểm tra audio và các giới hạn quan sát được trên video đích.

## Đầu ra cần review theo từng mốc

| Mốc | Bằng chứng bàn giao để review |
| --- | --- |
| Video I/O | Lệnh chạy lại và video ngắn có box cố định; xác nhận đúng chiều, thời lượng và audio đồng bộ. |
| Detection | Ảnh debug từ các frame có một hoặc hai dòng; có trường hợp dòng ngắn `田`; polygon và box không vượt biên. |
| Event và smoothing | Clip ngắn trước, trong và sau lúc đổi câu; xem box có mất 1–2 frame, nhảy tọa độ hoặc lưu lại sau khi chữ biến mất không. |
| Đầu ra | MP4 đầy đủ, JSON có thể đối chiếu từng frame; SRT và các event không nhận dạng được nếu bật `--srt`. |
| Benchmark | Bảng CPU/GPU cùng cấu hình, lệnh tái chạy, thời gian từng bước, FPS, bộ nhớ và sai khác kết quả nếu có. |

Khi hoàn tất, bàn giao mã nguồn, hướng dẫn cài/chạy, kết quả unit test, đường dẫn các đầu ra và danh sách lỗi còn quan sát được trên video. Review lần lượt theo từng mốc trước khi xem kết quả cuối.

## Giả định cần xác minh khi bắt đầu

- Người dùng sẽ chép **một MP4 nguồn** vào `data/`; `--srt` mặc định tắt. Có thể cắt đoạn thử từ chính MP4 này trong quá trình phát triển.
- Bắt đầu từ bộ khung dự án hiện có nếu thực sự có; nếu chưa có code thì tạo các module nêu trên từ đầu. Kiểm tra hệ điều hành, CPU/GPU, driver, FFmpeg và các thư viện trước khi chọn bản PaddlePaddle CPU/GPU phù hợp.
