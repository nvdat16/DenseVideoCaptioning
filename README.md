# End-to-End Dual-Mode Soccer Commentary Pipeline

Hệ thống sinh bình luận bóng đá tự động **2 chế độ (`EVENT` / `NO_EVENT`)** theo kiến trúc **Single-stage End-to-End** kết hợp **Sliding Window + Context Memory**, đọc trực tiếp luồng video/MP4 lên RAM/VRAM và **hoàn toàn loại bỏ bước trích xuất hay lưu file trung gian `.npy`**.

---

## 1. Cấu trúc thư mục & Ánh xạ từng giai đoạn

```text
DenseVideoCaptioning/
├── configs/
│   └── default_config.yaml          # Cấu hình cửa sổ trượt, FPS, Qwen2.5-VL, bộ nhớ ngữ cảnh
├── data/
│   └── sample_match_metadata.json   # Mẫu metadata trận đấu (đội hình, diễn biến, kho tin bên lề)
├── src/
│   ├── schemas.py                   # Pydantic schemas & cấu trúc dữ liệu xuyên suốt 4 giai đoạn
│   ├── stage1_video_stream.py       # Giai đoạn 1: Sliding Window Chunking & Dynamic Sampling (Decord/PyAV -> VRAM)
│   ├── stage2_context_memory.py     # Giai đoạn 2: Live Match State + External KB (chống lặp) + Rolling History
│   ├── stage3_video_llm.py          # Giai đoạn 3: Unified Video-LLM (Qwen2.5-VL + MRoPE Single-pass Forward)
│   ├── stage4_output_parser.py      # Giai đoạn 4: Structured JSON Parser + Khử trùng vùng gối đầu + TTS Prosody
│   └── pipeline.py                  # Bộ điều phối End-to-End (Streaming Generator & Xuất JSONL/SRT)
├── main.py                          # CLI chạy trực tiếp trên file MP4 90 phút hoặc luồng RTSP/HLS
└── requirements.txt                 # Thư viện phụ thuộc
```

---

## 2. Cơ chế hoạt động chi tiết của 4 giai đoạn

1. **Giai đoạn 1 (`src/stage1_video_stream.py`) — Video Chunking & Dynamic Sampling**
   - Cửa sổ trượt `15.0s`, gối đầu `2.0s` (`stride = 13.0s`), lấy mẫu ở `2.0 FPS` (`30 frames/window`).
   - Hỗ trợ tự động cả `decord` (truy xuất batch frame cực nhanh kèm **Overlap Frame Cache** để không bao giờ giải mã lại 2 giây gối đầu) và `pyav` (hỗ trợ cả file `.mp4` lẫn luồng trực tiếp `rtsp://`, `rtmp://`, `.m3u8` qua **Ring Buffer**).
   - Chuẩn hóa kích thước ảnh theo bội số `28` (`ViT patch 14 x spatial merge 2`) ngay trên Tensor `(T, C, H, W)` và cấp tọa độ thời gian thực `t = 0.0s, 0.5s, 1.0s...` cho MRoPE.

2. **Giai đoạn 2 (`src/stage2_context_memory.py`) — Context & Memory Assembly**
   - **`LiveMatchStateTracker`**: Tự động đồng bộ phút thi đấu, hiệp đấu, tỷ số, thẻ phạt và cầu thủ thay người tính đến thời điểm hiện tại của cửa sổ (không rò rỉ dữ liệu tương lai).
   - **`ExternalKnowledgeBase`**: Truy xuất các thông tin bên lề chưa từng được nói, ưu tiên thực thể vừa xuất hiện trên sân và đa dạng hóa chủ đề. Chỉ đánh dấu *đã dùng* khi mô hình thực sự kích hoạt nhánh `NO_EVENT`.
   - **`RollingCommentaryMemory`**: Lưu 3–5 câu bình luận gần nhất và cảnh báo vùng gối đầu `2.0s` nếu cửa sổ trước vừa tường thuật một sự kiện ở sát rìa cuối.

3. **Giai đoạn 3 (`src/stage3_video_llm.py`) — Unified Multimodal Forward Pass**
   - Truyền trực tiếp `chunk.frames` + `fps` vào `Qwen2_5_VLForConditionalGeneration` để kích hoạt `second_per_grid_ts` trong **MRoPE**.
   - Hỗ trợ gắn thêm LoRA adapter (`adapter_path`) nếu sử dụng mô hình đã fine-tune trên dữ liệu bóng đá (như MatchTime / SoccerNet-Caption).
   - Tự động dọn dẹp tensor trung gian sau mỗi cửa sổ để mức tiêu thụ VRAM luôn phẳng trong suốt 90 phút.

4. **Giai đoạn 4 (`src/stage4_output_parser.py`) — Structured Decoding & TTS Routing**
   - Phân tích cú pháp JSON an toàn về đúng cấu trúc chuẩn:
     ```json
     {
       "mode": "EVENT",
       "event_type": "SHOT_ON_TARGET",
       "start_time": 4.5,
       "end_time": 8.0,
       "commentary": "Pha phối hợp một chạm rất nhanh! Saka tung cú cứa lòng chân trái hiểm hóc nhưng thủ môn đã bay người cứu thua xuất sắc."
     }
     ```
   - Tự động kiểm tra khử trùng sự kiện vùng gối đầu (`is_overlap_duplicate`) và sinh kèm gợi ý ngữ điệu `TTSProsodyHint` (`dramatic_peak`, `excited_fast`, `tense_focused`, `warm_storytelling`) cho bộ đọc TTS phía sau.

---

## 3. Hướng dẫn sử dụng (Khi sẵn sàng chạy thử)

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Chạy xuất file `.jsonl` và phụ đề `.srt`
```bash
python main.py \
  --video /duong/dan/tran_dau_90p.mp4 \
  --metadata data/sample_match_metadata.json \
  --config configs/default_config.yaml \
  --output-prefix arsenal_liverpool
```

### Chạy chế độ Streaming JSON thời gian thực (để nối thẳng sang TTS)
```bash
python main.py \
  --video rtsp://127.0.0.1:8554/live_match \
  --metadata data/sample_match_metadata.json \
  --stream-json-stdout
```

### Tích hợp trực tiếp bằng Python Generator
```python
from src import DualModeSoccerCommentaryPipeline, PipelineConfig

config = PipelineConfig.from_yaml("configs/default_config.yaml")
pipeline = DualModeSoccerCommentaryPipeline(config)

for window_res in pipeline.stream_commentary(
    video_source="match.mp4",
    match_metadata="data/sample_match_metadata.json",
):
    if not window_res.is_overlap_duplicate:
        payload = window_res.structured_output.to_json_dict()
        tts_style = window_res.tts_prosody.voice_style
        # Đẩy `payload` + `tts_style` sang module Text-to-Speech
```
