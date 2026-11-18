from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from src.pipeline import DualModeSoccerCommentaryPipeline, PipelineConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-End Single-Stage Dual-Mode Soccer Video Commentary Pipeline"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Đường dẫn file video .mp4/.mkv 90 phút hoặc URL luồng trực tiếp (rtsp://, rtmp://, .m3u8).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/sample_match_metadata.json",
        help="Đường dẫn file JSON chứa đội hình, trạng thái trận đấu và kho tin bên lề.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Đường dẫn file cấu hình YAML của toàn bộ đường ống.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ghi đè tên hoặc đường dẫn mô hình Video-LLM (vd: Qwen/Qwen2.5-VL-3B-Instruct).",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Đường dẫn LoRA adapter đã fine-tune trên bài toán bình luận bóng đá (tùy chọn).",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=None,
        help="Ghi đè độ dài cửa sổ trượt (giây, vd: 12.0 hoặc 15.0).",
    )
    parser.add_argument(
        "--overlap-sec",
        type=float,
        default=None,
        help="Ghi đè độ dài vùng gối đầu giữa 2 cửa sổ (giây, vd: 2.0).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Ghi đè tần số lấy mẫu khung hình (vd: 2.0 FPS).",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Mốc thời gian bắt đầu xử lý trong video (giây, mặc định 0.0).",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Mốc thời gian kết thúc xử lý trong video (giây, mặc định chạy hết trận).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="match_commentary",
        help="Tên tiền tố cho file kết quả .jsonl và .srt trong thư mục outputs/.",
    )
    parser.add_argument(
        "--stream-json-stdout",
        action="store_true",
        help="In trực tiếp chuỗi JSON chuẩn 5 trường ra stdout ngay sau mỗi cửa sổ (để pipe sang TTS).",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stderr,
    )

    args = build_arg_parser().parse_args()

    cfg_path = Path(args.config)
    config = PipelineConfig.from_yaml(cfg_path) if cfg_path.exists() else PipelineConfig()

    if args.model is not None:
        config.stage3_model.model_name_or_path = args.model
    if args.adapter is not None:
        config.stage3_model.adapter_path = args.adapter
    if args.window_sec is not None:
        config.stage1_video.window_duration_sec = args.window_sec
    if args.overlap_sec is not None:
        config.stage1_video.overlap_duration_sec = args.overlap_sec
    if args.fps is not None:
        config.stage1_video.sample_fps = args.fps

    pipeline = DualModeSoccerCommentaryPipeline(config=config)

    if args.stream_json_stdout:
        for res in pipeline.stream_commentary(
            video_source=args.video,
            match_metadata=args.metadata,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
        ):
            if not res.is_overlap_duplicate:
                print(
                    json.dumps(res.structured_output.to_json_dict(), ensure_ascii=False, indent=2),
                    flush=True,
                )
    else:
        pipeline.run_to_files(
            video_source=args.video,
            match_metadata=args.metadata,
            output_prefix=args.output_prefix,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
