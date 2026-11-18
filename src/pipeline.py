from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional

from pydantic import BaseModel, Field
import yaml

from .schemas import CommentaryHistoryEntry, CommentaryMode, WindowPipelineResult
from .stage1_video_stream import SlidingWindowVideoStreamer
from .stage2_context_memory import ContextAssembler
from .stage3_video_llm import UnifiedVideoLLMEngine
from .stage4_output_parser import StructuredOutputParser

logger = logging.getLogger(__name__)


class Stage1Config(BaseModel):
    window_duration_sec: float = 15.0
    overlap_duration_sec: float = 2.0
    sample_fps: float = 2.0
    decoder_backend: Literal["auto", "decord", "pyav"] = "auto"
    min_pixels: int = 224 * 224
    max_pixels: int = 360 * 640
    frame_tensor_device: str = "auto"


class Stage2Config(BaseModel):
    rolling_history_size: int = 4
    kb_top_k_candidates: int = 4
    kb_category_diversity: bool = True
    language: str = "vi"


class Stage3Config(BaseModel):
    model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    adapter_path: Optional[str] = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = 220
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.08


class Stage4Config(BaseModel):
    enable_overlap_dedup: bool = True
    overlap_tolerance_sec: float = 0.5


class OutputExportConfig(BaseModel):
    output_dir: str = "outputs"
    save_jsonl: bool = True
    save_srt: bool = True


class PipelineConfig(BaseModel):
    stage1_video: Stage1Config = Field(default_factory=Stage1Config)
    stage2_context: Stage2Config = Field(default_factory=Stage2Config)
    stage3_model: Stage3Config = Field(default_factory=Stage3Config)
    stage4_parser: Stage4Config = Field(default_factory=Stage4Config)
    output: OutputExportConfig = Field(default_factory=OutputExportConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
        return cls.model_validate(raw_cfg)


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


class DualModeSoccerCommentaryPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        self.video_streamer = SlidingWindowVideoStreamer(
            window_duration_sec=config.stage1_video.window_duration_sec,
            overlap_duration_sec=config.stage1_video.overlap_duration_sec,
            sample_fps=config.stage1_video.sample_fps,
            decoder_backend=config.stage1_video.decoder_backend,
            min_pixels=config.stage1_video.min_pixels,
            max_pixels=config.stage1_video.max_pixels,
            frame_tensor_device=config.stage1_video.frame_tensor_device,
        )

        self.video_llm = UnifiedVideoLLMEngine(
            model_name_or_path=config.stage3_model.model_name_or_path,
            adapter_path=config.stage3_model.adapter_path,
            torch_dtype=config.stage3_model.torch_dtype,
            attn_implementation=config.stage3_model.attn_implementation,
            device_map=config.stage3_model.device_map,
            max_new_tokens=config.stage3_model.max_new_tokens,
            temperature=config.stage3_model.temperature,
            top_p=config.stage3_model.top_p,
            do_sample=config.stage3_model.do_sample,
            repetition_penalty=config.stage3_model.repetition_penalty,
            min_pixels=config.stage1_video.min_pixels,
            max_pixels=config.stage1_video.max_pixels,
        )

        self.output_parser = StructuredOutputParser(
            enable_overlap_dedup=config.stage4_parser.enable_overlap_dedup,
            overlap_tolerance_sec=config.stage4_parser.overlap_tolerance_sec,
        )

    def preload_model(self) -> None:
        self.video_llm.load_model()

    def stream_commentary(
        self,
        video_source: str,
        match_metadata: dict[str, Any] | str | Path,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        on_window_callback: Optional[Callable[[WindowPipelineResult], None]] = None,
    ) -> Iterator[WindowPipelineResult]:
        if isinstance(match_metadata, (str, Path)):
            context_assembler = ContextAssembler.from_metadata_file(
                metadata_path=match_metadata,
                rolling_history_size=self.config.stage2_context.rolling_history_size,
                kb_top_k_candidates=self.config.stage2_context.kb_top_k_candidates,
                kb_category_diversity=self.config.stage2_context.kb_category_diversity,
            )
        else:
            context_assembler = ContextAssembler(
                metadata=match_metadata,
                rolling_history_size=self.config.stage2_context.rolling_history_size,
                kb_top_k_candidates=self.config.stage2_context.kb_top_k_candidates,
                kb_category_diversity=self.config.stage2_context.kb_category_diversity,
            )

        self.video_llm.load_model()

        for chunk in self.video_streamer.stream_chunks(
            video_source=video_source,
            start_sec=start_sec,
            end_sec=end_sec,
        ):
            sys_prompt, usr_prompt, live_state, kb_candidates = (
                context_assembler.assemble_window_context(chunk)
            )

            raw_llm_text = self.video_llm.generate_single_pass(
                chunk=chunk,
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
            )

            prev_entry = context_assembler.rolling_memory.last_entry()
            window_result = self.output_parser.parse_and_route(
                raw_text=raw_llm_text,
                chunk=chunk,
                match_state=live_state,
                prev_entry=prev_entry,
            )

            if not window_result.is_overlap_duplicate:
                if window_result.structured_output.mode == CommentaryMode.NO_EVENT:
                    used_kb_id = context_assembler.knowledge_base.mark_used_from_commentary(
                        candidates=kb_candidates,
                        commentary_text=window_result.structured_output.commentary,
                        window_index=chunk.window_index,
                    )
                    window_result.used_knowledge_id = used_kb_id

                context_assembler.rolling_memory.append(
                    CommentaryHistoryEntry(
                        window_index=chunk.window_index,
                        window_abs_start=chunk.window_start_sec,
                        window_abs_end=chunk.window_end_sec,
                        abs_event_start=window_result.abs_start_time,
                        abs_event_end=window_result.abs_end_time,
                        match_clock=window_result.match_clock,
                        mode=window_result.structured_output.mode,
                        event_type=window_result.structured_output.event_type,
                        commentary=window_result.structured_output.commentary,
                    )
                )
            else:
                logger.info(
                    "Bỏ qua đọc lặp sự kiện ở vùng gối đầu cửa sổ #%d ([%.1fs - %.1fs]).",
                    chunk.window_index,
                    window_result.abs_start_time,
                    window_result.abs_end_time,
                )

            del chunk

            if on_window_callback is not None:
                on_window_callback(window_result)

            yield window_result

    def run_to_files(
        self,
        video_source: str,
        match_metadata: dict[str, Any] | str | Path,
        output_prefix: str = "match_commentary",
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> list[WindowPipelineResult]:
        out_dir = Path(self.config.output.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = out_dir / f"{output_prefix}.jsonl"
        srt_path = out_dir / f"{output_prefix}.srt"

        results: list[WindowPipelineResult] = []
        jsonl_file = (
            open(jsonl_path, "w", encoding="utf-8") if self.config.output.save_jsonl else None
        )

        try:
            for res in self.stream_commentary(
                video_source=video_source,
                match_metadata=match_metadata,
                start_sec=start_sec,
                end_sec=end_sec,
            ):
                results.append(res)

                if jsonl_file is not None:
                    record = {
                        **res.structured_output.to_json_dict(),
                        "window_index": res.window_index,
                        "match_clock": res.match_clock,
                        "abs_start_time": res.abs_start_time,
                        "abs_end_time": res.abs_end_time,
                        "is_overlap_duplicate": res.is_overlap_duplicate,
                        "tts_prosody": res.tts_prosody.model_dump(),
                    }
                    jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_file.flush()

                logger.info(
                    "[Window #%03d | %s] %s (%s) [%.1fs-%.1fs]: %s",
                    res.window_index,
                    res.match_clock,
                    res.structured_output.mode.value,
                    res.structured_output.event_type,
                    res.structured_output.start_time,
                    res.structured_output.end_time,
                    res.structured_output.commentary,
                )
        finally:
            if jsonl_file is not None:
                jsonl_file.close()

        if self.config.output.save_srt:
            self._export_srt(results, srt_path)

        return results

    @staticmethod
    def _export_srt(results: list[WindowPipelineResult], srt_path: Path) -> None:
        active_results = [r for r in results if not r.is_overlap_duplicate]
        lines: list[str] = []

        for idx, res in enumerate(active_results, start=1):
            start_ts = _format_srt_timestamp(res.abs_start_time)
            end_sec = res.abs_end_time
            if idx < len(active_results):
                end_sec = min(end_sec, max(res.abs_start_time + 1.0, active_results[idx].abs_start_time - 0.05))
            end_ts = _format_srt_timestamp(end_sec)

            tag = (
                f"[{res.structured_output.event_type}] "
                if res.structured_output.mode == CommentaryMode.EVENT
                else "[BÊN LỀ] "
            )
            lines.append(f"{idx}")
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(f"{tag}{res.structured_output.commentary}")
            lines.append("")

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
