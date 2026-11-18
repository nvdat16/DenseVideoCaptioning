from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from .schemas import (
    CommentaryHistoryEntry,
    CommentaryMode,
    LiveMatchState,
    SoccerEventType,
    StructuredCommentaryOutput,
    TTSProsodyHint,
    VideoChunk,
    WindowPipelineResult,
)

logger = logging.getLogger(__name__)


class StructuredOutputParser:
    DRAMATIC_EVENTS = {
        SoccerEventType.GOAL.value,
        SoccerEventType.PENALTY.value,
    }
    EXCITED_EVENTS = {
        SoccerEventType.SHOT_ON_TARGET.value,
        SoccerEventType.SHOT_OFF_TARGET.value,
        SoccerEventType.SAVE.value,
        SoccerEventType.DANGEROUS_ATTACK.value,
    }

    def __init__(
        self,
        enable_overlap_dedup: bool = True,
        overlap_tolerance_sec: float = 0.5,
    ) -> None:
        self.enable_overlap_dedup = bool(enable_overlap_dedup)
        self.overlap_tolerance_sec = float(overlap_tolerance_sec)

    def parse_and_route(
        self,
        raw_text: str,
        chunk: VideoChunk,
        match_state: LiveMatchState,
        prev_entry: Optional[CommentaryHistoryEntry] = None,
    ) -> WindowPipelineResult:
        parsed_dict = self._extract_json_dict(raw_text, default_duration=chunk.duration_sec)
        structured = StructuredCommentaryOutput.model_validate(parsed_dict)
        structured = self._clamp_timestamps(structured, chunk.duration_sec)

        abs_start = round(chunk.window_start_sec + structured.start_time, 2)
        abs_end = round(chunk.window_start_sec + structured.end_time, 2)

        is_dup = False
        if self.enable_overlap_dedup and prev_entry is not None and chunk.overlap_sec > 0.0:
            is_dup = self._check_overlap_duplicate(
                structured=structured,
                chunk=chunk,
                abs_start=abs_start,
                prev_entry=prev_entry,
            )

        tts_hint = self._map_tts_prosody(structured)

        return WindowPipelineResult(
            window_index=chunk.window_index,
            window_abs_start=chunk.window_start_sec,
            window_abs_end=chunk.window_end_sec,
            match_clock=match_state.format_clock(),
            structured_output=structured,
            abs_start_time=abs_start,
            abs_end_time=abs_end,
            tts_prosody=tts_hint,
            is_overlap_duplicate=is_dup,
            raw_model_response=raw_text,
        )

    @classmethod
    def _extract_json_dict(cls, raw_text: str, default_duration: float) -> dict[str, Any]:
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        start_brace = cleaned.find("{")
        end_brace = cleaned.rfind("}")
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            candidate = cleaned[start_brace : end_brace + 1]
            candidate_fixed = re.sub(r",\s*}", "}", candidate)
            for attempt in (candidate, candidate_fixed):
                try:
                    data = json.loads(attempt)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue

        logger.warning("Không parse được JSON chuẩn, kích hoạt cơ chế khôi phục trường dữ liệu.")
        return cls._regex_fallback_parse(cleaned, default_duration=default_duration)

    @staticmethod
    def _regex_fallback_parse(text: str, default_duration: float) -> dict[str, Any]:
        mode_match = re.search(r'"mode"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
        event_match = re.search(r'"event_type"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
        start_match = re.search(r'"start_time"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
        end_match = re.search(r'"end_time"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
        comm_match = re.search(r'"commentary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)

        mode = mode_match.group(1) if mode_match else "NO_EVENT"
        event_type = event_match.group(1) if event_match else "NONE"
        start_t = float(start_match.group(1)) if start_match else 0.0
        end_t = float(end_match.group(1)) if end_match else default_duration
        commentary = comm_match.group(1).strip() if comm_match else text.strip()

        return {
            "mode": mode,
            "event_type": event_type,
            "start_time": start_t,
            "end_time": end_t,
            "commentary": commentary or "Hai đội vẫn đang duy trì thế trận giằng co trên sân.",
        }

    @staticmethod
    def _clamp_timestamps(
        output: StructuredCommentaryOutput,
        window_duration: float,
    ) -> StructuredCommentaryOutput:
        max_t = round(max(0.5, window_duration), 2)
        if output.mode == CommentaryMode.NO_EVENT:
            output.start_time = 0.0
            output.end_time = max_t
            return output

        output.start_time = round(min(max(0.0, output.start_time), max(0.0, max_t - 0.5)), 2)
        output.end_time = round(min(max(output.start_time + 0.5, output.end_time), max_t), 2)
        return output

    def _check_overlap_duplicate(
        self,
        structured: StructuredCommentaryOutput,
        chunk: VideoChunk,
        abs_start: float,
        prev_entry: CommentaryHistoryEntry,
    ) -> bool:
        if structured.mode != CommentaryMode.EVENT or prev_entry.mode != CommentaryMode.EVENT:
            return False

        confined_to_overlap = structured.end_time <= (chunk.overlap_sec + self.overlap_tolerance_sec)
        temporally_overlaps_prev = abs_start <= (prev_entry.abs_event_end + self.overlap_tolerance_sec)

        return bool(confined_to_overlap and temporally_overlaps_prev)

    @classmethod
    def _map_tts_prosody(cls, output: StructuredCommentaryOutput) -> TTSProsodyHint:
        if output.mode == CommentaryMode.NO_EVENT:
            return TTSProsodyHint(
                voice_style="warm_storytelling",
                speaking_rate=0.96,
                pitch_shift_st=-0.3,
                energy_level="calm",
            )

        if output.event_type in cls.DRAMATIC_EVENTS:
            return TTSProsodyHint(
                voice_style="dramatic_peak",
                speaking_rate=1.18,
                pitch_shift_st=1.5,
                energy_level="high",
            )

        if output.event_type in cls.EXCITED_EVENTS:
            return TTSProsodyHint(
                voice_style="excited_fast",
                speaking_rate=1.12,
                pitch_shift_st=1.0,
                energy_level="high",
            )

        return TTSProsodyHint(
            voice_style="tense_focused",
            speaking_rate=1.05,
            pitch_shift_st=0.3,
            energy_level="medium",
        )
