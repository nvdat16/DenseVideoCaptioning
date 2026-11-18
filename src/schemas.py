from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

import torch
from pydantic import BaseModel, Field, field_validator, model_validator


class CommentaryMode(str, Enum):
    EVENT = "EVENT"
    NO_EVENT = "NO_EVENT"


class SoccerEventType(str, Enum):
    GOAL = "GOAL"
    SHOT_ON_TARGET = "SHOT_ON_TARGET"
    SHOT_OFF_TARGET = "SHOT_OFF_TARGET"
    SAVE = "SAVE"
    DANGEROUS_ATTACK = "DANGEROUS_ATTACK"
    CORNER = "CORNER"
    FREE_KICK = "FREE_KICK"
    PENALTY = "PENALTY"
    FOUL = "FOUL"
    YELLOW_CARD = "YELLOW_CARD"
    RED_CARD = "RED_CARD"
    OFFSIDE = "OFFSIDE"
    SUBSTITUTION = "SUBSTITUTION"
    VAR_CHECK = "VAR_CHECK"
    KICK_OFF = "KICK_OFF"
    FULL_TIME = "FULL_TIME"
    OTHER_EVENT = "OTHER_EVENT"
    NONE = "NONE"


@dataclass
class VideoChunk:
    window_index: int
    window_start_sec: float
    window_end_sec: float
    overlap_sec: float
    frames: torch.Tensor
    relative_timestamps: list[float]
    effective_fps: float

    @property
    def duration_sec(self) -> float:
        return round(max(0.0, self.window_end_sec - self.window_start_sec), 3)

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])


class MatchTimelineUpdate(BaseModel):
    video_timestamp_sec: float
    match_minute: int
    stoppage_minute: int = 0
    update_type: Literal[
        "GOAL",
        "YELLOW_CARD",
        "RED_CARD",
        "SUBSTITUTION",
        "HALF_START",
        "SCORE_SYNC",
        "NOTE",
    ]
    team: str = ""
    player: str = ""
    player_out: str = ""
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    description: str = ""


class LiveMatchState(BaseModel):
    competition: str = "Giao hữu Quốc tế"
    stadium: str = ""
    home_team: str = "Home Team"
    away_team: str = "Away Team"
    home_coach: str = ""
    away_coach: str = ""
    home_formation: str = ""
    away_formation: str = ""
    home_score: int = 0
    away_score: int = 0
    half: str = "Hiệp 1"
    match_minute: int = 1
    stoppage_minute: int = 0
    home_lineup: list[str] = Field(default_factory=list)
    away_lineup: list[str] = Field(default_factory=list)
    cards_issued: list[str] = Field(default_factory=list)
    recent_substitutions: list[str] = Field(default_factory=list)
    recent_goals: list[str] = Field(default_factory=list)

    def format_clock(self) -> str:
        if self.stoppage_minute > 0:
            return f"Phút {self.match_minute}+{self.stoppage_minute} ({self.half})"
        return f"Phút {self.match_minute} ({self.half})"


class KnowledgeItem(BaseModel):
    id: str
    category: str = "general"
    content: str
    entities: list[str] = Field(default_factory=list)
    priority: float = 1.0
    used_count: int = 0
    last_used_window: Optional[int] = None


class CommentaryHistoryEntry(BaseModel):
    window_index: int
    window_abs_start: float
    window_abs_end: float
    abs_event_start: float
    abs_event_end: float
    match_clock: str
    mode: CommentaryMode
    event_type: str
    commentary: str


class StructuredCommentaryOutput(BaseModel):
    mode: CommentaryMode
    event_type: str = "NONE"
    start_time: float = 0.0
    end_time: float = 0.0
    commentary: str

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, v: Any) -> str:
        raw = str(v).strip().upper().replace("-", "_").replace(" ", "_")
        if raw in {"EVENT", "HAS_EVENT", "PLAY_BY_PLAY"}:
            return CommentaryMode.EVENT.value
        return CommentaryMode.NO_EVENT.value

    @field_validator("event_type", mode="before")
    @classmethod
    def _normalize_event_type(cls, v: Any) -> str:
        if v is None:
            return SoccerEventType.NONE.value
        cleaned = str(v).strip().upper().replace("-", "_").replace(" ", "_")
        return cleaned if cleaned else SoccerEventType.NONE.value

    @model_validator(mode="after")
    def _enforce_mode_consistency(self) -> "StructuredCommentaryOutput":
        if self.mode == CommentaryMode.NO_EVENT:
            self.event_type = SoccerEventType.NONE.value
        elif self.event_type == SoccerEventType.NONE.value:
            self.event_type = SoccerEventType.OTHER_EVENT.value

        self.start_time = round(max(0.0, float(self.start_time)), 2)
        self.end_time = round(max(self.start_time, float(self.end_time)), 2)
        self.commentary = self.commentary.strip()
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "event_type": self.event_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "commentary": self.commentary,
        }


class TTSProsodyHint(BaseModel):
    voice_style: Literal[
        "dramatic_peak",
        "excited_fast",
        "tense_focused",
        "warm_storytelling",
    ]
    speaking_rate: float = 1.0
    pitch_shift_st: float = 0.0
    energy_level: Literal["high", "medium", "calm"] = "calm"


class WindowPipelineResult(BaseModel):
    window_index: int
    window_abs_start: float
    window_abs_end: float
    match_clock: str
    structured_output: StructuredCommentaryOutput
    abs_start_time: float
    abs_end_time: float
    tts_prosody: TTSProsodyHint
    used_knowledge_id: Optional[str] = None
    is_overlap_duplicate: bool = False
    raw_model_response: str = ""
