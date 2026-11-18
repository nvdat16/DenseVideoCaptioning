from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import re
from typing import Any, Optional

from .schemas import (
    CommentaryHistoryEntry,
    CommentaryMode,
    KnowledgeItem,
    LiveMatchState,
    MatchTimelineUpdate,
    VideoChunk,
)


class LiveMatchStateTracker:
    def __init__(self, metadata: dict[str, Any]) -> None:
        match_info = metadata.get("match_info", {})
        lineups = metadata.get("lineups", {})

        self.kickoff_offset_sec: float = float(match_info.get("kickoff_video_offset_sec", 0.0))
        self.second_half_offset_sec: Optional[float] = (
            float(match_info["second_half_video_offset_sec"])
            if match_info.get("second_half_video_offset_sec") is not None
            else None
        )

        self.state = LiveMatchState(
            competition=match_info.get("competition", "Giải bóng đá chuyên nghiệp"),
            stadium=match_info.get("stadium", ""),
            home_team=match_info.get("home_team", "Đội chủ nhà"),
            away_team=match_info.get("away_team", "Đội khách"),
            home_coach=match_info.get("home_coach", ""),
            away_coach=match_info.get("away_coach", ""),
            home_formation=match_info.get("home_formation", ""),
            away_formation=match_info.get("away_formation", ""),
            home_score=int(match_info.get("initial_home_score", 0)),
            away_score=int(match_info.get("initial_away_score", 0)),
            home_lineup=list(lineups.get("home_starting_xi", [])),
            away_lineup=list(lineups.get("away_starting_xi", [])),
        )

        raw_updates = metadata.get("timeline_updates", [])
        self._pending_updates: list[MatchTimelineUpdate] = sorted(
            [MatchTimelineUpdate.model_validate(u) for u in raw_updates],
            key=lambda x: x.video_timestamp_sec,
        )
        self._applied_idx: int = 0

    def push_realtime_update(self, update: MatchTimelineUpdate) -> None:
        self._pending_updates.append(update)
        self._pending_updates.sort(key=lambda x: x.video_timestamp_sec)

    def sync_to_timestamp(self, video_abs_sec: float) -> LiveMatchState:
        self._update_match_clock(video_abs_sec)

        while (
            self._applied_idx < len(self._pending_updates)
            and self._pending_updates[self._applied_idx].video_timestamp_sec <= video_abs_sec
        ):
            upd = self._pending_updates[self._applied_idx]
            self._apply_single_update(upd)
            self._applied_idx += 1

        return self.state

    def _update_match_clock(self, video_abs_sec: float) -> None:
        if (
            self.second_half_offset_sec is not None
            and video_abs_sec >= self.second_half_offset_sec
        ):
            self.state.half = "Hiệp 2"
            elapsed_h2 = max(0.0, video_abs_sec - self.second_half_offset_sec)
            raw_min = 45 + int(elapsed_h2 // 60) + 1
            if raw_min > 90:
                self.state.match_minute = 90
                self.state.stoppage_minute = raw_min - 90
            else:
                self.state.match_minute = raw_min
                self.state.stoppage_minute = 0
        else:
            self.state.half = "Hiệp 1"
            elapsed_h1 = max(0.0, video_abs_sec - self.kickoff_offset_sec)
            raw_min = int(elapsed_h1 // 60) + 1
            if raw_min > 45:
                self.state.match_minute = 45
                self.state.stoppage_minute = raw_min - 45
            else:
                self.state.match_minute = max(1, raw_min)
                self.state.stoppage_minute = 0

    def _apply_single_update(self, upd: MatchTimelineUpdate) -> None:
        min_label = (
            f"{upd.match_minute}+{upd.stoppage_minute}'"
            if upd.stoppage_minute > 0
            else f"{upd.match_minute}'"
        )

        if upd.home_score is not None:
            self.state.home_score = upd.home_score
        if upd.away_score is not None:
            self.state.away_score = upd.away_score

        if upd.update_type == "GOAL":
            scorer_info = f"{min_label}: {upd.player} ({upd.team})"
            self.state.recent_goals.append(scorer_info)
        elif upd.update_type in {"YELLOW_CARD", "RED_CARD"}:
            card_vn = "Thẻ vàng" if upd.update_type == "YELLOW_CARD" else "Thẻ đỏ"
            self.state.cards_issued.append(f"{card_vn} - {upd.player} ({upd.team}, {min_label})")
        elif upd.update_type == "SUBSTITUTION":
            sub_str = f"{min_label} ({upd.team}): {upd.player} vào thay {upd.player_out}"
            self.state.recent_substitutions.append(sub_str)
            self._swap_lineup_player(upd.team, upd.player_out, upd.player)

    def _swap_lineup_player(self, team: str, player_out: str, player_in: str) -> None:
        if not player_out or not player_in:
            return
        target_list = (
            self.state.home_lineup
            if team.lower() == self.state.home_team.lower()
            else self.state.away_lineup
        )
        for i, entry in enumerate(target_list):
            if player_out.lower() in entry.lower():
                target_list[i] = f"{player_in} (thay {player_out})"
                return
        target_list.append(f"{player_in} (dự bị vào sân)")


class ExternalKnowledgeBase:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items: list[KnowledgeItem] = [KnowledgeItem.model_validate(it) for it in items]
        self._last_used_category: Optional[str] = None

    @staticmethod
    def _tokenize_vi(text: str) -> set[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return {tok for tok in cleaned.split() if len(tok) > 2}

    def retrieve_candidates(
        self,
        match_state: LiveMatchState,
        rolling_history: list[CommentaryHistoryEntry],
        top_k: int = 4,
        category_diversity: bool = True,
    ) -> list[KnowledgeItem]:
        if not self.items:
            return []

        unused_pool = [it for it in self.items if it.used_count == 0]
        if not unused_pool:
            min_used = min(it.used_count for it in self.items)
            unused_pool = [it for it in self.items if it.used_count == min_used]

        active_context_blobs: list[str] = []
        active_context_blobs.extend(match_state.recent_substitutions[-2:])
        active_context_blobs.extend(match_state.cards_issued[-2:])
        active_context_blobs.extend(h.commentary for h in rolling_history[-2:])
        hot_text = " ".join(active_context_blobs).lower()

        scored: list[tuple[float, KnowledgeItem]] = []
        for item in unused_pool:
            score = float(item.priority)
            for ent in item.entities:
                if ent.lower() in hot_text:
                    score += 0.35
            if category_diversity and self._last_used_category and item.category == self._last_used_category:
                score -= 0.25
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        if not category_diversity:
            return [it for _, it in scored[:top_k]]

        selected: list[KnowledgeItem] = []
        seen_categories: set[str] = set()
        for _, item in scored:
            if item.category not in seen_categories:
                selected.append(item)
                seen_categories.add(item.category)
                if len(selected) >= top_k:
                    return selected

        for _, item in scored:
            if item not in selected:
                selected.append(item)
                if len(selected) >= top_k:
                    break

        return selected

    def mark_used_from_commentary(
        self,
        candidates: list[KnowledgeItem],
        commentary_text: str,
        window_index: int,
    ) -> Optional[str]:
        if not candidates or not commentary_text.strip():
            return None

        comm_tokens = self._tokenize_vi(commentary_text)
        best_item: Optional[KnowledgeItem] = None
        best_overlap = 0.0

        for cand in candidates:
            cand_tokens = self._tokenize_vi(cand.content)
            if not cand_tokens:
                continue
            overlap_ratio = len(comm_tokens & cand_tokens) / float(len(cand_tokens))
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_item = cand

        if best_item is not None and best_overlap >= 0.18:
            best_item.used_count += 1
            best_item.last_used_window = window_index
            self._last_used_category = best_item.category
            return best_item.id

        return None


class RollingCommentaryMemory:
    def __init__(self, max_history: int = 4) -> None:
        self.max_history = max(1, int(max_history))
        self._buffer: deque[CommentaryHistoryEntry] = deque(maxlen=self.max_history)

    def append(self, entry: CommentaryHistoryEntry) -> None:
        self._buffer.append(entry)

    def entries(self) -> list[CommentaryHistoryEntry]:
        return list(self._buffer)

    def last_entry(self) -> Optional[CommentaryHistoryEntry]:
        return self._buffer[-1] if self._buffer else None

    def build_overlap_context_note(self, chunk: VideoChunk) -> str:
        if chunk.overlap_sec <= 0.0 or not self._buffer:
            return "Đây là cửa sổ mở đầu luồng xử lý (không có vùng gối đầu)."

        prev = self._buffer[-1]
        if (
            prev.mode == CommentaryMode.EVENT
            and prev.abs_event_end > chunk.window_start_sec - 0.1
        ):
            rel_covered_end = min(
                chunk.overlap_sec,
                round(max(0.0, prev.abs_event_end - chunk.window_start_sec), 2),
            )
            return (
                f"LƯU Ý GỐI ĐẦU ({chunk.overlap_sec:.1f}s đầu): Khoảng [0.0s - {rel_covered_end:.1f}s] "
                f"trùng với phần đuôi sự kiện `{prev.event_type}` vừa được bình luận ở cửa sổ trước. "
                f"KHÔNG lặp lại sự kiện đó trừ khi sau mốc {rel_covered_end:.1f}s xuất hiện tình huống mới tiếp nối "
                f"(như đá bồi, bàn thắng, phạt góc)."
            )

        return (
            f"Khoảng [0.0s - {chunk.overlap_sec:.1f}s] là vùng gối đầu nối tiếp từ cửa sổ trước "
            f"để đảm bảo không cắt ngang chuyển động."
        )


class ContextAssembler:
    SYSTEM_PROMPT_VI = """Bạn là bình luận viên bóng đá chuyên nghiệp trên sóng truyền hình trực tiếp, vận hành theo kiến trúc End-to-End 2 chế độ (Dual-Mode Soccer Commentator).

NHIỆM VỤ CỐT LÕI (IMPLICIT ROUTING):
Quan sát trực tiếp chuỗi khung hình video kèm tọa độ thời gian tương đối và tự động định tuyến sang 1 trong 2 chế độ:

1. Chế độ "EVENT" (Bình luận diễn biến trực tiếp):
   - Kích hoạt KHI VÀ CHỈ KHI trong video xuất hiện tình huống bóng đá đáng chú ý: bàn thắng (GOAL), sút trúng đích (SHOT_ON_TARGET), sút chệch cột/vọt xà (SHOT_OFF_TARGET), thủ môn cứu thua (SAVE), pha phối hợp/tấn công nguy hiểm sát vòng cấm (DANGEROUS_ATTACK), phạt góc (CORNER), đá phạt (FREE_KICK), phạt đền (PENALTY), phạm lỗi (FOUL), thẻ vàng/đỏ (YELLOW_CARD / RED_CARD), việt vị (OFFSIDE), thay người (SUBSTITUTION), kiểm tra VAR (VAR_CHECK).
   - Yêu cầu: Bám sát chuẩn xác hành động thị giác trên sân, giàu cảm xúc, nhịp độ nhanh, xác định chính xác khoảng thời gian tương đối `start_time` và `end_time` (tính bằng giây trong đoạn video hiện tại) nơi tình huống diễn ra.

2. Chế độ "NO_EVENT" (Bình luận thông tin bên lề & giữ nhịp):
   - Kích hoạt khi trên sân chỉ là các pha chuyền bóng qua lại giảm nhịp độ ở giữa sân, bóng chết chờ phát bóng, cầu thủ đi bộ, hoặc máy quay lia lên khán đài / băng ghế huấn luyện.
   - Yêu cầu: Mở đầu ngắn gọn 1 vế mô tả nhịp độ hiện tại trên sân để nối mượt với lịch sử bình luận, sau đó lồng ghép TỰ NHIÊN đúng 1 thông tin thú vị từ phần [KHO TIN NGOÀI LỀ CHƯA DÙNG]. Tuyệt đối KHÔNG bịa số liệu thống kê ngoài danh sách cung cấp và KHÔNG lặp lại nội dung đã nói trong [LỊCH SỬ BÌNH LUẬN GẦN NHẤT]. Đặt `event_type = "NONE"`, `start_time = 0.0`, `end_time` bằng tổng thời lượng đoạn video.

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Chỉ trả về DUY NHẤT 1 đối tượng JSON hợp lệ (không thêm markdown hay giải thích bên ngoài) theo cấu trúc cố định:
{
  "mode": "EVENT" | "NO_EVENT",
  "event_type": "GOAL | SHOT_ON_TARGET | SHOT_OFF_TARGET | SAVE | DANGEROUS_ATTACK | CORNER | FREE_KICK | PENALTY | FOUL | YELLOW_CARD | RED_CARD | OFFSIDE | SUBSTITUTION | VAR_CHECK | NONE",
  "start_time": <float_giây_bắt_đầu_tương_đối>,
  "end_time": <float_giây_kết_thúc_tương_đối>,
  "commentary": "<Lời bình tiếng Việt tự nhiên, mạch lạc, đúng ngữ cảnh>"
}"""

    def __init__(
        self,
        metadata: dict[str, Any],
        rolling_history_size: int = 4,
        kb_top_k_candidates: int = 4,
        kb_category_diversity: bool = True,
    ) -> None:
        self.match_tracker = LiveMatchStateTracker(metadata)
        self.knowledge_base = ExternalKnowledgeBase(metadata.get("external_knowledge_base", []))
        self.rolling_memory = RollingCommentaryMemory(max_history=rolling_history_size)
        self.kb_top_k_candidates = int(kb_top_k_candidates)
        self.kb_category_diversity = bool(kb_category_diversity)

    @classmethod
    def from_metadata_file(
        cls,
        metadata_path: str | Path,
        rolling_history_size: int = 4,
        kb_top_k_candidates: int = 4,
        kb_category_diversity: bool = True,
    ) -> "ContextAssembler":
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return cls(
            metadata=metadata,
            rolling_history_size=rolling_history_size,
            kb_top_k_candidates=kb_top_k_candidates,
            kb_category_diversity=kb_category_diversity,
        )

    def assemble_window_context(
        self, chunk: VideoChunk
    ) -> tuple[str, str, LiveMatchState, list[KnowledgeItem]]:
        state = self.match_tracker.sync_to_timestamp(chunk.window_end_sec).model_copy(deep=True)
        history_entries = self.rolling_memory.entries()

        kb_candidates = self.knowledge_base.retrieve_candidates(
            match_state=state,
            rolling_history=history_entries,
            top_k=self.kb_top_k_candidates,
            category_diversity=self.kb_category_diversity,
        )

        match_state_block = self._format_match_state_block(state)
        kb_block = self._format_kb_block(kb_candidates)
        history_block = self._format_history_block(history_entries)
        overlap_note = self.rolling_memory.build_overlap_context_note(chunk)

        user_prompt = (
            f"[THÔNG TIN CỬA SỔ VIDEO HIỆN TẠI]\n"
            f"- Cửa sổ thứ: #{chunk.window_index} | Thời lượng cửa sổ: {chunk.duration_sec:.1f} giây "
            f"(trục thời gian tương đối: 0.0s -> {chunk.duration_sec:.1f}s, {chunk.num_frames} frames @ {chunk.effective_fps:.1f} FPS).\n"
            f"- {overlap_note}\n\n"
            f"{match_state_block}\n\n"
            f"{kb_block}\n\n"
            f"{history_block}\n\n"
            f"Hãy quan sát kỹ chuỗi khung hình trên và xuất ra đúng 1 JSON duy nhất cho cửa sổ [0.0s, {chunk.duration_sec:.1f}s]."
        )

        return self.SYSTEM_PROMPT_VI, user_prompt, state, kb_candidates

    @staticmethod
    def _format_match_state_block(state: LiveMatchState) -> str:
        lines = [
            "[1. LIVE MATCH STATE (TRẠNG THÁI TRẬN ĐẤU HIỆN TẠI)]",
            f"- Giải đấu: {state.competition}" + (f" | Sân: {state.stadium}" if state.stadium else ""),
            f"- Thời gian & Tỷ số: {state.format_clock()} | {state.home_team} {state.home_score} - {state.away_score} {state.away_team}",
        ]
        if state.home_coach or state.away_coach:
            lines.append(
                f"- HLV trưởng: {state.home_team}: {state.home_coach or 'N/A'} | "
                f"{state.away_team}: {state.away_coach or 'N/A'}"
            )
        if state.home_lineup:
            lines.append(f"- Đội hình trên sân ({state.home_team}): {', '.join(state.home_lineup)}")
        if state.away_lineup:
            lines.append(f"- Đội hình trên sân ({state.away_team}): {', '.join(state.away_lineup)}")
        if state.recent_goals:
            lines.append(f"- Ghi bàn trước đó: {'; '.join(state.recent_goals)}")
        if state.cards_issued:
            lines.append(f"- Thẻ phạt đã rút: {'; '.join(state.cards_issued)}")
        if state.recent_substitutions:
            lines.append(f"- Thay người gần đây: {'; '.join(state.recent_substitutions[-3:])}")
        return "\n".join(lines)

    @staticmethod
    def _format_kb_block(candidates: list[KnowledgeItem]) -> str:
        lines = ["[2. EXTERNAL KNOWLEDGE BASE (KHO TIN NGOÀI LỀ CHƯA DÙNG - CHỈ DÙNG KHI NO_EVENT)]"]
        if not candidates:
            lines.append("- (Không còn thông tin bên lề mới; nếu NO_EVENT hãy nhận xét ngắn về thế trận trên sân.)")
            return "\n".join(lines)

        for idx, item in enumerate(candidates, start=1):
            lines.append(f"- ({idx}) [{item.category}] {item.content}")
        return "\n".join(lines)

    @staticmethod
    def _format_history_block(history: list[CommentaryHistoryEntry]) -> str:
        lines = ["[3. ROLLING COMMENTARY HISTORY (LỊCH SỬ CÁC CÂU BÌNH LUẬN GẦN NHẤT)]"]
        if not history:
            lines.append("- (Chưa có câu bình luận nào trước đó - đây là đoạn mở đầu.)")
            return "\n".join(lines)

        for entry in history:
            lines.append(
                f"- [{entry.match_clock} | {entry.mode.value}:{entry.event_type}] \"{entry.commentary}\""
            )
        return "\n".join(lines)
