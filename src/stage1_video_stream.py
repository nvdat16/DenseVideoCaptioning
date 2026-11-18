from __future__ import annotations

from collections import deque
import importlib.util
import logging
import math
from typing import Iterator, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .schemas import VideoChunk

logger = logging.getLogger(__name__)

QWEN_VL_SPATIAL_FACTOR = 28
QWEN_VL_TEMPORAL_FACTOR = 2


def resolve_tensor_device(device_pref: str = "auto") -> torch.device:
    if device_pref != "auto":
        return torch.device(device_pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def smart_resize_dimensions(
    height: int,
    width: int,
    factor: int = QWEN_VL_SPATIAL_FACTOR,
    min_pixels: int = 224 * 224,
    max_pixels: int = 360 * 640,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"Kích thước khung hình không hợp lệ: {height}x{width}")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return int(h_bar), int(w_bar)


class SlidingWindowVideoStreamer:
    def __init__(
        self,
        window_duration_sec: float = 15.0,
        overlap_duration_sec: float = 2.0,
        sample_fps: float = 2.0,
        decoder_backend: Literal["auto", "decord", "pyav"] = "auto",
        min_pixels: int = 224 * 224,
        max_pixels: int = 360 * 640,
        frame_tensor_device: str = "auto",
    ) -> None:
        if window_duration_sec <= 0:
            raise ValueError("window_duration_sec phải > 0.")
        if overlap_duration_sec < 0 or overlap_duration_sec >= window_duration_sec:
            raise ValueError("overlap_duration_sec phải thuộc khoảng [0, window_duration_sec).")
        if sample_fps <= 0:
            raise ValueError("sample_fps phải > 0.")

        self.window_duration_sec = float(window_duration_sec)
        self.overlap_duration_sec = float(overlap_duration_sec)
        self.stride_sec = self.window_duration_sec - self.overlap_duration_sec
        self.sample_fps = float(sample_fps)
        self.decoder_backend = decoder_backend
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.target_device = resolve_tensor_device(frame_tensor_device)

    @staticmethod
    def _is_live_stream_uri(video_source: str) -> bool:
        lowered = video_source.lower()
        return lowered.startswith(("rtsp://", "rtmp://", "rtp://", "udp://", "http://", "https://"))

    def _select_backend(self, video_source: str) -> Literal["decord", "pyav"]:
        if self._is_live_stream_uri(video_source):
            return "pyav"

        if self.decoder_backend == "decord":
            return "decord"
        if self.decoder_backend == "pyav":
            return "pyav"

        if importlib.util.find_spec("decord") is not None:
            return "decord"
        return "pyav"

    def stream_chunks(
        self,
        video_source: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> Iterator[VideoChunk]:
        backend = self._select_backend(video_source)
        logger.info(
            "Khởi tạo Stage 1 Video Streamer | backend=%s | window=%.1fs | overlap=%.1fs | fps=%.1f | device=%s",
            backend,
            self.window_duration_sec,
            self.overlap_duration_sec,
            self.sample_fps,
            self.target_device,
        )

        if backend == "decord":
            yield from self._stream_with_decord(video_source, start_sec=start_sec, end_sec=end_sec)
        else:
            yield from self._stream_with_pyav(video_source, start_sec=start_sec, end_sec=end_sec)

    def _stream_with_decord(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> Iterator[VideoChunk]:
        import decord

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        native_fps = float(vr.get_avg_fps())
        total_frames = len(vr)
        total_duration_sec = total_frames / native_fps

        effective_end_sec = (
            min(total_duration_sec, float(end_sec)) if end_sec is not None else total_duration_sec
        )
        if start_sec >= effective_end_sec:
            return

        dt = 1.0 / self.sample_fps
        overlap_frame_cache: dict[int, torch.Tensor] = {}

        window_idx = 0
        w_start = float(start_sec)

        while w_start < effective_end_sec:
            w_end = min(w_start + self.window_duration_sec, effective_end_sec)
            actual_duration = w_end - w_start

            if window_idx > 0 and actual_duration <= self.overlap_duration_sec + 0.25:
                break

            num_samples = max(2, int(round(actual_duration * self.sample_fps)))
            rel_timestamps = [round(i * dt, 4) for i in range(num_samples) if i * dt < actual_duration + 1e-6]
            if len(rel_timestamps) < 2:
                rel_timestamps = [0.0, round(max(0.5, actual_duration * 0.5), 4)]

            frame_indices = [
                min(total_frames - 1, max(0, int(round((w_start + t_rel) * native_fps))))
                for t_rel in rel_timestamps
            ]

            missing_indices = sorted({idx for idx in frame_indices if idx not in overlap_frame_cache})
            if missing_indices:
                decoded_np = vr.get_batch(missing_indices).asnumpy()
                decoded_tensor = self._preprocess_frames_to_tensor_raw(decoded_np)
                for local_pos, f_idx in enumerate(missing_indices):
                    overlap_frame_cache[f_idx] = decoded_tensor[local_pos]

            ordered_frames = torch.stack([overlap_frame_cache[idx] for idx in frame_indices], dim=0)
            if ordered_frames.shape[0] % QWEN_VL_TEMPORAL_FACTOR != 0:
                ordered_frames = torch.cat([ordered_frames, ordered_frames[-1:]], dim=0)
                rel_timestamps.append(rel_timestamps[-1])

            next_w_start = w_start + self.stride_sec
            next_start_frame_idx = int(math.floor(next_w_start * native_fps))
            overlap_frame_cache = {
                f_idx: f_tensor
                for f_idx, f_tensor in overlap_frame_cache.items()
                if f_idx >= next_start_frame_idx
            }

            yield VideoChunk(
                window_index=window_idx,
                window_start_sec=round(w_start, 3),
                window_end_sec=round(w_end, 3),
                overlap_sec=0.0 if window_idx == 0 else self.overlap_duration_sec,
                frames=ordered_frames,
                relative_timestamps=rel_timestamps,
                effective_fps=self.sample_fps,
            )

            if w_end >= effective_end_sec:
                break
            w_start = next_w_start
            window_idx += 1

    def _preprocess_frames_to_tensor_raw(self, frames_nhwc: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frames_nhwc).permute(0, 3, 1, 2).contiguous()
        if self.target_device.type != "cpu":
            tensor = tensor.to(self.target_device, non_blocking=True)

        _, _, h, w = tensor.shape
        target_h, target_w = smart_resize_dimensions(
            h,
            w,
            factor=QWEN_VL_SPATIAL_FACTOR,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        if (target_h, target_w) != (h, w):
            tensor = (
                F.interpolate(
                    tensor.float(),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .clamp(0.0, 255.0)
                .to(torch.uint8)
            )
        return tensor

    def _stream_with_pyav(
        self,
        video_source: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> Iterator[VideoChunk]:
        import av

        container = av.open(video_source)
        try:
            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"

            if start_sec > 0.0 and not self._is_live_stream_uri(video_source) and video_stream.time_base:
                seek_pts = int(start_sec / float(video_stream.time_base))
                container.seek(seek_pts, any_frame=False, backward=True, stream=video_stream)

            dt = 1.0 / self.sample_fps
            frames_per_window = max(2, int(round(self.window_duration_sec * self.sample_fps)))
            stride_frames = max(1, int(round(self.stride_sec * self.sample_fps)))

            ring_buffer: deque[tuple[float, torch.Tensor]] = deque()
            next_sample_ts: Optional[float] = start_sec if start_sec > 0.0 else None
            fallback_fps = float(video_stream.average_rate or 25.0)

            window_idx = 0

            for raw_idx, frame in enumerate(container.decode(video_stream)):
                if frame.pts is not None and frame.time_base is not None:
                    pts_sec = float(frame.pts * frame.time_base)
                else:
                    pts_sec = raw_idx / fallback_fps

                if pts_sec < start_sec - 1e-3:
                    continue
                if end_sec is not None and pts_sec > end_sec + 1e-3:
                    break

                if next_sample_ts is None:
                    next_sample_ts = pts_sec

                if pts_sec + (0.25 * dt) >= next_sample_ts:
                    rgb_np = frame.to_ndarray(format="rgb24")[None, ...]
                    chw_tensor = self._preprocess_frames_to_tensor_raw(rgb_np)[0]
                    ring_buffer.append((pts_sec, chw_tensor))
                    while next_sample_ts <= pts_sec + (0.25 * dt):
                        next_sample_ts += dt

                    if len(ring_buffer) >= frames_per_window:
                        window_slice = list(ring_buffer)[:frames_per_window]
                        yield self._build_chunk_from_slice(window_idx, window_slice)
                        window_idx += 1

                        for _ in range(min(stride_frames, len(ring_buffer))):
                            ring_buffer.popleft()

            overlap_frames = max(0, frames_per_window - stride_frames)
            if len(ring_buffer) > max(1, overlap_frames):
                yield self._build_chunk_from_slice(window_idx, list(ring_buffer))
        finally:
            container.close()

    def _build_chunk_from_slice(
        self,
        window_idx: int,
        sampled_slice: list[tuple[float, torch.Tensor]],
    ) -> VideoChunk:
        w_start = sampled_slice[0][0]
        dt = 1.0 / self.sample_fps
        w_end = sampled_slice[-1][0] + dt

        rel_timestamps = [round(max(0.0, ts - w_start), 4) for ts, _ in sampled_slice]
        stacked = torch.stack([t for _, t in sampled_slice], dim=0)

        if stacked.shape[0] % QWEN_VL_TEMPORAL_FACTOR != 0:
            stacked = torch.cat([stacked, stacked[-1:]], dim=0)
            rel_timestamps.append(rel_timestamps[-1])

        return VideoChunk(
            window_index=window_idx,
            window_start_sec=round(w_start, 3),
            window_end_sec=round(w_end, 3),
            overlap_sec=0.0 if window_idx == 0 else self.overlap_duration_sec,
            frames=stacked,
            relative_timestamps=rel_timestamps,
            effective_fps=self.sample_fps,
        )
