from __future__ import annotations

import importlib.util
import inspect
import logging
from typing import Any, Optional

import torch

from .schemas import VideoChunk

logger = logging.getLogger(__name__)


class UnifiedVideoLLMEngine:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        adapter_path: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 220,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.08,
        min_pixels: int = 224 * 224,
        max_pixels: int = 360 * 640,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.torch_dtype_str = torch_dtype
        self.attn_implementation = attn_implementation
        self.device_map = device_map

        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.do_sample = bool(do_sample)
        self.repetition_penalty = float(repetition_penalty)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)

        self.model: Any = None
        self.processor: Any = None
        self._supports_do_sample_frames: bool = False
        self._accepts_second_per_grid_ts: bool = False
        self._temporal_patch_size: int = 2

    @staticmethod
    def _resolve_dtype(dtype_name: str) -> torch.dtype | str:
        if dtype_name == "auto":
            return "auto"
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_name.lower() not in mapping:
            raise ValueError(f"torch_dtype không hợp lệ: {dtype_name}")
        resolved = mapping[dtype_name.lower()]
        if resolved == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            return torch.float16
        return resolved

    @staticmethod
    def _resolve_attn_impl(attn_pref: str) -> str:
        if attn_pref != "auto":
            return attn_pref
        if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        return "sdpa"

    def load_model(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        resolved_dtype = self._resolve_dtype(self.torch_dtype_str)
        resolved_attn = self._resolve_attn_impl(self.attn_implementation)

        logger.info(
            "Đang nạp Unified Video-LLM: %s | dtype=%s | attn=%s | device_map=%s",
            self.model_name_or_path,
            resolved_dtype,
            resolved_attn,
            self.device_map,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        pretrained_sig = inspect.signature(Qwen2_5_VLForConditionalGeneration.from_pretrained)
        dtype_kwarg = "dtype" if "dtype" in pretrained_sig.parameters else "torch_dtype"
        model_load_kwargs: dict[str, Any] = {
            dtype_kwarg: resolved_dtype,
            "attn_implementation": resolved_attn,
            "device_map": self.device_map,
        }

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            **model_load_kwargs,
        )

        if self.adapter_path:
            from peft import PeftModel

            logger.info("Đang gắn LoRA Adapter chuyên biệt bóng đá từ: %s", self.adapter_path)
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()

        if hasattr(self.processor, "video_processor") and self.processor.video_processor is not None:
            vp = self.processor.video_processor
            vp_sig = inspect.signature(vp.__call__)
            self._supports_do_sample_frames = (
                hasattr(vp, "do_sample_frames") or "do_sample_frames" in vp_sig.parameters
            )

        fwd_sig = inspect.signature(self.model.forward)
        self._accepts_second_per_grid_ts = "second_per_grid_ts" in fwd_sig.parameters

        vision_cfg = getattr(self.model.config, "vision_config", None)
        if vision_cfg is not None and hasattr(vision_cfg, "temporal_patch_size"):
            self._temporal_patch_size = int(vision_cfg.temporal_patch_size)

    @property
    def input_device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("Mô hình chưa được nạp. Hãy gọi `load_model()` trước.")
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def generate_single_pass(
        self,
        chunk: VideoChunk,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if self.model is None or self.processor is None:
            self.load_model()

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        chat_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        has_native_video_processor = (
            hasattr(self.processor, "video_processor")
            and self.processor.video_processor is not None
        )
        video_tensor_for_proc = (
            chunk.frames if has_native_video_processor else chunk.frames.cpu()
        )

        processor_kwargs: dict[str, Any] = {
            "text": [chat_prompt],
            "videos": [video_tensor_for_proc],
            "padding": True,
            "return_tensors": "pt",
        }
        if self._supports_do_sample_frames:
            processor_kwargs["do_sample_frames"] = False
        else:
            processor_kwargs["fps"] = float(chunk.effective_fps)

        model_inputs = self.processor(**processor_kwargs)

        if self._accepts_second_per_grid_ts and chunk.effective_fps > 0:
            num_videos = (
                int(model_inputs["video_grid_thw"].shape[0])
                if "video_grid_thw" in model_inputs
                else 1
            )
            sec_per_grid = float(self._temporal_patch_size) / float(chunk.effective_fps)
            model_inputs["second_per_grid_ts"] = torch.full(
                (num_videos,),
                sec_per_grid,
                dtype=torch.float32,
            )

        target_dev = self.input_device
        model_inputs = {
            k: (v.to(target_dev, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in model_inputs.items()
        }

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample and self.temperature > 0.0,
            "repetition_penalty": self.repetition_penalty,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        prompt_len = model_inputs["input_ids"].shape[1]
        generated_ids = self.model.generate(**model_inputs, **gen_kwargs)
        new_token_ids = generated_ids[:, prompt_len:]

        decoded_text = self.processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        del model_inputs, generated_ids, new_token_ids
        return decoded_text
