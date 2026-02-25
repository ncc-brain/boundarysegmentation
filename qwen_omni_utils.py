"""
Utility helpers for Qwen3-Omni multimodal inputs.

Provides process_mm_info(messages, use_audio_in_video) that extracts
per-message audio, image, and video payloads from chat-style messages
and returns lists consumable by the Qwen3-Omni processor.
"""
import base64
import io
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
from PIL import Image


def _decode_data_url_to_bytes(data_url: str) -> bytes:
    header, b64data = data_url.split(",", 1)
    return base64.b64decode(b64data)


def _load_image(image_val: Any) -> Image.Image | None:
    if isinstance(image_val, Image.Image):
        return image_val.convert("RGB")
    if isinstance(image_val, str):
        if image_val.startswith("data:image"):
            try:
                image_bytes = _decode_data_url_to_bytes(image_val)
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                return None
        try:
            return Image.open(image_val).convert("RGB")
        except Exception:
            return None
    return None


def _load_audio(audio_val: Any) -> np.ndarray | None:
    if isinstance(audio_val, tuple) and len(audio_val) == 2:
        audio_arr, sample_rate = audio_val
        if isinstance(audio_arr, np.ndarray) and isinstance(sample_rate, int):
            audio_arr = audio_arr.astype(np.float32)
            if audio_arr.ndim == 1:
                return audio_arr
            if audio_arr.ndim == 2:
                return audio_arr.mean(axis=0).astype(np.float32)
            return audio_arr.reshape(-1).astype(np.float32)
    if isinstance(audio_val, str):
        path = audio_val
        if path.startswith("data:audio"):
            try:
                audio_bytes = _decode_data_url_to_bytes(path)
                with sf.SoundFile(io.BytesIO(audio_bytes)) as snd:
                    data = snd.read(dtype="float32")
                    data = np.asarray(data, dtype=np.float32)
                    if data.ndim == 1:
                        return data
                    return data.mean(axis=1).astype(np.float32)
            except Exception:
                return None
        if os.path.exists(path):
            try:
                data, _ = sf.read(path, dtype="float32")
                data = np.asarray(data, dtype=np.float32)
                if data.ndim == 1:
                    return data
                return data.mean(axis=1).astype(np.float32)
            except Exception:
                return None
    return None


def process_mm_info(messages: List[Dict[str, Any]] | None, use_audio_in_video: bool = False) -> Tuple[List[List[np.ndarray]], List[List[Any]], List[List[Any]]]:
    audio_inputs: List[List[np.ndarray]] = []
    image_inputs: List[List[Any]] = []
    video_inputs: List[List[Any]] = []

    for message in messages or []:
        per_message_audio: List[np.ndarray] = []
        per_message_images: List[Any] = []
        per_message_videos: List[Any] = []

        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")

                if item_type == "audio" and not use_audio_in_video:
                    audio_val = item.get("audio")
                    loaded_audio = _load_audio(audio_val)
                    if loaded_audio is not None:
                        per_message_audio.append(loaded_audio)
                elif item_type == "image":
                    image_val = item.get("image")
                    loaded_image = _load_image(image_val)
                    if loaded_image is not None:
                        per_message_images.append(loaded_image)
                elif item_type == "video":
                    video_val = item.get("video")
                    if video_val is not None:
                        per_message_videos.append(video_val)

        audio_inputs.append(per_message_audio)
        image_inputs.append(per_message_images)
        video_inputs.append(per_message_videos)

    return audio_inputs, image_inputs, video_inputs

