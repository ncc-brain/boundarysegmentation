"""
Lightweight local fallback for Qwen VL utility used by qwen.py.

Provides process_vision_info(messages) that extracts image/video inputs
from chat-style message payloads and returns (images, videos) suitable
for passing to the Qwen AutoProcessor.
"""

from __future__ import annotations

import base64
import io
from typing import List, Tuple, Any, Dict

from PIL import Image


def _decode_data_url_to_pil(data_url: str) -> Image.Image:
    """
    Decode a data URL such as "data:image/jpeg;base64,<...>" to a PIL image.
    """
    try:
        header, b64data = data_url.split(",", 1)
        image_bytes = base64.b64decode(b64data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        # As a fallback, try to open directly if it is a path-like string
        return Image.open(data_url).convert("RGB")


def process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[List[Any], List[Any]]:
    """
    Extract image and video inputs from chat-format messages and return per-sample lists.

    For a batch of N messages, returns:
      - image_inputs: List of length N where each item is a List[Image.Image]
      - video_inputs: List of length N where each item is a List[Any]
    """
    batch_image_inputs: List[List[Any]] = []
    batch_video_inputs: List[List[Any]] = []

    for message in messages or []:
        per_message_images: List[Any] = []
        per_message_videos: List[Any] = []
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "image":
                    image_val = item.get("image")
                    if image_val is None:
                        continue
                    if isinstance(image_val, Image.Image):
                        per_message_images.append(image_val)
                    elif isinstance(image_val, str) and image_val.startswith("data:image"):
                        try:
                            pil_img = _decode_data_url_to_pil(image_val)
                            per_message_images.append(pil_img)
                        except Exception:
                            pass
                    elif isinstance(image_val, str):
                        try:
                            pil_img = Image.open(image_val).convert("RGB")
                            per_message_images.append(pil_img)
                        except Exception:
                            pass
                elif item_type == "video":
                    video_val = item.get("video")
                    if video_val is not None:
                        per_message_videos.append(video_val)
        batch_image_inputs.append(per_message_images)
        batch_video_inputs.append(per_message_videos)

    return batch_image_inputs, batch_video_inputs


