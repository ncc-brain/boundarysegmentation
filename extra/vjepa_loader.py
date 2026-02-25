"""
Utilities to initialize and load V-JEPA (JEPA2) models from Hugging Face.

This provides minimal helpers to obtain the HF model and processor for feature extraction.
"""

from typing import Tuple

import torch


def init_vjepa(
    model_id: str = "facebook/vjepa2-vitl-fpc64-256",
    device: torch.device | None = None,
):
    """
    Initialize a V-JEPA model and its processor from Hugging Face.

    Args:
        model_id: HF repo id for V-JEPA (defaults to a recent JEPA2 variant).
        device: Optional torch.device to move the model to.

    Returns:
        (model, processor)
    """
    from transformers import AutoModel, AutoVideoProcessor

    model = AutoModel.from_pretrained(model_id)
    if device is not None:
        model = model.to(device)
    model.eval()
    processor = AutoVideoProcessor.from_pretrained(model_id)
    return model, processor


def get_vjepa_video_embeddings(model, processor, video_tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a video tensor (T x C x H x W) or batch, process and return embeddings.
    """
    if not isinstance(video_tensor, torch.Tensor):
        raise TypeError("video_tensor must be a torch.Tensor")
    inputs = processor(video_tensor, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        if hasattr(model, "get_vision_features"):
            return model.get_vision_features(**inputs)
        outputs = model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        raise RuntimeError("V-JEPA model did not produce recognizable vision features")



