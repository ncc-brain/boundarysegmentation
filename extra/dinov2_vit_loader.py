"""
Utilities to initialize and load DINOv2 ViT backbones from checkpoints.

This module provides minimal helpers for model initialization and checkpoint loading only.
"""

from typing import Dict, Optional, Tuple

import torch


def _lazy_import_timm():
    try:
        import timm  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "timm is required to initialize DINOv2 ViT models. Install via `pip install timm`."
        ) from exc
    return timm


def _torch_load_with_safe_globals(checkpoint_path: str, map_location: Optional[str] = "cpu") -> Dict:
    """Load a checkpoint allowing OmegaConf globals under PyTorch 2.6+ safe load."""
    dict_cfg = list_cfg = None
    try:
        from omegaconf import DictConfig as _DictConfig, ListConfig as _ListConfig  # type: ignore
        dict_cfg, list_cfg = _DictConfig, _ListConfig
    except Exception:
        pass

    def _try_load(weights_only_value: Optional[bool]):
        try:
            if weights_only_value is None:
                return torch.load(checkpoint_path, map_location=map_location)
            return torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only_value)  # type: ignore[call-arg]
        except TypeError:
            return torch.load(checkpoint_path, map_location=map_location)

    try:
        if dict_cfg is not None and hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([dict_cfg, list_cfg]):  # type: ignore[attr-defined]
                return _try_load(True)
        if dict_cfg is not None and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([dict_cfg, list_cfg])  # type: ignore[attr-defined]
            return _try_load(True)
        return _try_load(True)
    except Exception:
        return _try_load(False)


def init_dinov2_vit(
    *,
    model_name: str = "vit_large_patch14_224",
    global_pool: Optional[str] = None,
    device: Optional[torch.device] = None,
    img_size: Optional[int] = 224,
) -> torch.nn.Module:
    """
    Initialize a timm ViT backbone suitable for loading DINOv2 weights.

    Defaults to a ViT-L/14. Adjust `model_name` to match your checkpoint (B/L/giant etc.).
    """
    timm = _lazy_import_timm()
    create_kwargs: Dict[str, object] = {
        "pretrained": False,
        "num_classes": 0,
    }
    if img_size is not None:
        create_kwargs["img_size"] = img_size
    if global_pool is not None:
        create_kwargs["global_pool"] = global_pool
    try:
        model = timm.create_model(model_name, **create_kwargs)  # type: ignore[arg-type]
    except TypeError:
        create_kwargs.pop("global_pool", None)
        model = timm.create_model(model_name, **create_kwargs)  # type: ignore[arg-type]
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def _extract_student_backbone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    DINOv2 training often stores weights under e.g. `model['student.backbone.*']`.
    Extract only the student backbone parameters and strip the prefix.
    """
    prefixes = (
        "student.backbone.",
        "backbone.",
        "module.backbone.",
    )
    filtered: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                filtered[new_key] = tensor
                break
    return filtered if filtered else state_dict


def load_dinov2_vit_backbone_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    map_location: Optional[str] = "cpu",
    strict: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Load DINOv2 ViT backbone weights into an initialized timm model.

    Supports checkpoints that store weights under keys like `model['student.backbone.*']`.
    """
    obj = _torch_load_with_safe_globals(checkpoint_path, map_location=map_location)
    if not isinstance(obj, dict):
        raise RuntimeError("Checkpoint is not a dict; unsupported format for this loader.")

    # Common containers in DINOv2 exports
    raw_state = obj.get("state_dict") or obj.get("model") or obj
    if isinstance(raw_state, dict) and any(k.startswith("student.backbone.") for k in raw_state.keys()):
        backbone_state = _extract_student_backbone_state_dict(raw_state)
    elif isinstance(raw_state, dict):
        backbone_state = _extract_student_backbone_state_dict(raw_state)
    else:
        raise RuntimeError("Unsupported checkpoint structure for DINOv2 loader.")

    filtered_state: Dict[str, torch.Tensor] = {}
    for k, v in backbone_state.items():
        if isinstance(v, torch.Tensor):
            filtered_state[k] = v

    load_result = model.load_state_dict(filtered_state, strict=strict)
    return tuple(load_result.missing_keys), tuple(load_result.unexpected_keys)


def create_dinov2_vit_from_checkpoint(
    checkpoint_path: str,
    *,
    model_name: str = "vit_large_patch14_224",
    global_pool: Optional[str] = None,
    device: Optional[torch.device] = None,
    map_location: Optional[str] = "cpu",
    strict: bool = False,
    img_size: Optional[int] = 224,
) -> Tuple[torch.nn.Module, Tuple[str, ...], Tuple[str, ...]]:
    """
    Convenience helper to initialize a ViT and load DINOv2 weights from checkpoint.
    """
    model = init_dinov2_vit(model_name=model_name, global_pool=global_pool, device=device, img_size=img_size)
    missing, unexpected = load_dinov2_vit_backbone_from_checkpoint(
        model,
        checkpoint_path,
        map_location=map_location,
        strict=strict,
    )
    return model, missing, unexpected


