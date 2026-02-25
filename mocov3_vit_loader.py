"""
Utilities to initialize and load MoCo v3 ViT backbones from PyTorch Lightning checkpoints.

This module focuses only on model initialization and checkpoint loading for the MoCo v3 ViT base
variant (trained with a ViT-B/16 backbone). It is intentionally minimal and does not perform any
evaluation or feature extraction.
"""

from typing import Dict, Optional, Tuple

import torch


def _torch_load_with_safe_globals(checkpoint_path: str, map_location: Optional[str] = "cpu") -> Dict:
    """Load a checkpoint allowing OmegaConf globals under PyTorch 2.6+ safe load.

    Tries weights_only=True with allowlisted OmegaConf globals, falls back to standard load.
    """
    # Try to allowlist OmegaConf globals when available
    dict_cfg = list_cfg = None
    try:
        from omegaconf import DictConfig as _DictConfig, ListConfig as _ListConfig  # type: ignore
        dict_cfg, list_cfg = _DictConfig, _ListConfig
    except Exception:
        pass

    # Prefer weights_only=True on newer PyTorch
    def _try_load(weights_only_value: Optional[bool]):
        try:
            if weights_only_value is None:
                return torch.load(checkpoint_path, map_location=map_location)
            return torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only_value)  # type: ignore[call-arg]
        except TypeError:
            # weights_only not supported in this torch version
            return torch.load(checkpoint_path, map_location=map_location)

    try:
        if dict_cfg is not None and hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([dict_cfg, list_cfg]):  # type: ignore[attr-defined]
                return _try_load(True)
        if dict_cfg is not None and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([dict_cfg, list_cfg])  # type: ignore[attr-defined]
            return _try_load(True)
        # No allowlist available; try default load (may still work)
        return _try_load(True)
    except Exception:
        # As a last resort, allow full unpickling if user trusts source
        return _try_load(False)


def _lazy_import_timm():
    """Import timm lazily to avoid hard dependency if unused."""
    try:
        import timm  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time environment dependent
        raise RuntimeError(
            "timm is required to initialize MoCo v3 ViT models. Install via `pip install timm`."
        ) from exc
    return timm


def init_mocov3_vit_base(
    *,
    model_name: str = "vit_base_patch16_224",
    global_pool: Optional[str] = "avg",
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Initialize a timm ViT-B/16 backbone suitable for loading MoCo v3 weights.

    Args:
        model_name: timm model name for ViT-B/16. Defaults to "vit_base_patch16_224".
        global_pool: Global pooling strategy. "avg" is common for MoCo v3; set None to use default.
        device: Optional device to place the model on.

    Returns:
        A torch.nn.Module instance of the ViT backbone with no classifier head (num_classes=0).
    """
    timm = _lazy_import_timm()

    # Many timm ViT variants accept `global_pool`. Fallback gracefully if not supported.
    create_kwargs: Dict[str, object] = {
        "pretrained": False,
        "num_classes": 0,  # ensure no classification head
    }
    if global_pool is not None:
        create_kwargs["global_pool"] = global_pool

    try:
        model = timm.create_model(model_name, **create_kwargs)  # type: ignore[arg-type]
    except TypeError:
        # Older timm versions may not support `global_pool`; retry without it
        create_kwargs.pop("global_pool", None)
        model = timm.create_model(model_name, **create_kwargs)  # type: ignore[arg-type]

    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def _extract_backbone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract the ViT backbone weights from a Lightning-style state_dict.

    The input state_dict is expected to contain keys prefixed with one of:
    - "backbone."
    - "model.backbone."
    - "module.backbone."

    Returns a new dict with prefixes stripped to match timm ViT key names.
    """
    prefixes = ("backbone.", "model.backbone.", "module.backbone.")
    filtered: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                filtered[new_key] = tensor
                break
    return filtered if filtered else state_dict


def load_mocov3_vit_backbone_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    map_location: Optional[str] = "cpu",
    strict: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Load MoCo v3 ViT backbone weights into an initialized timm model.

    Args:
        model: An initialized timm ViT model (e.g., from init_mocov3_vit_base) with num_classes=0.
        checkpoint_path: Path to the Lightning .ckpt file.
        map_location: torch.load map_location, defaults to "cpu" for portability.
        strict: Whether to enforce that the keys in state_dict match exactly.

    Returns:
        A tuple of (missing_keys, unexpected_keys) reported by load_state_dict.
    """
    ckpt_obj = _torch_load_with_safe_globals(checkpoint_path, map_location=map_location)
    if not isinstance(ckpt_obj, dict):
        raise RuntimeError("Checkpoint is not a dict; unsupported format for this loader.")

    raw_state = ckpt_obj.get("state_dict", ckpt_obj)
    backbone_state = _extract_backbone_state_dict(raw_state)

    # Filter non-Tensor entries just in case
    filtered_state: Dict[str, torch.Tensor] = {}
    for k, v in backbone_state.items():
        if isinstance(v, torch.Tensor):
            filtered_state[k] = v

    load_result = model.load_state_dict(filtered_state, strict=strict)
    missing = tuple(load_result.missing_keys)
    unexpected = tuple(load_result.unexpected_keys)
    return missing, unexpected


def create_mocov3_vit_from_checkpoint(
    checkpoint_path: str,
    *,
    model_name: str = "vit_base_patch16_224",
    global_pool: Optional[str] = "avg",
    device: Optional[torch.device] = None,
    map_location: Optional[str] = "cpu",
    strict: bool = False,
) -> Tuple[torch.nn.Module, Tuple[str, ...], Tuple[str, ...]]:
    """
    Convenience helper: initialize a ViT-B/16 backbone and load MoCo v3 weights from checkpoint.

    Args:
        checkpoint_path: Path to the Lightning .ckpt file.
        model_name: timm model name. Defaults to "vit_base_patch16_224".
        global_pool: Global pooling strategy; "avg" is commonly used for MoCo v3.
        device: Optional device to move the model to.
        map_location: torch.load map_location used for reading the checkpoint.
        strict: Whether to enforce exact key matching when loading.

    Returns:
        (model, missing_keys, unexpected_keys)
    """
    model = init_mocov3_vit_base(model_name=model_name, global_pool=global_pool, device=device)
    missing, unexpected = load_mocov3_vit_backbone_from_checkpoint(
        model,
        checkpoint_path,
        map_location=map_location,
        strict=strict,
    )
    return model, missing, unexpected


