"""
Utilities to initialize and load MoCo v3 ResNet backbones from Lightning checkpoints.

This module provides only initialization and checkpoint loading for ResNet backbones, e.g., RN50.
"""

from typing import Dict, Optional, Tuple

import torch


def _lazy_import_torchvision():
    try:
        import torchvision  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "torchvision is required to initialize ResNet models. Install via `pip install torchvision`."
        ) from exc
    return torchvision


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


def init_mocov3_resnet50(
    *,
    pretrained: bool = False,
    zero_init_residual: bool = True,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Initialize a torchvision ResNet-50 backbone suitable for loading MoCo v3 weights.
    The classification head is removed by setting `num_classes=0` in newer torchvision or
    by replacing `fc` with Identity for compatibility.
    """
    tv = _lazy_import_torchvision()
    try:
        model = tv.models.resnet50(weights=None)
    except TypeError:
        # older torchvision API
        model = tv.models.resnet50(pretrained=pretrained)

    # Remove classifier head to make a pure backbone
    if hasattr(model, "fc"):
        model.fc = torch.nn.Identity()

    if zero_init_residual and hasattr(model, "zero_init_residual"):
        model.zero_init_residual()

    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def _extract_backbone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("backbone.", "model.backbone.", "module.backbone.")
    filtered: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                filtered[new_key] = tensor
                break
    return filtered if filtered else state_dict


def load_mocov3_resnet_backbone_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    map_location: Optional[str] = "cpu",
    strict: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Load MoCo v3 ResNet backbone weights into an initialized torchvision model.
    """
    ckpt_obj = _torch_load_with_safe_globals(checkpoint_path, map_location=map_location)
    if not isinstance(ckpt_obj, dict):
        raise RuntimeError("Checkpoint is not a dict; unsupported format for this loader.")

    raw_state = ckpt_obj.get("state_dict", ckpt_obj)
    backbone_state = _extract_backbone_state_dict(raw_state)

    filtered_state: Dict[str, torch.Tensor] = {}
    for k, v in backbone_state.items():
        if isinstance(v, torch.Tensor):
            filtered_state[k] = v

    load_result = model.load_state_dict(filtered_state, strict=strict)
    return tuple(load_result.missing_keys), tuple(load_result.unexpected_keys)


def create_mocov3_resnet50_from_checkpoint(
    checkpoint_path: str,
    *,
    device: Optional[torch.device] = None,
    map_location: Optional[str] = "cpu",
    strict: bool = False,
) -> Tuple[torch.nn.Module, Tuple[str, ...], Tuple[str, ...]]:
    """
    Convenience helper to initialize a ResNet50 backbone and load MoCo v3 weights.
    """
    model = init_mocov3_resnet50(device=device)
    missing, unexpected = load_mocov3_resnet_backbone_from_checkpoint(
        model,
        checkpoint_path,
        map_location=map_location,
        strict=strict,
    )
    return model, missing, unexpected


