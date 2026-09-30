"""Inference math for the isolated local-specular replacement modes."""

from __future__ import annotations

import torch


_MODE_CHANNELS = {"normal_only": 2, "asg_mix_only": 8}


def channels(mode: str) -> int:
    try:
        return _MODE_CHANNELS[mode]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported specular replacement mode: {mode!r}") from exc


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    return value


def normal_frames(base_frames: torch.Tensor, raw: torch.Tensor, scale: float = 0.35) -> torch.Tensor:
    base_frames = _require_tensor(base_frames, "base_frames")
    raw = _require_tensor(raw, "raw")
    if base_frames.ndim < 2 or base_frames.shape[-2:] != (3, 3):
        raise ValueError("base_frames must have shape [..., 3, 3]")
    if raw.ndim < 1 or raw.shape[-1] != 2:
        raise ValueError("raw must have shape [..., 2]")
    if base_frames.shape[:-2] != raw.shape[:-1]:
        raise ValueError("base_frames and raw leading shapes must match")
    if base_frames.device != raw.device or base_frames.dtype != raw.dtype:
        raise ValueError("base_frames and raw must share dtype and device")

    a, b = (scale * torch.tanh(raw)).unbind(dim=-1)
    normal = torch.stack((a, b, torch.ones_like(a)), dim=-1)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    nx, ny, nz = normal.unbind(dim=-1)
    denominator = 1 + nz
    one = torch.ones_like(denominator)
    r00 = one - nx * nx / denominator
    r01 = -nx * ny / denominator
    r02 = nx
    r10 = r01
    r11 = one - ny * ny / denominator
    r12 = ny
    r20 = -nx
    r21 = -ny
    r22 = one - (nx * nx + ny * ny) / denominator
    rotation = torch.stack(
        (
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1),
        ),
        dim=-2,
    )
    return base_frames @ rotation


def mix_asg(
    components: torch.Tensor,
    fresnel: torch.Tensor,
    weights: torch.Tensor,
    raw: torch.Tensor,
) -> torch.Tensor:
    components = _require_tensor(components, "components")
    fresnel = _require_tensor(fresnel, "fresnel")
    weights = _require_tensor(weights, "weights")
    raw = _require_tensor(raw, "raw")
    if components.ndim != 3 or components.shape[1] != 8:
        raise ValueError("components must have shape [N, 8, C]")
    n = components.shape[0]
    if fresnel.shape != (n, 1, 1):
        raise ValueError("fresnel must have shape [N, 1, 1]")
    if weights.shape != (n, 8, 1):
        raise ValueError("weights must have shape [N, 8, 1]")
    if raw.ndim != 4 or raw.shape[:2] != (n, 8):
        raise ValueError("raw must have shape [N, 8, H, W]")

    coefficients = weights.unsqueeze(-1) * (2 * torch.sigmoid(raw))
    response = (coefficients.unsqueeze(2) * components.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    return response * fresnel.unsqueeze(-1)


__all__ = ["channels", "normal_frames", "mix_asg"]
