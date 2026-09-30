"""Shared-ASG local response experiments; all expensive ASGs stay per Gaussian."""

import math

import torch


def response_rank(mode):
    modes = {"legacy": 0, "full": 8, "rank2": 2, "rank4": 4,
             "full_sharpness": 8, "rank2_sharpness": 2}
    if mode not in modes:
        raise ValueError(f"Unknown texture_specular_response: {mode}")
    return modes[mode]


def response_channel_count(mode):
    rank = response_rank(mode)
    count = 9 if mode.removesuffix("_sharpness") == "full" else rank
    return count + int(mode.endswith("_sharpness"))


def initialize_response_alpha(alpha, mode):
    """Pack auxiliary basis contrasts beside the existing mean weights.

    Reusing the alpha_asg tensor preserves its optimizer, cloning, pruning,
    capture and appearance-file handling. get_alpha_asg exposes only the mean.
    A private generator leaves the scene initialization RNG unchanged.
    """
    rank = response_rank(mode)
    if mode.removesuffix("_sharpness") in {"legacy", "full"}:
        return alpha
    generator = torch.Generator(device=alpha.device).manual_seed(20260906)
    contrast = 0.05 * torch.randn(
        (*alpha.shape[:2], rank - 1), generator=generator,
        device=alpha.device, dtype=alpha.dtype,
    )
    return torch.cat((alpha, contrast), dim=2)


def local_asg_response(components, fresnel, weights, packed_alpha, tex_specular, mode):
    """Return [N,C,H,W] specular factors, before Gaussian ks/cosine/distance."""
    rank = response_rank(mode)
    keep_sharpness = mode.endswith("_sharpness")
    core_mode = mode.removesuffix("_sharpness")
    if mode == "legacy" or tex_specular.ndim != 4:
        raise ValueError("Local response experiments require static 2D texture grids")
    n, channels, h, w = tex_specular.shape
    if components.shape[:2] != (n, 8) or weights.shape != (n, 8, 1):
        raise ValueError("Local response experiments require eight scalar ASG components")
    if channels != response_channel_count(mode):
        raise ValueError(f"Wrong tex_specular channels for {mode}: {channels}")
    local = tex_specular[:, 2 if keep_sharpness else 1:].flatten(2).transpose(1, 2)
    if core_mode == "full":
        effective = (2.0 * torch.sigmoid(local)) * weights.transpose(1, 2)
        response = torch.bmm(effective, components) * fresnel
    elif core_mode == "rank2":
        if packed_alpha.shape != (n, 8, 2):
            raise ValueError("rank2 requires packed mean + one basis contrast")
        weighted = components * weights
        mean = weighted.sum(dim=1, keepdim=True) * fresnel
        contrast = (weighted * torch.tanh(packed_alpha[:, :, 1:2])).sum(dim=1, keepdim=True) * fresnel
        response = mean + torch.tanh(local) * contrast
    else:
        if packed_alpha.shape != (n, 8, rank):
            raise ValueError("Wrong number of packed shared-response weights")
        logits = packed_alpha[:, :, 1:].transpose(1, 2)
        logits = torch.cat((torch.zeros_like(logits[:, :1]), logits), dim=1)
        factors = rank * torch.softmax(logits, dim=1)
        shared = torch.bmm(factors * weights.transpose(1, 2), components) * fresnel
        local_logits = torch.cat((torch.zeros_like(local[:, :, :1]), local), dim=2)
        response = torch.bmm(torch.softmax(local_logits, dim=2), shared)
    gain = torch.exp(2.0 * torch.tanh(tex_specular[:, :1]))
    response = response.transpose(1, 2).reshape(n, -1, h, w)
    if keep_sharpness:
        sharpness = torch.exp(math.log(2.0) * torch.tanh(tex_specular[:, 1:2]))
        response = response.clamp_min(1e-8).pow(sharpness)
    return response * gain
