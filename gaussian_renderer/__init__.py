#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import importlib.machinery
import math
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_COMPILED_SUBMODULE_PACKAGES = {
    "surfel-texture": "surfel_texture",
    "surfel-texture-deferred": "surfel_texture_deferred",
    "diff-surfel-rasterization-shadow": "diff_surfel_rasterization_shadow",
}


def _has_compatible_extension(path, package):
    package_dir = os.path.join(path, package)
    return any(
        os.path.exists(os.path.join(package_dir, "_C" + suffix))
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


for _folder, _package in reversed(tuple(_COMPILED_SUBMODULE_PACKAGES.items())):
    _path = os.path.join(_REPO_ROOT, "submodules", _folder)
    if _has_compatible_extension(_path, _package) and _path not in sys.path:
        sys.path.insert(0, _path)

from surfel_texture import GaussianRasterizationSettings as surfel_settings
from surfel_texture import GaussianRasterizer as surfel_rasterizer
from surfel_texture_deferred import GaussianRasterizer as surfel_rasterizer_deferred
from diff_surfel_rasterization_shadow import GaussianRasterizationSettings as surfel_shadow_settings
from diff_surfel_rasterization_shadow import GaussianRasterizer as surfel_shadow_rasterizer
from gaussian_renderer.texture_branch import render_2dgs_texture_deferred
from scene.gaussian_model_native_2dgs import GaussianModel
from utils.graphics_utils import getProjectionMatrix
from utils.native_backend import validate_native_backend


def _build_2dgs_raster_settings(viewpoint_camera, pipe, bg_color, scaling_modifier, sh_degree):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    return surfel_settings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
    )


def _look_at_2dgs(camera_position, target_position, up_dir):
    camera_direction = camera_position - target_position
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    if abs(np.dot(up_dir, camera_direction)) > 0.9:
        up_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    camera_right = np.cross(up_dir, camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)

    rotation_transform = np.zeros((4, 4), dtype=np.float32)
    rotation_transform[0, :3] = camera_right
    rotation_transform[1, :3] = camera_up
    rotation_transform[2, :3] = camera_direction
    rotation_transform[3, 3] = 1.0

    translation_transform = np.eye(4, dtype=np.float32)
    translation_transform[:3, -1] = -np.asarray(camera_position, dtype=np.float32)

    look_at_transform = rotation_transform @ translation_transform
    look_at_transform[1:3, :] *= -1
    return look_at_transform.T


def _build_light_transform_2dgs(viewpoint_camera, means3d, pipe):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx_origin = viewpoint_camera.image_width / (2.0 * tanfovx)
    fy_origin = viewpoint_camera.image_height / (2.0 * tanfovy)

    object_center = means3d.mean(dim=0).detach().cpu().numpy()
    light_position = viewpoint_camera.pl_pos.detach().cpu().numpy().reshape(-1)
    world_view_transform_light = _look_at_2dgs(
        light_position,
        object_center,
        up_dir=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    world_view_transform_light = torch.tensor(
        world_view_transform_light,
        device=viewpoint_camera.world_view_transform.device,
        dtype=viewpoint_camera.world_view_transform.dtype,
    )

    camera_position = viewpoint_camera.camera_center.detach().cpu().numpy() * getattr(pipe, "shadow_light_scale", 1.0)
    light_norm = max(float(np.sum(light_position * light_position)), 1e-8)
    camera_norm = max(float(np.sum(camera_position * camera_position)), 1e-8)
    f_scale_ratio = math.sqrt(light_norm / camera_norm)
    fx_far = fx_origin * f_scale_ratio
    fy_far = fy_origin * f_scale_ratio

    tanfovx_far = 0.5 * viewpoint_camera.image_width / fx_far
    tanfovy_far = 0.5 * viewpoint_camera.image_height / fy_far
    resolution_scale = max(float(getattr(pipe, "shadow_resolution_scale", 1.0)), 0.25)
    h_light = min(max(32, int(resolution_scale * viewpoint_camera.image_height)), 2048)
    w_light = min(max(32, int(resolution_scale * viewpoint_camera.image_width)), 2048)

    light_persp_proj_matrix = getProjectionMatrix(
        znear=viewpoint_camera.znear,
        zfar=viewpoint_camera.zfar,
        fovX=2.0 * math.atan(tanfovx_far),
        fovY=2.0 * math.atan(tanfovy_far),
    ).transpose(0, 1).cuda()
    light_projmatrix = (
        world_view_transform_light.unsqueeze(0).bmm(light_persp_proj_matrix.unsqueeze(0))
    ).squeeze(0)

    return dict(
        world_view_transform_light=world_view_transform_light,
        light_projmatrix=light_projmatrix,
        tanfovx_far=tanfovx_far,
        tanfovy_far=tanfovy_far,
        h_light=h_light,
        w_light=w_light,
        light_position=light_position,
    )


def _compute_shadow_pass_2dgs_native(viewpoint_camera, gau, pipe, bg_color, scaling_modifier=1.0):
    if viewpoint_camera.pl_pos is None or gau.get_xyz.numel() == 0:
        return None

    means3d = gau.get_xyz
    opacity = gau.get_opacity
    scales = gau.get_scaling
    rotations = gau.get_rotation
    lt = _build_light_transform_2dgs(viewpoint_camera, means3d, pipe)

    shadow_settings = surfel_shadow_settings(
        image_height=lt["h_light"],
        image_width=lt["w_light"],
        tanfovx=lt["tanfovx_far"],
        tanfovy=lt["tanfovy_far"],
        bg=bg_color[:3],
        scale_modifier=scaling_modifier,
        viewmatrix=lt["world_view_transform_light"],
        projmatrix=lt["light_projmatrix"],
        sh_degree=gau.active_sh_degree,
        campos=torch.tensor(lt["light_position"], dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=getattr(pipe, "debug", False),
        low_pass_filter_radius=0.3,
        ortho=False,
        use_textures=False,
    )
    shadow_rasterizer = surfel_shadow_rasterizer(raster_settings=shadow_settings)
    light_colors = torch.ones((means3d.shape[0], 3), dtype=torch.float32, device="cuda")

    _, _, _, out_trans, non_trans, _ = shadow_rasterizer(
        means3D=means3d,
        means2D=torch.zeros_like(means3d, requires_grad=True),
        shs=None,
        colors_precomp=light_colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=None,
        texture_alpha=torch.empty(0, device="cuda"),
        texture_sigma_factor=3.0,
        non_trans=None,
        offset=getattr(pipe, "shadow_offset", 0.015),
        thres=-1.0,
        is_train=False,
    )
    per_point_shadow = (out_trans / torch.clamp_min(non_trans, 1e-6)).unsqueeze(-1)
    return {
        "per_point_shadow": per_point_shadow,
        "light_viewmatrix": lt["world_view_transform_light"],
        "light_projmatrix": lt["light_projmatrix"],
    }


def _render_2dgs_native_3ch(viewpoint_camera, gau, pipe, bg_color, modelset, scaling_modifier=1.0, fix_labert=False, shadow_map=False, iteration=0):
    means3D = gau.get_xyz
    means2D = torch.zeros_like(means3D, dtype=torch.float32, requires_grad=True, device=means3D.device)
    opacity = gau.get_opacity
    scales = gau.get_scaling
    rotations = gau.get_rotation
    transmat_grad_holder = None
    if getattr(viewpoint_camera, "cam_pose_adj", None) is not None and viewpoint_camera.cam_pose_adj.requires_grad:
        transmat_grad_holder = torch.zeros(
            (means3D.shape[0], 9),
            dtype=torch.float32,
            device=means3D.device,
            requires_grad=True,
        )

    if gau.use_MBRDF:
        shadow_pkg = _compute_shadow_pass_2dgs_native(viewpoint_camera, gau, pipe, bg_color, scaling_modifier)
    else:
        shadow_pkg = None

    colors_precomp = None
    shs = None
    shadow_img = None
    other_img = None
    if gau.use_MBRDF:
        pl_pos_expand = viewpoint_camera.pl_pos.expand(gau.get_xyz.shape[0], -1)
        wi_ray = pl_pos_expand - gau.get_xyz
        wi_dist2 = torch.sum(wi_ray**2, dim=-1, keepdim=True).clamp_min(1e-12)
        dist_2_inv = 1.0 / wi_dist2
        wi = wi_ray * torch.sqrt(dist_2_inv)
        camera_center_for_brdf = viewpoint_camera.camera_center
        if os.getenv("GS3_2DGS_DETACH_VIEWDIR", "0") == "1":
            camera_center_for_brdf = camera_center_for_brdf.detach()
        wo = _safe_normalize(camera_center_for_brdf - gau.get_xyz)

        local_axises = gau.get_local_axis
        local_z = local_axises[:, :, 2]
        wi_local = torch.einsum('Ki,Kij->Kj', wi, local_axises)
        wo_local = torch.einsum('Ki,Kij->Kj', wo, local_axises)
        cosTheta = _NdotWi(local_z, wi, torch.nn.ELU(alpha=0.01), 0.01)
        diffuse = gau.get_kd / math.pi
        asg_scales = gau.asg_func.get_asg_lam_miu
        asg_axises = gau.asg_func.get_asg_axis
        asg_1 = gau.asg_func(wi_local, wo_local, gau.get_alpha_asg, asg_scales, asg_axises)
        shadow_hint = None if shadow_pkg is None else shadow_pkg["per_point_shadow"]
        decay, other_effects, _, _ = gau.neural_phasefunc(
            wi, wo, gau.get_xyz, gau.get_neural_material,
            hint=shadow_hint,
        )
        if decay is None:
            decay = torch.ones((means3D.shape[0], 1), dtype=torch.float32, device=means3D.device)
        if fix_labert:
            basecolor = diffuse * cosTheta * dist_2_inv
        else:
            specular = gau.get_ks * asg_1
            basecolor = (diffuse + specular) * cosTheta * dist_2_inv
        if other_effects is not None:
            other_effects = other_effects * dist_2_inv
        else:
            other_effects = torch.zeros_like(basecolor)
        colors_precomp = basecolor * decay + other_effects
    else:
        shs = gau.get_features

    raster_settings_2dgs = _build_2dgs_raster_settings(
        viewpoint_camera, pipe, bg_color[:3], scaling_modifier, gau.active_sh_degree
    )
    rasterizer_2dgs = surfel_rasterizer(raster_settings=raster_settings_2dgs)
    rendered_rgb, radii, allmap = rasterizer_2dgs(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=shs if colors_precomp is None else None,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        texture_color=None,
        texture_alpha=None,
        use_textures=False,
        transmat_grad_holder=transmat_grad_holder,
    )
    try:
        means2D.retain_grad()
    except:
        pass

    render_alpha = allmap[1:2].clamp_min(1e-8)
    expected_depth = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
    shadow_img = torch.ones((1, rendered_rgb.shape[1], rendered_rgb.shape[2]), dtype=rendered_rgb.dtype, device=rendered_rgb.device)
    other_img = torch.zeros((3, rendered_rgb.shape[1], rendered_rgb.shape[2]), dtype=rendered_rgb.dtype, device=rendered_rgb.device)
    return {
        "render": rendered_rgb,
        "shadow": shadow_img,
        "other_effects": other_img,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        "out_weight": torch.zeros((means3D.shape[0], 1), dtype=torch.float32, device="cuda"),
        "backward_info": {},
        "shadow_stage": get_shadow_backward_stage(modelset, iteration),
        "transmat_grad_holder": transmat_grad_holder,
        "expected_depth": expected_depth,
    }


def _render_2dgs_native_deferred(viewpoint_camera, gau, pipe, bg_color, modelset, scaling_modifier=1.0, fix_labert=False, shadow_map=False, iteration=0):
    means3D = gau.get_xyz
    means2D = torch.zeros_like(means3D, dtype=torch.float32, requires_grad=True, device=means3D.device)
    opacity = gau.get_opacity
    scales = gau.get_scaling
    rotations = gau.get_rotation
    transmat_grad_holder = None
    if getattr(viewpoint_camera, "cam_pose_adj", None) is not None and viewpoint_camera.cam_pose_adj.requires_grad:
        transmat_grad_holder = torch.zeros(
            (means3D.shape[0], 9),
            dtype=torch.float32,
            device=means3D.device,
            requires_grad=True,
        )

    if gau.use_MBRDF:
        shadow_pkg = _compute_shadow_pass_2dgs_native(viewpoint_camera, gau, pipe, bg_color, scaling_modifier)
    else:
        shadow_pkg = None

    colors_precomp = None
    if gau.use_MBRDF:
        pl_pos_expand = viewpoint_camera.pl_pos.expand(gau.get_xyz.shape[0], -1)
        wi_ray = pl_pos_expand - gau.get_xyz
        wi_dist2 = torch.sum(wi_ray**2, dim=-1, keepdim=True).clamp_min(1e-12)
        dist_2_inv = 1.0 / wi_dist2
        wi = wi_ray * torch.sqrt(dist_2_inv)
        camera_center_for_brdf = viewpoint_camera.camera_center
        if os.getenv("GS3_2DGS_DETACH_VIEWDIR", "0") == "1":
            camera_center_for_brdf = camera_center_for_brdf.detach()
        wo = _safe_normalize(camera_center_for_brdf - gau.get_xyz)

        local_axises = gau.get_local_axis
        local_z = local_axises[:, :, 2]
        wi_local = torch.einsum('Ki,Kij->Kj', wi, local_axises)
        wo_local = torch.einsum('Ki,Kij->Kj', wo, local_axises)
        cosTheta = _NdotWi(local_z, wi, torch.nn.ELU(alpha=0.01), 0.01)
        diffuse = gau.get_kd / math.pi
        asg_scales = gau.asg_func.get_asg_lam_miu
        asg_axises = gau.asg_func.get_asg_axis
        asg_1 = gau.asg_func(wi_local, wo_local, gau.get_alpha_asg, asg_scales, asg_axises)
        shadow_hint = None if shadow_pkg is None else shadow_pkg["per_point_shadow"]
        decay, other_effects, _, _ = gau.neural_phasefunc(
            wi, wo, gau.get_xyz, gau.get_neural_material,
            hint=shadow_hint,
        )
        if decay is None:
            decay = torch.ones((means3D.shape[0], 1), dtype=torch.float32, device=means3D.device)
        if fix_labert:
            basecolor = diffuse * cosTheta * dist_2_inv
        else:
            specular = gau.get_ks * asg_1
            basecolor = (diffuse + specular) * cosTheta * dist_2_inv
        if other_effects is None:
            other_effects = torch.zeros_like(basecolor)
        else:
            other_effects = other_effects * dist_2_inv
        colors_precomp = torch.cat([basecolor, decay, other_effects], dim=1)
    else:
        colors_precomp = None

    surfel_bg = bg_color
    if surfel_bg.shape[0] == 3:
        surfel_bg = torch.cat(
            [surfel_bg, torch.zeros(4, dtype=surfel_bg.dtype, device=surfel_bg.device)],
            dim=0,
        )
    raster_settings_2dgs = _build_2dgs_raster_settings(
        viewpoint_camera, pipe, surfel_bg, scaling_modifier, gau.active_sh_degree
    )
    rasterizer_2dgs = surfel_rasterizer_deferred(raster_settings=raster_settings_2dgs)
    rendered_7ch, radii, allmap = rasterizer_2dgs(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=None,
        colors_precomp=colors_precomp,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        texture_color=None,
        texture_alpha=None,
        use_textures=False,
        transmat_grad_holder=transmat_grad_holder,
    )
    try:
        means2D.retain_grad()
    except:
        pass

    render_alpha = allmap[1:2].clamp_min(1e-8)
    expected_depth = torch.nan_to_num(allmap[0:1] / render_alpha, 0, 0)
    return {
        "render": rendered_7ch[0:3],
        "shadow": rendered_7ch[3:4],
        "other_effects": rendered_7ch[4:7],
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        "out_weight": torch.zeros((means3D.shape[0], 1), dtype=torch.float32, device="cuda"),
        "backward_info": {},
        "shadow_stage": get_shadow_backward_stage(modelset, iteration),
        "transmat_grad_holder": transmat_grad_holder,
        "expected_depth": expected_depth,
    }


def get_shadow_backward_stage(modelset, iteration: int):
    if getattr(modelset, "detach_shadow", False):
        return {
            "xyz": False,
            "opacity": False,
            "scaling": False,
            "rotation": False,
        }
    if not getattr(modelset, "shadow_backward_stage_enabled", False):
        return {
            "xyz": True,
            "opacity": True,
            "scaling": True,
            "rotation": True,
        }
    return {
        "xyz": iteration >= int(getattr(modelset, "shadow_backward_xyz_from_iter", 0)),
        "opacity": iteration >= int(getattr(modelset, "shadow_backward_opacity_from_iter", 0)),
        "scaling": iteration >= int(getattr(modelset, "shadow_backward_scaling_from_iter", 0)),
        "rotation": iteration >= int(getattr(modelset, "shadow_backward_rotation_from_iter", 0)),
    }


def render(
    viewpoint_camera,
    gau: GaussianModel,
    light_stream,
    calc_stream,
    local_axises,
    asg_scales,
    asg_axises,
    pipe,
    bg_color: torch.Tensor,
    modelset,
    shadowmap_render=False,
    scaling_modifier=1.0,
    override_color=None,
    fix_labert=False,
    inten_scale=1.0,
    is_train=False,
    asg_mlp=False,
    iteration=0,
    inference_state=None,
):
    """Render native 2DGS, including untextured initialization and local texels."""
    validate_native_backend(modelset)
    backend = validate_native_backend(gau)
    if shadowmap_render:
        raise ValueError("shadowmap_render is a retired 3DGS path; use native LoFT-GS rendering.")
    if not hasattr(gau, "get_scaling") or gau.get_scaling.shape[-1] != 2:
        raise ValueError("Native LoFT-GS requires a two-axis 2DGS model.")

    if backend == "2dgs_3ch":
        return _render_2dgs_native_3ch(
            viewpoint_camera, gau, pipe, bg_color, modelset,
            scaling_modifier=scaling_modifier, fix_labert=fix_labert,
            iteration=iteration,
        )
    if bool(getattr(gau, "use_textures", False)):
        return render_2dgs_texture_deferred(
            viewpoint_camera, gau, pipe, bg_color, modelset,
            scaling_modifier=scaling_modifier, fix_labert=fix_labert,
            iteration=iteration, light_stream=light_stream,
            calc_stream=calc_stream, inference_state=inference_state,
        )
    return _render_2dgs_native_deferred(
        viewpoint_camera, gau, pipe, bg_color, modelset,
        scaling_modifier=scaling_modifier, fix_labert=fix_labert,
        iteration=iteration,
    )


def _dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def _safe_normalize(x):
    return torch.nn.functional.normalize(x, dim = -1, eps=1e-8)


def _NdotWi(nrm, wi, elu, a):
    """
    nrm: (N, 3)
    wi: (N, 3)
    _dot(nrm, wi): (N, 1)
    return (N, 1)
    """
    tmp  = a * (1. - 1 / math.e)
    return (elu(_dot(nrm, wi)) + tmp) / (1. + tmp)
