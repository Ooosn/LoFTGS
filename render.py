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


import torch
import copy
import json
import sys
import math
import cv2
from scipy.spatial.transform import Rotation as Rot
import torchvision.transforms.functional as F
from PIL import Image
from scene import Scene
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from gaussian_renderer.inference import TextureInferenceState
from gaussian_renderer.shadow_render import shadow_render
import torchvision
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from arguments import explicit_cli_value as _explicit_cli_value
from gaussian_renderer import GaussianModel
from scene.gaussian_model_2dgs_adapter import GaussianModel2DGSAdapter
from scene.neural_phase_function import Neural_phase
from scene.mixture_ASG import Mixture_of_ASG
from utils.system_utils import searchForMaxIteration
from utils.shadow_transport import shadow_sensitivity_anchor
from rich import print
from rich.panel import Panel
import matplotlib.pyplot as plt
from write_vedio import images_to_video
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=8)


def save_render_image(image, path):
    return executor.submit(torchvision.utils.save_image, image, path)

def create_render_streams():
    return torch.cuda.Stream(), torch.cuda.Stream()


def _is_2dgs_model(modelset):
    return str(getattr(modelset, "rasterizer", "")).startswith("2dgs")

def _make_gaussian_model(modelset, opt=None):
    if _is_2dgs_model(modelset):
        return GaussianModel2DGSAdapter(modelset, opt)
    return GaussianModel(modelset)

def _unpack_training_checkpoint(checkpoint):
    if not isinstance(checkpoint, (tuple, list)):
        return checkpoint, None, None
    if len(checkpoint) == 2:
        model_params, first_iter = checkpoint
        return model_params, first_iter, None
    if len(checkpoint) == 3:
        model_params, first_iter, scene_state = checkpoint
        return model_params, first_iter, scene_state
    raise ValueError(f"Unsupported checkpoint format with {len(checkpoint)} entries")

def render_set(modelset, name, iteration, views, gaussians, pipeline, background,
               gamma, hdr, write_images=False, force_save=False,
               synthesize_video=False, shadowmap_render=False, fast_inference=False):
    if not views:
        raise ValueError(f"No cameras available for the {name} split")
    if synthesize_video and not write_images:
        raise ValueError("--synthesize_video requires --write_images")
    model_path = modelset.model_path
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    if write_images:
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

    light_stream, calc_stream = create_render_streams()
    state = None
    render_model = gaussians
    if fast_inference:
        if str(getattr(gaussians, "rasterizer", "")) != "2dgs" or not getattr(gaussians, "use_textures", False):
            raise ValueError("--fast_inference requires the textured 2DGS renderer")
        state = TextureInferenceState()
        render_model = state.prepare(gaussians)
        state.wait_ready(light_stream)
        state.wait_ready(calc_stream)
    local_axises = render_model.get_local_axis
    asg_scales = render_model.asg_func.get_asg_lam_miu
    asg_axises = render_model.asg_func.get_asg_axis

    render_shadow = None
    render_other_effects = None
    render_base = None
    rendering = None
    renderArgs = {"modelset": modelset, "pipe": pipeline, "bg_color": background, "iteration": iteration}
    if state is not None:
        renderArgs["inference_state"] = state
    def _render_views():
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            render_pkg = render(view, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, **renderArgs, shadowmap_render=shadowmap_render)
            render_shadow = render_pkg["shadow"]
            render_other_effects = render_pkg["other_effects"]
            render_base = render_pkg["render"]
            rendering = render_pkg["render"]* render_pkg["shadow"] + render_pkg["other_effects"]
            expected_depth = render_pkg.get("expected_depth")
            depth_image = render_pkg.get("depth_image")
            image_folder = {}
            if write_images:
                gt = view.original_image[0:3, :, :]
                # 如果图片是 hdr 格式，则 gamma 校正
                # png 格式，是直接在 sRGB 空间 下重建的，因此不需要 gamma 校正
                if gamma:
                    if hdr:
                        gt = torch.pow(gt, 1/2.2)
                    rendering = torch.clip(rendering, 0.0, 1.0)
                    rendering = torch.pow(rendering, 1/2.2)
                if not torch.isfinite(rendering).all() or not torch.isfinite(gt).all():
                    raise FloatingPointError(f"Non-finite RGB/GT in {name} frame {idx}")
                
                file_prefix = "shadowmap" if shadowmap_render else "volume"
                image_shadow_path = os.path.join(render_path, file_prefix + "_shadow")
                image_other_effects_path = os.path.join(render_path, 'other_effects')
                image_base_path = os.path.join(render_path, 'base')
                image_final_image = os.path.join(render_path, file_prefix + "_final_image")
                image_folder[gts_path] = gt
                image_folder[image_shadow_path] = render_shadow
                image_folder[image_other_effects_path] = render_other_effects
                image_folder[image_base_path] = render_base
                image_folder[image_final_image] = rendering
                
                if expected_depth is not None:
                    image_expected_depth_path = os.path.join(render_path, 'expected_depth')
                    mask = torch.isnan(expected_depth) | (expected_depth <= 0)
                    expected_depth[mask] = torch.nan

                    # 2. 转视差（可选）
                    disp = 1.0 / expected_depth
                    disp[mask] = torch.nan
                    disp_np = disp.cpu().numpy()
                    # 3. 归一化（用 1–99% 分位数裁剪）
                    vmin, vmax = np.nanpercentile(disp_np, (1, 99))
                    disp_clipped = torch.clip(disp, vmin, vmax)
                    normed_1 = (disp_clipped - vmin) / (vmax - vmin)
                    image_folder[image_expected_depth_path] = normed_1
                    
                    print("percentile:", vmin, vmax)
                    
                if depth_image is not None:
                    image_depth_image_path = os.path.join(render_path, 'depth_image')
                    print("depth min/max:", torch.min(depth_image), torch.max(depth_image))
                    mask = torch.isnan(depth_image) | (depth_image <= 1e-3)
                    depth_image[mask] = torch.nan

                    # 2. 转视差（可选）
                    disp = 1.0 / depth_image
                    disp[mask] = torch.nan
                    disp_np = disp.cpu().numpy()
                    # 3. 归一化（用 1–99% 分位数裁剪）  
                    vmin, vmax = np.nanpercentile(disp_np, (1, 99))
                    print("percentile:", vmin, vmax)
                    disp_clipped = torch.clip(disp, vmin, vmax)
                    normed_2 = (disp_clipped - vmin) / (vmax - vmin)
                    normed_2[mask] = 0
                    image_folder[image_depth_image_path] = normed_2
                
                writes = []
                for key, value in image_folder.items():
                    os.makedirs(key, exist_ok=True)
                    image_path = os.path.join(key, '{0:05d}'.format(idx) + ".png")
                    if force_save or not os.path.exists(image_path):
                        writes.append(save_render_image(value, image_path))
                for write in writes:
                    write.result()
            
        if synthesize_video:
            # 获取文件夹下的图片，按顺序合成视频
            # video_path = os.path.join(render_path, 'video')
            video_path = render_path
            os.makedirs(video_path, exist_ok=True)
            for key, value in image_folder.items():
                out_file_name = os.path.basename(key) + "_video.mp4"
                print(out_file_name)
                out_file_path = os.path.join(video_path, out_file_name)
                images_to_video(key, out_file_path, fps=30, size="auto", codec='mp4v')

    _render_views()

def render_sets(modelset : ModelParams,
                iteration : int, 
                pipeline : PipelineParams, 
                skip_train : bool, 
                skip_test : bool, 
                opt_pose: bool, 
                gamma: bool,
                hdr: bool,
                valid: bool,
                write_images: bool,
                force_save: bool,
                synthesize_video: bool,
                shadowmap_render: bool,
                shadow_transport_override=None,
                opt=None,
                fast_inference=False,
                shadow_anchor_override=None):
    if fast_inference and shadowmap_render:
        raise ValueError("--fast_inference supports native RGB/component rendering only")
    if gamma and not hdr:
        raise ValueError("--gamma requires --hdr; omit both for LDR/NRHints scenes")
    if skip_train and skip_test and not valid:
        raise ValueError("At least one camera split must be rendered")
    if getattr(modelset, "texture_specular_response", None) is None:
        modelset.texture_specular_response = "legacy"
    modelset.data_device = "cpu"

    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(modelset.model_path, "point_cloud"))
    scene_state = None
    if modelset.use_nerual_phasefunc:
        checkpoint_path = os.path.join(modelset.model_path, f"chkpnt{iteration}.pth")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Could not find: {checkpoint_path}")
        model_params, first_iter, scene_state = _unpack_training_checkpoint(
            torch.load(checkpoint_path, weights_only=False)
        )

    if opt_pose:
        modelset.source_path = os.path.join(modelset.model_path, f'point_cloud/iteration_{iteration}')


    with torch.no_grad():

        # load Gaussians attributes, establish scene
        gaussians = _make_gaussian_model(modelset, opt)
        
        light_direction = torch.tensor([-80, -47, 50], dtype=torch.float32, device="cuda")
        gaussians.light_direction = torch.nn.Parameter(light_direction, requires_grad=True)
        l = [
            {'params': [gaussians.light_direction], 'lr': 0.001, "name": "light_direction"}
        ]
        optimizer = torch.optim.Adam(l, lr=0.001)
        
        # 创建场景实例
        # iteration 为 -1 则加载最新的模型，否则加载指定迭代次数的模型，如果不存在则创建新的模型文件夹，但这里作为渲染，iteration 肯定要的哇
        # valid 渲染验证集，skip_train 不渲染训练集，skip_test 不渲染测试集
        force_input_camera = os.environ.get("MICRO_FORCE_INPUT_CAMERA", "0") == "1"
        using_baked_optimized_pose = bool(opt_pose)
        scene = Scene(
            modelset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            valid=valid,
            skip_train=skip_train,
            skip_test=skip_test,
            load_optimized_cameras=not (
                force_input_camera or using_baked_optimized_pose
                or (_is_2dgs_model(modelset) and scene_state is not None)
            ),
        )
        
        """
        本项目中: scene 中储存的 model 也就是 ply 文件，只存储了每个高斯的属性，！！！但不包括每个高斯的神经网络参数！！！
            因此需要先从 ply 文件中加载高斯模型参数，然后从 _model_path = os.path.join(modelset.model_path, f"chkpnt{iteration}.pth")
                中提取神经网络参数和一些共用的参数（比如 asg 基函数参数）
        这里的 model_path 是 模型文件夹，包括两个子模型，一个是 point_cloud; 一个是 chkpnt (torch.save(gaussians.capture()) 得到的文件)
        point_cloud 中存储了高斯模型参数，chkpnt 中存储了优化器状态以及神经网络参数和一些共用的参数
        """
        if modelset.use_nerual_phasefunc:
            if _is_2dgs_model(modelset):
                if opt is None:
                    raise RuntimeError("2DGS checkpoint rendering requires OptimizationParams for restore().")
                gaussians.restore(model_params, opt, load_optimizer=False)
                gaussians.training_setup(opt)
                if scene_state is not None and not (force_input_camera or using_baked_optimized_pose):
                    scene.restore(scene_state)
                if bool(getattr(modelset, "use_textures", False)):
                    if hasattr(gaussians, "texture_effect_mode"):
                        gaussians.texture_effect_mode = str(getattr(modelset, "texture_effect_mode", gaussians.texture_effect_mode))
                    if hasattr(gaussians, "mbrdf_normal_source"):
                        gaussians.mbrdf_normal_source = str(getattr(modelset, "mbrdf_normal_source", gaussians.mbrdf_normal_source)).lower()
                    texture_model_overrides = {
                        "texture_shadow_confidence_enabled": bool,
                        "texture_shadow_confidence_start_iter": int,
                        "texture_shadow_confidence_strength": float,
                        "texture_shadow_confidence_gamma": float,
                        "texture_shadow_confidence_zero_dc": bool,
                        "texture_shadow_transport_mode": lambda value: str(value).lower(),
                        "texture_shadow_transport_eps": float,
                        "texture_factor_surgery": lambda value: str(value).lower(),
                        "texture_factor_surgery_seed": int,
                    }
                    for name, caster in texture_model_overrides.items():
                        if hasattr(gaussians, name):
                            setattr(gaussians, name, caster(getattr(modelset, name, getattr(gaussians, name))))
                if getattr(gaussians, "neural_phasefunc", None) is not None:
                    gaussians.neural_phasefunc.eval()
            else:
                gaussians.asg_func = Mixture_of_ASG(modelset.basis_asg_num, modelset.asg_channel_num)
                gaussians.neural_phasefunc = Neural_phase(hidden_feature_size=modelset.phasefunc_hidden_size, \
                                            hidden_feature_layers=modelset.phasefunc_hidden_layers, \
                                            frequency=modelset.phasefunc_frequency, \
                                            neural_material_size=modelset.neural_material_size, \
                                            asg_mlp = gaussians.asg_mlp).to(device="cuda")
                if isinstance(model_params, dict):
                    gaussians.asg_func.asg_sigma = model_params["asg_sigma"]
                    gaussians.asg_func.asg_rotation = model_params["asg_rotation"]
                    gaussians.asg_func.asg_scales = model_params["asg_scales"]
                    gaussians.neural_phasefunc.load_state_dict(model_params["neural_phasefunc"])
                else:
                    gaussians.asg_func.asg_sigma = model_params[8]
                    gaussians.asg_func.asg_rotation = model_params[9]
                    gaussians.asg_func.asg_scales = model_params[10]
                    gaussians.neural_phasefunc.load_state_dict(model_params[14])
                gaussians.neural_phasefunc.eval()
            del model_params, first_iter, scene_state
            if hasattr(gaussians, "optimizer"):
                gaussians.optimizer = None
            torch.cuda.empty_cache()

        if shadow_transport_override is not None:
            gaussians.texture_shadow_transport_mode = str(shadow_transport_override).lower()
            print(f"Shadow transport override: {gaussians.texture_shadow_transport_mode}")
        if shadow_anchor_override is not None:
            gaussians.texture_shadow_sensitivity_anchor = shadow_sensitivity_anchor(shadow_anchor_override)
            print(f"Shadow anchor override: {gaussians.texture_shadow_sensitivity_anchor}")

        bg_color = [1, 1, 1, 1, 0, 0, 0] if modelset.white_background else [0, 0, 0, 0, 0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if valid:
            print("current dataset includes valid")
            render_set(modelset, "valid", scene.loaded_iter, scene.getValidCameras(), 
                    gaussians, pipeline, background, gamma, hdr, write_images=write_images, force_save=force_save,
                    synthesize_video=synthesize_video, shadowmap_render=shadowmap_render, fast_inference=fast_inference)
        
        if not skip_train:
            print("current dataset includes train")
            render_set(modelset, "train", scene.loaded_iter, scene.getTrainCameras(), 
                    gaussians, pipeline, background, gamma, hdr, write_images=write_images, force_save=force_save,
                    synthesize_video=synthesize_video, shadowmap_render=shadowmap_render, fast_inference=fast_inference)

        if not skip_test:
            print("current dataset includes test")
            render_set(modelset, "test", scene.loaded_iter, scene.getTestCameras(), 
                    gaussians, pipeline, background, gamma, hdr, write_images=write_images, force_save=force_save,
                    synthesize_video=synthesize_video, shadowmap_render=shadowmap_render, fast_inference=fast_inference)
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    mp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--load_iteration", default=-1, type=int)   # -1 代表加载最新的模型
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gamma", action="store_true", default=False)
    parser.add_argument("--opt_pose", action="store_true", default=False)
    parser.add_argument("--valid", action="store_true", default=False)
    parser.add_argument("--write_images", action="store_true", default=False)
    parser.add_argument("--force_save", action="store_true", default=False)
    parser.add_argument("--synthesize_video", action="store_true", default=False)
    parser.add_argument("--shadowmap_render", action="store_true", default=False)
    args = get_combined_args(parser)
    args.wang_debug = False
    model_params = mp.extract(args)
    args.fast_inference = (
        str(getattr(model_params, "rasterizer", "")) == "2dgs"
        and bool(getattr(model_params, "use_textures", False))
        and not args.shadowmap_render
    )
    shadow_transport_override = _explicit_cli_value(args, "texture_shadow_transport_mode")
    shadow_anchor_override = _explicit_cli_value(args, "texture_shadow_sensitivity_anchor")

    args_info = f"""
    model_args: {vars(model_params)}
    load_iteration: {args.load_iteration}
    skip_train: {args.skip_train}
    skip_test: {args.skip_test}
    opt_pose: {args.opt_pose}
    gamma: {args.gamma}
    hdr: {args.hdr}
    valid: {args.valid}
    fast_inference: {args.fast_inference}
    """
    
    print(Panel(args_info, title="Arguments", expand=False))
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model_params, args.load_iteration, pp.extract(args), \
                args.skip_train, args.skip_test, args.opt_pose, args.gamma, args.hdr, args.valid, args.write_images,
                args.force_save, args.synthesize_video, args.shadowmap_render, shadow_transport_override, op.extract(args), args.fast_inference,
                shadow_anchor_override=shadow_anchor_override)
