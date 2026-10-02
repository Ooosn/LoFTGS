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

from argparse import ArgumentParser, Namespace
import ast
import sys
import os
from utils.shadow_transport import shadow_sensitivity_anchor

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        explicit_bool_pairs = {
            "detach_shadow",
            "use_shadow_refine_mlp",
            "shadow_backward_stage_enabled",
            "shadow_scale_reg_enabled",
            "texture_render_use_alpha",
            "texture_shadow_use_alpha",
            "texture_shadow_output_uv",
            "texture_shadow_alpha_bilinear",
            "texture_shadow_confidence_enabled",
            "texture_shadow_confidence_zero_dc",
            "texture_shadow_spatial_resolution",
            "texture_shadow_spatial_power2",
            "texture_freeze_gaussian_densify",
            "texture_rtg_freeze_gaussian_densify",
            "texture_tor_enabled",
            "texture_oob_enabled",
            "native_resume_load_optimizer",
            "cam_opt",
            "pl_opt",
        }
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):        #检查参数是否支持简写，即使用'_'+首字母作为简写
                shorthand = True
                key = key[1:]              #去掉下划线
            t = shadow_sensitivity_anchor if key == "texture_shadow_sensitivity_anchor" else type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:   #如果参数是布尔类型，则添加一个store_true的action，此外由于key已经规范化，所以可以直接使用key[0:1]
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if key in explicit_bool_pairs:
                        group.add_argument("--" + key, dest=key, default=value, action="store_true")
                        group.add_argument("--no_" + key, dest=key, action="store_false")
                    else:
                        group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()   #创建一个空对象，用于存储修改后的参数
        for arg in vars(args).items():  #vars(args) 返回的是 args 的属性和它们的值组成的字典    #items() 以列表返回可遍历的(键, 值) 元组数组
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):   #vars(self) 返回的是 self 的全部属性和它们的值组成的字典,因此检测当前属性名是否在 self 对象的属性中
                setattr(group, arg[0], arg[1])  #super().__setattr__ 只能操作当前对象（self）的属性 # 用 arg[1] 覆盖 self 各个属性的原始值
        return group #返回修改后的对象

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):

        # 渲染器
        self.rasterizer = "2dgs"

        # 源路径
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        # 调整图片分辨率，默认 -1，表示自动调整过大图片
        self._resolution = 1
        # 是否使用白背景，默认 False
        self._white_background = False
        # 图片储存设备，默认 cuda
        self.data_device = "cuda"
        self.hdr_image_cache_dir = ""
        # 是否评估 测试集，默认评估，如果为 False，则不评估，并且测试集和训练集合并
        self.eval = True
        # 当前输入 image 是否为 hdr 格式，默认 False
        self.hdr = False
        # 是否使用神经相位函数，默认 True
        self.use_nerual_phasefunc = True
        # 输入图片数量最大值，默认 2000
        self.view_num = 2000
        # MLP parameter
        self.phasefunc_hidden_size = 32
        self.phasefunc_hidden_layers = 3
        # encoding frequency
        self.phasefunc_frequency = 4
        # latent size
        self.neural_material_size = 6
        self.shadow_view_independent = False
        self.texture_specular_response = "legacy"
        # basis angular Gaussian num
        self.basis_asg_num = 8
        # optimize cam and pl or not
        self.cam_opt= True
        self.pl_opt= True
        # maximum gaussian number
        self.maximum_gs = 550_000
        self.force_native_2dgs_core = False
        self.native_resume_load_optimizer = False
        self.use_textures = False
        self.texture_resolution = 4
        self.texture_sigma_factor = 3.0
        self.texture_dynamic_resolution = False
        self.texture_min_resolution = 4
        self.texture_max_resolution = 64
        self.texture_effect_mode = "uvshadow_specular_lobe"
        self.texture_normal_scale = 0.35
        self.mbrdf_normal_source = "local_q"
        self.texture_shadow_confidence_enabled = False
        self.texture_shadow_confidence_start_iter = 30_000
        self.texture_shadow_confidence_strength = 1.0
        self.texture_shadow_confidence_gamma = 1.0
        self.texture_shadow_confidence_zero_dc = False
        self.texture_shadow_transport_mode = "additive"
        self.texture_shadow_transport_eps = 1e-4
        self.texture_shadow_sensitivity_anchor = "no_shadow"
        self.texture_shadow_spatial_resolution = False
        self.texture_shadow_spatial_texel_size = 0.01
        self.texture_shadow_spatial_min_resolution = 4
        self.texture_shadow_spatial_max_resolution = 32
        self.texture_shadow_spatial_power2 = True
        self.texture_shadow_hole_fill = "none"
        self.texture_shadow_hole_fill_chunk = 16
        self.texture_factor_surgery = "none"
        self.texture_factor_surgery_seed = 0


        """
        mine
        """
        # save time for debug
        self.wang_debug = False
        self.asg_channel_num = 1
        self.asg_mlp = False
        self.alpha_change = False
        self.asg_alpha_num = 1
        self.use_hgs_finetune = False
        self.load_num = 400

        # hgs 相关
        self.use_hgs = False
        self.gamma_change = False
        
        # offset
        self.offset = 0.015
        self.detach_shadow = False
        self.use_shadow_refine_mlp = True
        self.shadow_backward_stage_enabled = False
        self.shadow_backward_xyz_from_iter = 5_000
        self.shadow_backward_opacity_from_iter = 5_000
        self.shadow_backward_scaling_from_iter = 22_000
        self.shadow_backward_rotation_from_iter = 40_000
        self.shadow_scale_reg_enabled = False
        self.shadow_scale_reg_from_iter = 30_000
        self.shadow_scale_min_axis_floor = 1e-6
        self.shadow_scale_floor_weight = 0.0
        self.shadow_scale_anisotropy_thresh = 20.0
        self.shadow_scale_anisotropy_weight = 0.0
        

        super().__init__(parser, "Loading Parameters", sentinel)

    # 提取出经过命令行参数修改后的参数
    def extract(self, args):
        g = super().extract(args)

        # os.path.abspath() 的参数为空字符串 ""，它会返回当前工作目录的绝对路径。
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.non_trans = 0
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        self.texture_render_use_alpha = False
        self.texture_shadow_use_alpha = False
        self.texture_shadow_output_uv = True
        self.texture_shadow_alpha_bilinear = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 100_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        
        self.kd_lr = 0.01
        self.ks_lr = 0.01
        
        # only use diffuse term
        self.spcular_freeze_step = 9_000
        # gradually change to linear if trained under hdr mode (for a stable initialization)
        self.fit_linear_step = 7_000
        # ASG is initialized as aniso and freezed for a quick convergence 在早期训练时，冻结 asg 参数，快速收敛
        self.asg_freeze_step = 22000
        
        # ASG lr
        self.asg_lr_freeze_step = 40_000
        self.asg_lr_init = 0.01
        self.asg_lr_final = 0.0001
        self.asg_lr_delay_mult = 0.01
        self.asg_lr_max_steps = 50_000
        
        # local frame lr
        self.local_q_lr_freeze_step = 40_000
        self.local_q_lr_init = 0.01
        self.local_q_lr_final = 0.0001
        self.local_q_lr_delay_mult = 0.01
        self.local_q_lr_max_steps = 50_000
        
        # latent and MLP lr
        self.freeze_phasefunc_steps = 50_000
        self.neural_phasefunc_lr_init = 0.001
        self.neural_phasefunc_lr_final = 0.00001
        self.neural_phasefunc_lr_delay_mult = 0.01
        self.neural_phasefunc_lr_max_steps = 50_000
        
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3_000
        self.densify_from_iter = 500
        self.densify_until_iter = 80_000
        self.densify_grad_threshold = 0.0002
        # Optional clone-only densification control. 0 keeps legacy behavior:
        # clone and split both use densify_grad_threshold.
        self.densify_clone_grad_threshold = 0.0
        self.random_background = False

        # cam and pl opt lr:
        self.train_cam_freeze_step = 5_000
        self.opt_cam_lr_init = 0.001
        self.opt_cam_lr_final = 0.00001
        self.opt_cam_lr_delay_step = 20_000
        self.opt_cam_lr_delay_mult = 0.2
        self.opt_cam_lr_max_steps = 80_000

        self.train_pl_freeze_step = 15000
        self.opt_pl_lr_init = 0.001
        self.opt_pl_lr_final = 0.00005
        self.opt_pl_lr_delay_step = 30_000
        self.opt_pl_lr_delay_mult = 0.1
        self.opt_pl_lr_max_steps = 80_000

        """
        mine
        """
        self.mlp_zero = False
        self.asg_mlp_freeze = 40000
        self.asg_change_freeze = 40000

        self.bigsize_threshold = 0.1
        self.texture_lr = 0.0025
        self.texture_specular_lr_scale = 1.0
        self.texture_normal_lr_scale = 1.0
        self.texture_start_iter = 0
        self.texture_clone_threshold_scale = 1.0
        self.texture_freeze_gaussian_densify = False
        self.texture_rtg_enabled = False
        self.texture_rtg_refine_from_iter = 30_000
        self.texture_rtg_refine_until_iter = 100_000
        self.texture_rtg_refine_interval = 1_000
        self.texture_rtg_refine_fraction = 0.0
        self.texture_rtg_ema = 0.9
        self.texture_rtg_alpha_weight = 0.0
        self.texture_rtg_min_score = 1e-6
        self.texture_rtg_score_margin = 1.0
        self.texture_rtg_gate_reference_resolution = 4.0
        self.texture_rtg_resolution_gamma = 1.0
        self.texture_rtg_chunk_texels = 262_144
        self.texture_rtg_freeze_gaussian_densify = False
        self.texture_rtg_optimizer_state_scale = 0.5
        self.texture_rtg_resize_mode = "activated_seeded_bilinear_exact"
        self.texture_rtg_dry_run = False
        self.texture_rtg_stability_windows = 1
        # Optional sub-window stability gate. When enabled, RTG is probed every
        # texture_rtg_probe_interval iterations, but resize still only happens
        # on texture_rtg_refine_interval. A chart is refined only if it passes
        # the gate in at least texture_rtg_required_probe_windows sub-windows.
        # 0 keeps the legacy per-refine-window behavior.
        self.texture_rtg_probe_interval = 0
        self.texture_rtg_required_probe_windows = 0
        self.texture_rtd_enabled = False
        self.texture_rtd_compress_from_iter = 85_000
        self.texture_rtd_compress_until_iter = 95_000
        self.texture_rtd_compress_interval = 5_000
        self.texture_rtd_min_resolution = 1
        self.texture_rtd_step_mode = "step"
        self.texture_rtd_error_threshold = 0.65
        self.texture_rtd_error_threshold_4to3 = 0.65
        self.texture_rtd_error_threshold_3to2 = 0.45
        self.texture_rtd_error_threshold_2to1 = 0.30
        self.texture_rtd_kd_weight = 1.0
        self.texture_rtd_shadow_weight = 1.0
        self.texture_rtd_specular_weight = 1.0
        self.texture_rtd_normal_weight = 0.0
        self.texture_rtd_max_fraction = 0.25
        self.texture_rtd_shadow_sample_interval = 1
        self.texture_rtd_dry_run = False
        self.texture_rmd_enabled = False
        self.texture_rmd_start_iter = 110_000
        self.texture_rmd_end_iter = 120_000
        self.texture_rmd_weight = 1e-4
        self.texture_rmd_kd_weight = 1.0
        self.texture_rmd_specular_weight = 0.5
        self.texture_rmd_target_resolution = 2
        self.texture_tor_enabled = False
        self.texture_tor_start_iter = 30_000
        self.texture_tor_gate_floor = 0.05
        self.texture_tor_gate_scale = 1.0
        self.texture_tor_strength = 1.0
        self.texture_oob_enabled = False
        self.texture_oob_start_iter = 30_000
        self.texture_oob_weight = 0.0
        self.texture_oob_gate_scale = 1.0
        self.texture_oob_gate_power = 1.0
        self.texture_oob_target_ratio = 0.0
        
        super().__init__(parser, "Optimization Parameters")

def explicit_cli_value(args, name, argv=None):
    flag = f"--{name}"
    for token in (sys.argv[1:] if argv is None else argv):
        if token == flag or token.startswith(flag + "="):
            return getattr(args, name)
    return None


def read_model_config(path):
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    try:
        node = ast.parse(source, mode="eval").body
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "Namespace" and not node.args
                and all(item.arg is not None for item in node.keywords)):
            raise ValueError("expected Namespace with literal keyword values")
        return Namespace(**{item.arg: ast.literal_eval(item.value) for item in node.keywords})
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid model configuration: {path}") from exc


def get_combined_args(parser: ArgumentParser):
    command = sys.argv[1:]
    cli = parser.parse_args(command)
    if not cli.model_path:
        parser.error("--model_path/-m is required")
    config_path = os.path.join(cli.model_path, "cfg_args")
    saved = read_model_config(config_path)
    # Existing values survive defaults; explicitly supplied CLI options win.
    return parser.parse_args(command, namespace=saved)
