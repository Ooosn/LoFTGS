import torch
from scene.gaussian_model_native_2dgs import GaussianModel as _TextureGaussianModel2DGS
from utils.shadow_transport import shadow_sensitivity_anchor
from utils.native_backend import validate_native_backend


class _NativeTextureAdapter(_TextureGaussianModel2DGS):
    def __init__(self, modelset, opt=None):
        super().__init__(
            sh_degree=getattr(modelset, "sh_degree", 0),
            use_textures=getattr(modelset, "use_textures", False),
            texture_resolution=getattr(modelset, "texture_resolution", 4),
            texture_dynamic_resolution=getattr(modelset, "texture_dynamic_resolution", False),
            texture_min_resolution=getattr(modelset, "texture_min_resolution", getattr(modelset, "texture_resolution", 4)),
            texture_max_resolution=getattr(modelset, "texture_max_resolution", 64),
            use_mbrdf=getattr(modelset, "use_nerual_phasefunc", False),
            basis_asg_num=getattr(modelset, "basis_asg_num", 8),
            hidden_feature_size=getattr(modelset, "phasefunc_hidden_size", 32),
            hidden_feature_layers=getattr(modelset, "phasefunc_hidden_layers", 3),
            phase_frequency=getattr(modelset, "phasefunc_frequency", 4),
            neural_material_size=getattr(modelset, "neural_material_size", 6),
            asg_channel_num=getattr(modelset, "asg_channel_num", 1),
            asg_mlp=getattr(modelset, "asg_mlp", False),
            asg_alpha_num=getattr(modelset, "asg_alpha_num", 1),
            mbrdf_normal_source=getattr(modelset, "mbrdf_normal_source", "local_q"),
        )
        self.rasterizer = getattr(modelset, "rasterizer", "2dgs")
        self.use_MBRDF = self.use_mbrdf
        self.maximum_gs = getattr(modelset, "maximum_gs", 550_000)
        self.light_direction = torch.zeros(3, dtype=torch.float32, device="cuda")
        self.texture_sigma_factor = float(getattr(modelset, "texture_sigma_factor", 3.0))
        self.texture_effect_mode = str(getattr(modelset, "texture_effect_mode", "uvshadow_specular_lobe"))
        self.texture_specular_response = str(getattr(modelset, "texture_specular_response", "legacy"))
        self._validate_specular_response()
        self.texture_normal_scale = float(getattr(modelset, "texture_normal_scale", 0.35))
        self.mbrdf_normal_source = str(getattr(modelset, "mbrdf_normal_source", "local_q")).lower()
        self.texture_shadow_confidence_enabled = bool(getattr(modelset, "texture_shadow_confidence_enabled", False))
        self.texture_shadow_confidence_start_iter = int(getattr(modelset, "texture_shadow_confidence_start_iter", 30_000))
        self.texture_shadow_confidence_strength = float(getattr(modelset, "texture_shadow_confidence_strength", 1.0))
        self.texture_shadow_confidence_gamma = float(getattr(modelset, "texture_shadow_confidence_gamma", 1.0))
        self.texture_shadow_confidence_zero_dc = bool(getattr(modelset, "texture_shadow_confidence_zero_dc", False))
        self.texture_shadow_transport_mode = str(getattr(modelset, "texture_shadow_transport_mode", "additive")).lower()
        self.texture_shadow_transport_eps = float(getattr(modelset, "texture_shadow_transport_eps", 1e-4))
        self.texture_shadow_sensitivity_anchor = shadow_sensitivity_anchor(
            getattr(modelset, "texture_shadow_sensitivity_anchor", None) or "no_shadow")
        self.shadow_view_independent = bool(getattr(modelset, "shadow_view_independent", False))
        self.texture_shadow_spatial_resolution = bool(getattr(modelset, "texture_shadow_spatial_resolution", False))
        self.texture_shadow_spatial_texel_size = float(getattr(modelset, "texture_shadow_spatial_texel_size", 0.01))
        self.texture_shadow_spatial_min_resolution = int(getattr(modelset, "texture_shadow_spatial_min_resolution", 4))
        self.texture_shadow_spatial_max_resolution = int(getattr(modelset, "texture_shadow_spatial_max_resolution", 32))
        self.texture_shadow_spatial_power2 = bool(getattr(modelset, "texture_shadow_spatial_power2", True))
        self.texture_shadow_hole_fill = str(getattr(modelset, "texture_shadow_hole_fill", "none")).lower()
        self.texture_shadow_hole_fill_chunk = int(getattr(modelset, "texture_shadow_hole_fill_chunk", 16))
        self.texture_factor_surgery = str(getattr(modelset, "texture_factor_surgery", "none")).lower()
        self.texture_factor_surgery_seed = int(getattr(modelset, "texture_factor_surgery_seed", 0))
        self.gs2dgs_backend = "native"

    def update_learning_rate(self, iteration, asg_freeze_step=0, local_q_freeze_step=0, freeze_phasefunc_steps=0):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group["lr"] = self.xyz_scheduler_args(iteration)
            elif param_group["name"] in {"alpha_asg", "asg_sigma", "asg_rotation", "asg_scales"}:
                param_group["lr"] = self.asg_scheduler_args(max(0, iteration - asg_freeze_step))
            elif param_group["name"] == "tex_specular":
                param_group["lr"] = self._texture_specular_lr(iteration, asg_freeze_step)
            elif param_group["name"] == "local_q":
                if str(getattr(self, "mbrdf_normal_source", "local_q")) == "2dgs":
                    param_group["lr"] = 0.0
                else:
                    param_group["lr"] = self._local_q_lr(iteration, local_q_freeze_step)
            elif param_group["name"] == "tex_normal":
                param_group["lr"] = self._texture_normal_lr(iteration, local_q_freeze_step)
            elif param_group["name"] in {"neural_phasefunc", "neural_material"}:
                param_group["lr"] = self.neural_phasefunc_scheduler_args(max(0, iteration - freeze_phasefunc_steps))

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        bigsize_threshold=None,
        crop_extent=None,
        clone_grad_threshold=None,
    ):
        return super().densify_and_prune(
            max_grad,
            min_opacity,
            extent,
            max_screen_size,
            bigsize_threshold=bigsize_threshold,
            crop_extent=crop_extent,
            clone_grad_threshold=clone_grad_threshold,
        )

    def add_densification_stats(self, viewspace_point_tensor, update_filter, image_width=None, image_height=None, out_weight=None):
        return super().add_densification_stats(viewspace_point_tensor, update_filter)

    def change_alpha_asg(self, alpha_asg):
        if not self.use_mbrdf:
            return
        alpha_asg_3 = alpha_asg.repeat(1, 1, 3)
        optimizable = self.replace_tensor_to_optimizer(alpha_asg_3, "alpha_asg")
        self.alpha_asg = optimizable["alpha_asg"]
        self.asg_alpha_num = 3

    def reset_local_q(self, temp):
        if not self.use_mbrdf:
            return
        local_q_new = temp * 0.6 + self.local_q * 0.4
        optimizable = self.replace_tensor_to_optimizer(local_q_new, "local_q")
        self.local_q = optimizable["local_q"]


class GaussianModel2DGSAdapter:
    """Use the packaged native core, with or without local textures."""

    def __new__(cls, modelset, opt=None):
        validate_native_backend(modelset)
        return _NativeTextureAdapter(modelset, opt)
