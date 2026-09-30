"""Explicit, render-session-scoped caches for a frozen textured Gaussian model."""
import torch


_PROPERTIES = (
    "get_scaling", "get_rotation", "get_opacity", "get_local_axis", "get_ks",
    "get_alpha_asg", "get_texture_color", "get_texture_alpha",
    "get_texture_specular_gain", "get_texture_specular_lobe_scale",
)
_CONFIG = (
    "texture_effect_mode", "texture_resolution", "texture_dynamic_resolution",
    "texture_sigma_factor", "mbrdf_normal_source", "active_sh_degree",
    "texture_specular_response", "texture_specular_replacement",
    "texture_factor_surgery", "texture_factor_surgery_seed",
    "_dynamic_texture_layout_version",
)


def _tensor_signature(tensor):
    return (id(tensor), tensor.data_ptr(), tensor._version, tuple(tensor.shape),
            tensor.dtype, tensor.device)


def _model_signature(model):
    tensors = [(name, _tensor_signature(value)) for name, value in vars(model).items()
               if isinstance(value, torch.Tensor)]
    asg = model.asg_func
    tensors.extend(("asg." + name, _tensor_signature(value))
                   for name, value in list(asg.named_parameters()) + list(asg.named_buffers()))
    return (id(model), id(asg), tuple(sorted(tensors)),
            tuple(getattr(model, name, None) for name in _CONFIG))


class _FrozenView:
    def __init__(self, source, values):
        self._source = source
        self._values = values

    def __getattr__(self, name):
        if name in self._values:
            return self._values[name]
        return getattr(self._source, name)

    def __call__(self, *args, **kwargs):
        return self._source(*args, **kwargs)


class TextureInferenceState:
    """Reuse activated constants, never view/light/shadow or rendered values.

    Construct one state per render session and pass it explicitly to render().
    Normal tensor edits, replacement, device/dtype and layout changes invalidate
    it automatically. Call clear() after unsupported external .data mutations.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self._signature = None
        self._view = None
        self._ready = None

    def prepare(self, model):
        if torch.is_grad_enabled():
            raise RuntimeError("TextureInferenceState requires torch.no_grad()")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Use torch.no_grad(), not inference_mode(), for versioned texture caches")
        signature = _model_signature(model)
        if signature != self._signature:
            values = {name: getattr(model, name).detach() for name in _PROPERTIES}
            values["asg_func"] = _FrozenView(model.asg_func, {
                "get_asg_lam_miu": model.asg_func.get_asg_lam_miu.detach(),
                "get_asg_axis": model.asg_func.get_asg_axis.detach(),
            })
            self._view = _FrozenView(model, values)
            self._signature = signature
            self._ready = None
            if model.get_xyz.is_cuda:
                self._ready = torch.cuda.Event()
                self._ready.record()
        return self._view

    def get_prepared(self):
        if self._view is None:
            raise RuntimeError("TextureInferenceState must be prepared before rendering")
        return self._view

    def wait_ready(self, stream):
        if self._ready is not None:
            (stream if stream is not None else torch.cuda.current_stream()).wait_event(self._ready)
