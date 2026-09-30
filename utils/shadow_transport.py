"""Shadow-anchor formulas and checkpoint metadata, independent of CUDA backends."""

SHADOW_SENSITIVITY_ANCHORS = ("no_shadow", "visibility")


def shadow_sensitivity_anchor(value):
    anchor = str(value).lower()
    if anchor not in SHADOW_SENSITIVITY_ANCHORS:
        raise ValueError(
            "Unknown texture_shadow_sensitivity_anchor: %r. Expected one of %s."
            % (value, ", ".join(SHADOW_SENSITIVITY_ANCHORS)))
    return anchor


def checkpoint_shadow_anchor(model_params):
    if isinstance(model_params, (tuple, list)) and len(model_params) > 32:
        extra = model_params[32]
        if isinstance(extra, dict) and extra.get("texture_shadow_sensitivity_anchor"):
            return shadow_sensitivity_anchor(extra["texture_shadow_sensitivity_anchor"])
    return None


def apply_sensitivity_anchor(composed, uv, sensitivity, gau):
    anchor = shadow_sensitivity_anchor(
        getattr(gau, "texture_shadow_sensitivity_anchor", "no_shadow"))
    if anchor == "no_shadow":
        return 1.0 + sensitivity * (composed - 1.0)
    return uv + sensitivity * (composed - uv)
