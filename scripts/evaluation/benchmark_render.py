#!/usr/bin/env python3
"""Time native textured rendering with a TCNN-only neural phase."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_NAMES = (
    "diff_gaussian_rasterization",
    "diff_gaussian_rasterization_light",
    "diff_gaussian_rasterization_hgs",
    "v_3dgs",
    "v_3dgs_ortho",
    "diff_surfel_rasterization_shadow",
    "surfel_texture",
    "surfel_texture_deferred",
    "simple_knn",
)
CUDA_MODULE_PREFIXES = (
    "diff_gaussian_rasterization",
    "diff_surfel_rasterization",
    "surfel_texture",
    "simple_knn",
    "tinycudann",
    "gsplat_cuda",
)


def _parser():
    from argparse import ArgumentParser
    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    optimization = OptimizationParams(parser)
    parser.add_argument("--load_iteration", type=int, default=100000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--views", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--label", default="Ours")
    parser.add_argument("--scene-label")
    parser.add_argument("--paper-row")
    parser.add_argument("--factor-surgery", choices=("none", "no_gain", "no_lobe", "no_specular_factors"))
    parser.add_argument("--expected-specular-replacement", choices=("none", "normal_only", "asg_mix_only"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-result", type=Path)
    parser.add_argument("--native-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--stage-profiler", type=Path)
    parser.add_argument("--profile-stages", action="store_true")
    return parser, model, pipeline, optimization


def _configure_imports(native_root: Path):
    native_root = native_root.resolve()
    if not native_root.is_dir():
        raise RuntimeError(f"Native extension root is missing: {native_root}")
    paths = [REPO_ROOT]
    paths.extend(native_root / "submodules" / name for name in EXTENSION_NAMES)
    for path in reversed(paths):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.pop("GS3_ALLOW_TORCH_PHASE_FALLBACK", None)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def _backend(module):
    return f"{type(module).__module__}.{type(module).__name__}"


def _require_tcnn():
    try:
        importlib.import_module("tinycudann")
    except ImportError as exc:
        raise ImportError("Render timing requires tinycudann; PyTorch fallback is training-only.") from exc


def _replace_phase_with_tcnn(gaussians, seed):
    import torch
    from scene.neural_phase_function import Neural_phase

    source = gaussians.neural_phasefunc
    kwargs = {
        "hidden_feature_size": int(source.hidden_feature_size),
        "hidden_feature_layers": int(source.hidden_feature_layers),
        "frequency": int(source.frequency),
        "neural_material_size": int(source.neural_material_size),
        "shadow_output_dim": int(source.shadow_output_dim),
        "shadow_view_independent": bool(source.shadow_view_independent),
    }
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    phase = Neural_phase(**kwargs).to(gaussians.get_xyz.device).eval()
    for parameter in phase.parameters():
        parameter.requires_grad_(False)
    backends = {
        "encoding": _backend(phase.encoding),
        "shadow_func": _backend(phase.shadow_func),
        "other_effects_func": _backend(phase.other_effects_func),
    }
    if not all(value.startswith("tinycudann.") for value in backends.values()):
        raise RuntimeError(f"Benchmark requires TCNN for every neural module; got {backends}")
    gaussians.neural_phasefunc = phase
    return kwargs, backends


def _selected_views(views, requested, expected_names=None):
    by_name = {view.image_name: view for view in views}
    all_names = sorted(by_name, key=lambda name: int(name.rsplit("_", 1)[-1]))
    if expected_names is not None:
        if not set(expected_names).issubset(by_name):
            raise RuntimeError("Current test split does not contain the reference view names")
        names = list(expected_names)
    else:
        import numpy as np

        indices = np.linspace(0, len(all_names) - 1, min(requested, len(all_names))).round().astype(int)
        names = [all_names[index] for index in indices]
    return [by_name[name] for name in names]


def _frame_name(frame):
    image_path = frame.get("img_path", frame.get("file_path", ""))
    return Path(str(image_path).replace("\\", "/")).stem


def _select_test_names(transforms_path, requested):
    import numpy as np

    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    names = sorted({_frame_name(frame) for frame in data["frames"]},
                   key=lambda name: int(name.rsplit("_", 1)[-1]))
    indices = np.linspace(0, len(names) - 1, min(requested, len(names))).round().astype(int)
    return [names[index] for index in indices]


def _install_camera_subset(names):
    original_load = json.load

    def load(stream, *args, **kwargs):
        data = original_load(stream, *args, **kwargs)
        filename = str(getattr(stream, "name", "")).replace("\\", "/")
        if not isinstance(data, dict) or "frames" not in data:
            return data
        if filename.endswith("transforms_test.json"):
            by_name = {_frame_name(frame): frame for frame in data["frames"]}
            missing = set(names) - set(by_name)
            if missing:
                raise RuntimeError(f"Selected test frames are missing: {sorted(missing)}")
            data["frames"] = [by_name[name] for name in names]
        elif filename.endswith("transforms_train.json"):
            data["frames"] = data["frames"][:2]
        return data

    json.load = load
    return original_load


def _cuda_module_paths():
    modules = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and name.startswith(CUDA_MODULE_PREFIXES):
            modules[name] = str(Path(path).resolve())
    return modules


def _memory(device):
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device_used_mb": (total_bytes - free_bytes) / 2**20,
        "device_free_mb": free_bytes / 2**20,
        "device_total_mb": total_bytes / 2**20,
        "process_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
        "process_reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
    }


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--native-root", type=Path, default=REPO_ROOT)
    boot_args, _ = bootstrap.parse_known_args()
    _configure_imports(boot_args.native_root)
    parser, model_parser, pipeline_parser, optimization_parser = _parser()
    from arguments import get_combined_args
    from utils.system_utils import searchForMaxIteration

    args = get_combined_args(parser)
    if args.load_iteration == -1:
        args.load_iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    if args.views < 1 or args.epochs < 1 or args.repeats < 1:
        raise ValueError("views, epochs, and repeats must be positive")
    if args.memory_fraction is not None and not 0 < args.memory_fraction <= 1:
        raise ValueError("--memory-fraction must be in (0, 1]")
    _require_tcnn()
    if args.profile_stages and (args.stage_profiler is None or not args.stage_profiler.is_file()):
        raise FileNotFoundError("--profile-stages requires an existing --stage-profiler file")
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from rich import print as rich_print
    from utils.general_utils import safe_state

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for render timing")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.cuda.current_device()
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    device_before = _memory(device)

    modelset = model_parser.extract(args)
    pipeline = pipeline_parser.extract(args)
    opt = optimization_parser.extract(args)
    model_root = Path(modelset.model_path)
    checkpoint = model_root / f"chkpnt{args.load_iteration}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    transforms_path = model_root / "point_cloud" / f"iteration_{args.load_iteration}" / "transforms_test.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(transforms_path)
    expected = None
    if args.expected_result:
        expected = json.loads(args.expected_result.read_text(encoding="utf-8"))
        if expected["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise RuntimeError("Checkpoint hash differs from the selected reference cohort")
    if expected:
        frame_names = expected["frame_names"][:args.views]
    else:
        frame_names = _select_test_names(transforms_path, args.views)
    camera_manifest = json.loads((model_root / "cameras.json").read_text(encoding="utf-8"))
    native_dimensions = {(int(item["width"]), int(item["height"])) for item in camera_manifest}
    if len(native_dimensions) != 1:
        raise RuntimeError(f"Model camera manifest has mixed dimensions: {native_dimensions}")
    manifest_width, manifest_height = next(iter(native_dimensions))
    if expected and [manifest_width, manifest_height] != [expected["width"], expected["height"]]:
        raise RuntimeError("Camera manifest resolution differs from the reference cohort")
    safe_state(args.quiet)

    import render as renderer
    from gaussian_renderer.inference import TextureInferenceState

    record = {}

    def timed_set(current_modelset, name, iteration, views, gaussians, current_pipeline,
                  background, gamma, hdr, write_images=False, force_save=False,
                  synthesize_video=False, shadowmap_render=False, fast_inference=False):
        if name != "test" or iteration != args.load_iteration:
            raise RuntimeError(f"Unexpected render split/iteration: {name}/{iteration}")
        if str(getattr(gaussians, "rasterizer", "")) != "2dgs" or not getattr(gaussians, "use_textures", False):
            raise RuntimeError("TCNN render benchmark requires the textured native 2DGS path")
        if not getattr(gaussians, "use_MBRDF", False):
            raise RuntimeError("TCNN render benchmark requires the deferred material path")
        replacement_mode = str(getattr(gaussians, "texture_specular_replacement", "none"))
        if args.expected_specular_replacement is not None and replacement_mode != args.expected_specular_replacement:
            raise RuntimeError(
                f"Expected specular replacement {args.expected_specular_replacement!r}, got {replacement_mode!r}"
            )
        if args.factor_surgery is not None:
            gaussians.texture_factor_surgery = args.factor_surgery
            gaussians.texture_factor_surgery_seed = 0
        cameras = _selected_views(views, args.views, frame_names)
        dimensions = {(int(view.image_width), int(view.image_height)) for view in cameras}
        if len(dimensions) != 1:
            raise RuntimeError(f"Test views do not share native resolution: {dimensions}")
        width, height = next(iter(dimensions))
        if expected and [width, height] != [expected["width"], expected["height"]]:
            raise RuntimeError("Native resolution differs from the reference cohort")

        tcnn_config, backend = _replace_phase_with_tcnn(gaussians, args.seed)
        light_stream, calc_stream = renderer.create_render_streams()
        if light_stream is None or calc_stream is None or light_stream.cuda_stream == calc_stream.cuda_stream:
            raise RuntimeError("Expected distinct light-space and material CUDA streams")
        state = TextureInferenceState()
        with torch.no_grad():
            state.prepare(gaussians)
        state.wait_ready(light_stream)
        state.wait_ready(calc_stream)
        render_model = state.get_prepared()
        local_axes = render_model.get_local_axis
        asg_scales = render_model.asg_func.get_asg_lam_miu
        asg_axes = render_model.asg_func.get_asg_axis
        render_args = {
            "modelset": current_modelset,
            "iteration": iteration,
            "inference_state": state,
        }

        def render_package(view):
            return renderer.render(
                view, gaussians, light_stream, calc_stream, local_axes, asg_scales,
                asg_axes, current_pipeline, background, **render_args,
            )

        def render_one(view):
            package = render_package(view)
            return package["render"] * package["shadow"] + package["other_effects"]

        with torch.no_grad():
            warmup = render_one(cameras[0])
            if tuple(warmup.shape) != (3, height, width) or not torch.isfinite(warmup).all().item():
                raise RuntimeError("TCNN smoke render produced an invalid RGB tensor")
            del warmup
            for view in cameras[1:]:
                image = render_one(view)
                del image
        torch.cuda.synchronize(device)

        baseline = _memory(device)
        torch.cuda.reset_peak_memory_stats(device)
        durations = []
        event_seconds = []
        frames_per_repeat = len(cameras) * args.epochs
        for repeat in range(args.repeats):
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            wall_start = time.perf_counter()
            with torch.no_grad():
                for _ in range(args.epochs):
                    for view in cameras:
                        image = render_one(view)
                del image
            end_event.record()
            torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - wall_start)
            event_seconds.append(start_event.elapsed_time(end_event) / 1000.0)
            print(f"TIMED {args.label} {repeat + 1}/{args.repeats} wall={durations[-1]:.6f}s", flush=True)

        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
        median_seconds = statistics.median(durations)
        record.update({
            "status": "timed",
            "method": args.label,
            "scene": expected.get("scene") if expected else (args.scene_label or args.label),
            "paper_row": args.paper_row or args.label,
            "iteration": int(iteration),
            "frame_names": [view.image_name for view in cameras],
            "width": width,
            "height": height,
            "gaussians": int(gaussians.get_xyz.shape[0]),
            "model": str(model_root),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "network_policy": "fresh fixed-seed TCNN phase; checkpoint weights intentionally not loaded for timing-only measurement",
            "specular_replacement": replacement_mode,
            "factor_surgery": str(getattr(gaussians, "texture_factor_surgery", "none")),
            "factor_surgery_seed": int(getattr(gaussians, "texture_factor_surgery_seed", 0)),
            "tcnn_seed": args.seed,
            "tcnn_config": tcnn_config,
            "phase_backend": backend,
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "streams": {"light": light_stream.cuda_stream, "calc": calc_stream.cuda_stream,
                        "current": torch.cuda.current_stream(device).cuda_stream},
            "views": len(cameras),
            "epochs": args.epochs,
            "repeats": args.repeats,
            "timed_frames_per_repeat": frames_per_repeat,
            "wall_seconds": durations,
            "cuda_event_seconds": event_seconds,
            "fps": frames_per_repeat / median_seconds,
            "fps_repeats": [frames_per_repeat / value for value in durations],
            "memory_scope": "current benchmark process only; excludes other WDDM processes",
            "device_memory_before_process_load_mb": device_before["device_used_mb"],
            "device_memory_before_timing_mb": baseline["device_used_mb"],
            "process_allocated_before_timing_mb": baseline["process_allocated_mb"],
            "process_reserved_before_timing_mb": baseline["process_reserved_mb"],
            "peak_allocated_mb": peak_allocated,
            "peak_reserved_mb": peak_reserved,
            "incremental_peak_allocated_mb": peak_allocated - baseline["process_allocated_mb"],
            "incremental_peak_reserved_mb": peak_reserved - baseline["process_reserved_mb"],
            "native_extensions": _cuda_module_paths(),
            "hdr": bool(args.hdr),
            "gamma": False,
            "opt_pose": True,
            "render_source_sha256": {
                name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                for name in ("render.py", "gaussian_renderer/__init__.py",
                             "gaussian_renderer/texture_branch.py", "gaussian_renderer/inference.py")
            },
        })
        if args.profile_stages:
            branch = importlib.import_module("gaussian_renderer.texture_branch")
            profiler_spec = importlib.util.spec_from_file_location("paired_stage_profile", args.stage_profiler)
            if profiler_spec is None or profiler_spec.loader is None:
                raise RuntimeError(f"Cannot load stage profiler: {args.stage_profiler}")
            profiler_module = importlib.util.module_from_spec(profiler_spec)
            sys.path.insert(0, str(args.stage_profiler.parent.resolve()))
            profiler_spec.loader.exec_module(profiler_module)
            profile_stages = profiler_module.profile_stages
            profile_stages(renderer, branch, gaussians, cameras, render_package,
                           args.output.parent if args.output else Path.cwd(), "Ours-TCNN")
            record["stage_profile"] = "paired_stages.json"

    renderer.render_set = timed_set
    import cv2
    import numpy as np
    from PIL import Image

    original_json_load = _install_camera_subset(frame_names)
    original_image_open = Image.open
    original_imread = cv2.imread

    def placeholder_image_open(*args, **kwargs):
        return Image.new("RGBA", (manifest_width, manifest_height), (0, 0, 0, 255))

    def placeholder_imread(*args, **kwargs):
        image = np.zeros((manifest_height, manifest_width, 4), dtype=np.float32)
        image[:, :, 3] = 1.0
        return image

    Image.open = placeholder_image_open
    cv2.imread = placeholder_imread
    try:
        renderer.render_sets(
            modelset, args.load_iteration, pipeline,
            skip_train=True, skip_test=False, opt_pose=True,
            gamma=False, hdr=args.hdr, valid=False, write_images=False,
            force_save=False, synthesize_video=False, shadowmap_render=False,
            shadow_transport_override=None, opt=opt, fast_inference=True,
        )
    finally:
        json.load = original_json_load
        Image.open = original_image_open
        cv2.imread = original_imread
    if record.get("status") != "timed":
        raise RuntimeError("Renderer did not invoke the benchmark test callback")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("BENCHMARK_JSON " + json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
