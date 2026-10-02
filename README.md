# LoFT-GS

Factorized Local Transport for Relightable Gaussian Splatting.

[Project page](https://ooosn.github.io/LoFTGS/) | [Paper](https://ooosn.github.io/LoFTGS/assets/paper/LoFT-GS.pdf)

## Install

Install PyTorch for your CUDA version, then:

```
pip install setuptools wheel
pip install --no-build-isolation -r requirements.txt
pip install --no-build-isolation "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```

## Train

Set `data_root` and `output_root` at the top of `train.sh`, then:

```
bash train.sh
```

In training, if `tinycudann` is not installed, add
`export GS3_ALLOW_TORCH_PHASE_FALLBACK=1` at the top of `train.sh` to use PyTorch.
Without this switch, missing `tinycudann` raises an error. Speed tests require
`tinycudann` for reproducibility.

The release builds only four local CUDA extensions: `simple-knn`,
`surfel-texture`, `surfel-texture-deferred`, and
`diff-surfel-rasterization-shadow`. The shared GLM headers in
`submodules/third_party` are required by these builds.
Legacy HGS, 3DGS/gsplat renderers, and lifted 3DGS shadows are not included.
Their checkpoints require their original source revision. Native 2DGS,
untextured initialization, and the textured LoFT-GS path remain supported.

The default is 1-anchor (`no_shadow`). U-anchor can also work well on some scenes;
to use it, change `--texture_shadow_sensitivity_anchor no_shadow` to
`--texture_shadow_sensitivity_anchor visibility` in `train.sh`.

## Render

```
python render.py -m <model_dir> \
    --load_iteration 100000 \
    --skip_train \
    --opt_pose \
    --hdr \
    --gamma \
    --write_images \
    --force_save
```

For NRHints scenes, omit both `--hdr` and `--gamma`.

## Render speed

```
python scripts/evaluation/benchmark_render.py -m <model_dir> \
    --views 32 --repeats 3
```

## Image metrics

All metrics reported in the paper are computed from rendered images, not from training logs.

```
python scripts/evaluation/unified_image_metrics.py \
    --renders <render_dir> --gt <gt_dir> --out <output_dir>
```
