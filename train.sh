#!/usr/bin/env bash
set -euo pipefail

export OPENCV_IO_ENABLE_OPENEXR=1

data_root="/path/to/data"
output_root="./output"

python train.py -s "$data_root/Synthetic/Hotdog" \
                -m "$output_root/Hotdog" \
                --hdr \
                --white_background \
                --data_device cpu \
                --view_num 2000 \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 50000 \
                --densify_until_iter 50000 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --rasterizer 2dgs \
                --sh_degree 0 \
                --resolution 1 \
                --use_textures \
                --texture_resolution 4 \
                --texture_start_iter 0 \
                --texture_effect_mode uvshadow_specular_lobe \
                --texture_shadow_transport_mode anchored_range_alpha \
                --texture_shadow_sensitivity_anchor no_shadow \
                --texture_specular_lr_scale 1.0 \
                --texture_normal_lr_scale 1.0 \
                --eval

python train.py -s "$data_root/NRHints/Pikachu" \
                -m "$output_root/Pikachu" \
                --data_device cpu \
                --view_num 2000 \
                --iterations 100000 \
                --asg_freeze_step 22000 \
                --spcular_freeze_step 9000 \
                --fit_linear_step 7000 \
                --asg_lr_freeze_step 40000 \
                --asg_lr_max_steps 50000 \
                --asg_lr_init 0.01 \
                --asg_lr_final 0.0001 \
                --local_q_lr_freeze_step 40000 \
                --local_q_lr_init 0.01 \
                --local_q_lr_final 0.0001 \
                --local_q_lr_max_steps 50000 \
                --neural_phasefunc_lr_init 0.001 \
                --neural_phasefunc_lr_final 0.00001 \
                --freeze_phasefunc_steps 50000 \
                --neural_phasefunc_lr_max_steps 50000 \
                --position_lr_max_steps 70000 \
                --densify_until_iter 80000 \
                --densify_grad_threshold 0.00015 \
                --test_iterations 2000 7000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --save_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --checkpoint_iterations 7000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
                --unfreeze_iterations 5000 \
                --use_nerual_phasefunc \
                --cam_opt \
                --pl_opt \
                --rasterizer 2dgs \
                --sh_degree 0 \
                --resolution 1 \
                --use_textures \
                --texture_resolution 4 \
                --texture_start_iter 0 \
                --texture_effect_mode uvshadow_specular_lobe \
                --texture_shadow_transport_mode anchored_range_alpha \
                --texture_shadow_sensitivity_anchor no_shadow \
                --texture_specular_lr_scale 1.0 \
                --texture_normal_lr_scale 1.0 \
                --eval
