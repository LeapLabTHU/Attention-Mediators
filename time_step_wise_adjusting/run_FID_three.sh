# debug
torchrun --nnodes=1 --nproc_per_node=1 exps/e05v03_sit/sample_ddp_three.py SDE \
--model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32 \
--agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.9 --switch_ratio2 0.5 \
--first_several_num 3 --distence_func mse \
--sample_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/debug/e05v03_sit/Test_ThreeModel_Agent4_16_64_Change0x9_0x5_FID_cfg1x0/samples/ \
--ckpt1 /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt \
--ckpt2 /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt \
--ckpt3 /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt;



# 0x9
# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x8_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.9 --switch_ratio2 0.8
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x8_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x8_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x7_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.9 --switch_ratio2 0.7
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x7_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x7_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x6_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.9 --switch_ratio2 0.6
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x6_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x6_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x5_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.9 --switch_ratio2 0.5
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x5_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x5_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x4_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.9 --switch_ratio2 0.4
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x4_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x4_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x3_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.9 --switch_ratio2 0.3
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x3_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=3 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x3_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x2_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.9 --switch_ratio2 0.2
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x2_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x2_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x1_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.9 --switch_ratio2 0.1
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x1_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x1_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz
























# 0x8
# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x7_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.8 --switch_ratio2 0.7
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x7_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x7_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x6_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.8 --switch_ratio2 0.6
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x6_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x6_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x5_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.8 --switch_ratio2 0.5
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x5_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x5_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x4_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.8 --switch_ratio2 0.4
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x4_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=3 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x4_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x3_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.8 --switch_ratio2 0.3
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x3_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x3_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x2_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.8 --switch_ratio2 0.2
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x2_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x2_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x1_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.8 --switch_ratio2 0.1
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x1_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x1_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz

























# 0x7
# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x6_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.7 --switch_ratio2 0.6
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x6_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x6_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x5_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.7 --switch_ratio2 0.5
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x5_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x5_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x4_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.7 --switch_ratio2 0.4
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x4_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x4_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x3_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.7 --switch_ratio2 0.3
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x3_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=3 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x3_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x2_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.7 --switch_ratio2 0.2
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x2_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x2_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x1_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.7 --switch_ratio2 0.1
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x1_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x1_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz
























# 0x6
# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x5_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.6 --switch_ratio2 0.5
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x5_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x5_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x4_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.6 --switch_ratio2 0.4
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x4_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x4_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x3_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.6 --switch_ratio2 0.3
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x3_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x3_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x2_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.6 --switch_ratio2 0.2
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x2_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=3 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x2_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x1_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.6 --switch_ratio2 0.1
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x1_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x1_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz

























# 0x5
# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x4_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.5 --switch_ratio2 0.4
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x4_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x4_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x3_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.5 --switch_ratio2 0.3
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x3_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x3_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz



# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x2_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.5 --switch_ratio2 0.2
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x2_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=3 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x2_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz



# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x1_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.5 --switch_ratio2 0.1
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x1_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x1_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz
























# 0x4
# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x3_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.4 --switch_ratio2 0.3
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x3_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x3_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz
torchrun --nnodes=1 --nproc_per_node=4 exps/e05v03_sit/sample_ddp_three.py SDE \
--model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32 \
--agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.4 --switch_ratio2 0.3 \
--first_several_num 3 --distence_func mse \
--sample_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x3_FID_cfg1x0/samples2/ \
--ckpt1 /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt \
--ckpt2 /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt \
--ckpt3 /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt;
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x3_FID_cfg1x0/samples2/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz

# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x2_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.4 --switch_ratio2 0.2
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x2_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x2_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x1_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.4 --switch_ratio2 0.1
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x1_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=3 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x1_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz
























# 0x3
# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x3_0x2_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.3 --switch_ratio2 0.2
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x3_0x2_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x3_0x2_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x3_0x1_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.3 --switch_ratio2 0.1
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x3_0x1_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=3 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x3_0x1_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz

torchrun --nnodes=1 --nproc_per_node=4 exps/e05v03_sit/sample_ddp_three.py SDE \
--model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 128 \
--agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.3 --switch_ratio2 0.1 \
--first_several_num 3 --distence_func mse \
--sample_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x3_0x1_FID_cfg1x0/samples/ \
--ckpt1 /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt \
--ckpt2 /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt \
--ckpt3 /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt;






















# 0x2
# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x2_0x1_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_two.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.2 --switch_ratio2 0.1
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x2_0x1_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x2_0x1_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz




















# extra scripts

# switch_ratio1 = 1x0
# # e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x9_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 1.0 --switch_ratio2 0.9
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x9_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x9_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x9_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x8_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 1.0 --switch_ratio2 0.8
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x8_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x8_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x8_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x7_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 1.0 --switch_ratio2 0.7
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x7_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x7_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x7_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x6_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 1.0 --switch_ratio2 0.6
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x6_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x6_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x6_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x5_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 1.0 --switch_ratio2 0.5
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x5_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x5_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x5_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x4_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 1.0 --switch_ratio2 0.4
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x4_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x4_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x4_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x3_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 1.0 --switch_ratio2 0.3
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x3_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x3_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x3_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x2_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 1.0 --switch_ratio2 0.2
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x2_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x2_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=3 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x2_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x1_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 1.0 --switch_ratio2 0.1
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x1_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x1_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x1_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz





















# switch_ratio2 = 0x0

# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x1_0x0_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.1 --switch_ratio2 0.0
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x1_0x0_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x1_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x1_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x2_0x0_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.2 --switch_ratio2 0.0
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x2_0x0_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x2_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=3 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x2_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x3_0x0_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.3 --switch_ratio2 0.0
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x3_0x0_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x3_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x3_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x0_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.4 --switch_ratio2 0.0
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x0_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/;
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x4_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x0_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.5 --switch_ratio2 0.0
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x0_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x5_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x0_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.6 --switch_ratio2 0.0
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x0_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x6_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x0_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.7 --switch_ratio2 0.0
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x0_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/
CUDA_VISIBLE_DEVICES=3 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x7_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x0_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.8 --switch_ratio2 0.0
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x0_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x8_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x0_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 0.9 --switch_ratio2 0.0
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x0_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/
CUDA_VISIBLE_DEVICES=1 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_0x9_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04.npz


# # e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x0_FID_cfg1x0
      - torchrun --nnodes=1 --nproc_per_node=8 exps/e05v03_sit/sample_ddp_three.py SDE
        --model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 32
        --agent_num1 4 --agent_num2 16 --agent_num3 64 --switch_ratio1 1.0 --switch_ratio2 0.0
        --first_several_num 3 --distence_func mse
        --sample_dir /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x0_FID_cfg1x0/samples/
        --ckpt1 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_014-SiT-S-2-Linear-velocity-None-AgentNum4/checkpoints/0400000.pt
        --ckpt2 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_007-SiT-S-2-Linear-velocity-None-AgentNum16/checkpoints/0400000.pt
        --ckpt3 /mnt/openseg_blob/puyifan/work_dirs/linear_dit/e01v01_sit/e01v01_008-SiT-S-2-Linear-velocity-None-AgentNum64/checkpoints/0400000.pt
CUDA_VISIBLE_DEVICES=0 python exps/e05v03_sit/create_npz.py SDE --num_fid_samples 50_000 \
--sample_folder_dir /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-128-SDE-250-Euler-sigma-Mean-0.04/
CUDA_VISIBLE_DEVICES=2 \
python tools/evaluator.py /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/official_ckpts/VIRTUAL_imagenet256_labeled.npz \
  /home/pyf/openseg_blob/puyifan/work_dirs/linear_dit/e05v03_sit/e05v03_SiTS2_3Model_Agent4_16_64_Change_1x0_0x0_FID_cfg1x0/samples/SiT-S-2-0400000-cfg-1.0-32-SDE-250-Euler-sigma-Mean-0.04.npz

