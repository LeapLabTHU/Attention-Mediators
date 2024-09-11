# Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators (ECCV 2024)

[Yifan Pu][]* , [Zhuofan Xia][]* , [Jiayi Guo][]* , [Dongchen Han][], Qixiu Li, Duo Li, [Yuhui Yuan], Ji Li, [Yizeng Han][], [Shiji Song][], [Gao Huang][], Xiu Li.

*: Equal contribution.



## Introduction

This repository contains the implementation of the ECCV 2024 paper, *Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators* [[arxiv]].

## Usage

### Dependencies

```bash
# create conda environment
conda create --name mediator python=3.10 -y
conda activate mediator
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install accelerate diffusers timm wandb
pip install torchdiffeq yacs einops termcolor
pip install xformers
```

### Scripts

- Train a SiT-S/2 with attention mediators from scratch with 256 resolution:

```bash
# number of mediator = 4
torchrun --nnodes=1 --nproc_per_node=4 attention_mediator/train.py --model SiT-S/2 \
--epochs 80 --image_size 256 --global_batch_size 256 --mediator_num 4 \
--data_path YOUR_DATA_PATH --results_dir YOUR_SAVE_PATH \
--wandb --wandb_name YOUR_EXP_NAME --wandb_entity YOUR_EXP_ENTITY --wandb_project YOUR_EXP_PROJ --wandb_key YOUR_EXP_KEY;

# number of mediator = 16
torchrun --nnodes=1 --nproc_per_node=4 attention_mediator/train.py --model SiT-S/2 \
--epochs 80 --image_size 256 --global_batch_size 256 --mediator_num 16 \
--data_path YOUR_DATA_PATH --results_dir YOUR_SAVE_PATH \
--wandb --wandb_name YOUR_EXP_NAME --wandb_entity YOUR_EXP_ENTITY --wandb_project YOUR_EXP_PROJ --wandb_key YOUR_EXP_KEY;

# number of mediator = 64
torchrun --nnodes=1 --nproc_per_node=4 attention_mediator/train.py --model SiT-S/2 \
--epochs 80 --image_size 256 --global_batch_size 256 --mediator_num 64 \
--data_path YOUR_DATA_PATH --results_dir YOUR_SAVE_PATH \
--wandb --wandb_name YOUR_EXP_NAME --wandb_entity YOUR_EXP_ENTITY --wandb_project YOUR_EXP_PROJ --wandb_key YOUR_EXP_KEY;
```

- Finetune a low resolution checkpoint to high resolution

```bash
# number of mediator = 64, load 256 resolution ckpt, finetune 512 resolution
torchrun --nnodes=1 --nproc_per_node=4 attention_mediator/train_256to512.py --model SiT-S/2 \
--epochs 20 --epoch_ckpt_every 2 --image_size 512 --global_batch_size 64 --mediator_num 64 \
--data_path YOUR_DATA_PATH --results_dir YOUR_SAVE_PATH --resume_ckpt_low_resolution YOUR_PRETRAINED_256_CKPT_PATH \
--wandb --wandb_name YOUR_EXP_NAME --wandb_entity YOUR_EXP_ENTITY --wandb_project YOUR_EXP_PROJ --wandb_key YOUR_EXP_KEY;

# number of mediator = 64, load 512 resolution ckpt, finetune 1024 resolution
torchrun --nnodes=1 --nproc_per_node=4 attention_mediator/train_512to1024.py --model SiT-S/2 \
--epochs 5 --epoch_ckpt_every 2 --image_size 1024 --global_batch_size 16 --mediator_num 64 \
--data_path YOUR_DATA_PATH --results_dir YOUR_SAVE_PATH --resume_ckpt_low_resolution YOUR_PRETRAINED_512_CKPT_PATH \
--wandb --wandb_name YOUR_EXP_NAME --wandb_entity YOUR_EXP_ENTITY --wandb_project YOUR_EXP_PROJ --wandb_key YOUR_EXP_KEY;
```

- Evaluation for a static number of mediator:

```bash
torchrun --nnodes=1 --nproc_per_node=4 attention_mediator/sample_ddp.py SDE \
--model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 \
--sample_dir YOUR_SAVE_PATH --ckpt YOUR_CKPT_PATH --mediator_num YOUR_CKPT_MEDIATOR_NUM;
```

- Evaluation with timestep-wise mediator adjusting:

```bash
torchrun --nnodes=1 --nproc_per_node=4 time_step_wise_adjusting/sample_ddp_three.py SDE \
--model SiT-S/2 --num_fid_samples 50_000 --image_size 256 --cfg_scale 1.0 --per_proc_batch_size 1 \
--first_several_num 3 --distence_func mae \
--sample_dir YOUR_SAVE_PATH \
--mediator_num1 YOUR_CKPT_MEDIATOR_NUM_1 --mediator_num2 YOUR_CKPT_MEDIATOR_NUM_2 --mediator_num3 YOUR_CKPT_MEDIATOR_NUM_3 \
--ckpt1 YOUR_MEDIATOR_NUM_1_CKPT --ckpt2 YOUR_MEDIATOR_NUM_2_CKPT --ckpt3 YOUR_MEDIATOR_NUM_3_CKPT \
--switch_ratio1 YOUR_SWITCH_RATIO_1 --switch_ratio2 YOUR_SWITCH_RATIO_2;
```

## Ackowledgements

We use the pytorch implementation of [SiT][]  in our experiments. Thanks for their neat code.


[Yifan Pu]: https://yifanpu001.github.io/
[Zhuofan Xia]: https://www.zhuofanxia.xyz/
[Jiayi Guo]: https://www.jiayiguo.net/
[Dongchen Han]: https://scholar.google.com/citations?user=wv3U3tkAAAAJ&hl=en/
[Yizeng Han]: https://yizenghan.top/
[Yuhui Yuan]: https://www.microsoft.com/en-us/research/people/yuyua/
[Shiji Song]: https://scholar.google.com/citations?user=rw6vWdcAAAAJ&hl=en
[Gao Huang]: http://www.gaohuang.net/

[arxiv]: https://arxiv.org/pdf/2408.05710
[SiT]: https://github.com/willisma/SiT