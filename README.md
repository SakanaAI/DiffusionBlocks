# DiffusionBlocks (ICLR 2026)

<div align="center">
<img alt="overview" src="./overview.jpg" title="overview">
</div>

> We propose ***DiffusionBlocks***, a principled framework that partitions transformers into independently trainable blocks, reducing memory requirements proportionally while maintaining competitive performance across diverse architectures and tasks.

This is an official implementation of *[DiffusionBlocks](https://arxiv.org/abs/2506.14202)* on image classification using Vision Transformers (ViT).

## Installation

Please install [uv](https://docs.astral.sh/uv/getting-started/installation/). Then, run:

```bash
# Install dependencies
uv sync

# make sure to login huggingface and wandb
uv run huggingface-cli login
uv run wandb login
```

We conducted our experiments in the following environment: Python Version 3.12 and CUDA Version 12.2 H100.

## Training

The model checkpoints are saved in `logs` folder.

**Baseline (ViT):**

```bash
uv run main.py train cifar100 --model_type vit
```

**DiffusionBlocks:**

```bash
uv run main.py train cifar100 --model_type dblock
```

* **NOTE:** the total epochs in DiffusionBlocks is multiplied by the number of blocks to align the total number of iterations with the baseline as one step in DiffusionBlocks corresponds to training for one block.

<details>

In the base setting, we don't reply on techniques such as heavy data augmentation. In case you want to see the performance with heavy data augmentation and learning rate scheduler, run as follows:

**Baseline (ViT):**

```bash
BATCH_SIZE=128
EPOCHS=1000
POSTFIX="-rand-augment"
WARMUP_STEPS=3900
MODEL_TYPE="dblock"
srun uv run main.py train cifar100 \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE --num_epochs $EPOCHS --postfix=$POSTFIX \
    --scheduler_type cosine_with_min_lr --num_warmup_steps $WARMUP_STEPS --lr 5e-4 \
    --scheduler_specific_kwargs '{"min_lr": 5e-5}' \
    --add_rand_aug
```

**DiffusionBlocks:**

```bash
BATCH_SIZE=128
EPOCHS=1000
POSTFIX="-rand-augment"
WARMUP_STEPS=$((3900 * 3)) # 3 indicates the number of blocks
MODEL_TYPE="dblock"
srun uv run main.py train cifar100 \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE --num_epochs $EPOCHS --postfix=$POSTFIX \
    --scheduler_type cosine_with_min_lr --num_warmup_steps $WARMUP_STEPS --lr 5e-4 \
    --scheduler_specific_kwargs '{"min_lr": 5e-5}' \
    --add_rand_aug
```

</details>

## Evaluation

**Baseline (ViT):**

```bash
CKPT_PATH="logs/path-to-last.ckpt"
uv run main.py test cifar100 --model_type vit --ckpt_path $CKPT
```

**DiffusionBlocks:**

```bash
CKPT_PATH="logs/path-to-last.ckpt"
uv run main.py test cifar100 --model_type dblock --ckpt_path $CKPT
```

## Generation (CIFAR-10, DiT-S/2)

Class-conditional image generation with a DiT-S/2 (12 layers, 384 hidden, 6 heads, patch 2) on CIFAR-10
pixels, trained with EDM. `--num_blocks 1` is the end-to-end baseline and `--num_blocks 3` is
DiffusionBlocks; both use the same code path. As for classification, DiffusionBlocks runs
`num_epochs × num_blocks` epochs so that each block gets as many updates as the baseline.

**Baseline (DiT, B=1):**

```bash
uv run main.py train cifar10 --model_type dit --num_blocks 1 \
    --batch_size 64 --accumulate_grad_batches 8 --lr 1e-4 --num_epochs 100 \
    --cfg_scale 2.0 --class_dropout_prob 0.1 --num_inference_steps 50 \
    --eval_batch_size 100
```

**DiffusionBlocks (B=3):**

```bash
uv run main.py train cifar10 --model_type dit --num_blocks 3 --gamma 0.05 \
    --batch_size 64 --accumulate_grad_batches 8 --lr 1e-4 --num_epochs 100 \
    --cfg_scale 2.0 --class_dropout_prob 0.1 --num_inference_steps 50 \
    --eval_batch_size 100
```

The settings follow the paper (App. E.2): batch 512 (here 64 × 8 accumulation; a micro-batch of 128 does not fit B=1 on a 19 GB Mac), AdamW with lr 1e-4,
100 epochs, CFG scale 2.0 with 10% label dropout, and 50 Euler steps. Sampling uses
`DiTDBlockModel.sample(labels)`.

**Evaluation (FID and sample grid):** as for classification, the test stage runs after training, or
on a checkpoint with:

```bash
uv run main.py test cifar10 --model_type dit --num_blocks 3 --cfg_scale 2.0 \
    --num_inference_steps 50 --eval_batch_size 100 --ckpt_path logs/path-to-last.ckpt
```

It generates one image per CIFAR-10 test label (10k, 1,000 per class) and logs FID computed with
[clean-fid](https://github.com/GaParmar/clean-fid)'s pipeline against its precomputed CIFAR-10 statistics:
`test/fid_train` in `legacy_tensorflow` mode (closest to the ADM suite the paper uses for train FID; its
Inception runs on the CPU, as it fails on MPS) and `test/fid_test` in `clean` mode, as in the paper.
These FIDs use 10k samples per evaluation, not the paper's 50k, so they are biased upward by a few points
and are not directly comparable to its Table 2. For variance, evaluate with `--seed 0`, `1` and `2` and report
the mean ± std and the minimum (the paper reports the minimum of 3). Two kinds of file are written to
`logs/<run>/`: `grid.png`, a 10×10 grid (one row per class, fixed noise so models can be compared), and
`results_seed<k>.json`, one per evaluation seed, with the FIDs and the run's config and metadata.

**Heun sampler:** `--sampler heun` replaces the 50 Euler steps with Heun's second-order method,
which evaluates the network twice per step. With `--num_inference_steps 25` it uses 49 network
evaluations against Euler's 50. Both evaluations of a step must run the same block, so no step may cross
a block boundary. The equi-probability schedule contains the boundaries as points when its number of
steps is a multiple of the number of blocks (25 points: 8 steps per block for B=3), and the state at a
boundary passes unchanged to the next block. Other step counts are rejected. Results are written to
`grid_heun.png` and `results_heun_seed<k>.json`, next to the Euler ones:

```bash
uv run main.py test cifar10 --model_type dit --num_blocks 3 --cfg_scale 2.0 --sampler heun \
    --num_inference_steps 25 --eval_batch_size 100 --ckpt_path logs/path-to-last.ckpt
```

**Cost benchmark (B=1 vs B=3):** FLOPs per generated image, training memory for one optimizer step and
sampling throughput, on randomly initialized models (no checkpoint needed):

```bash
uv run python -m scripts.benchmark --num_blocks 1 3
```

It logs progress, one result line per model and the gain of B=3 over B=1, and writes `logs/benchmark.json`. Training memory is
counted from the tensors a step keeps alive (weights, gradients, AdamW state, activations); the MPS
allocator's reading is reported next to it as a cross-check.

## Development

```bash
uv run pytest             # unit tests (CPU, a few seconds)
uv run pyright            # type check
uv run ruff check .       # lint
uv run ruff format .      # format
```

The same checks run in CI (`.github/workflows/validate.yml`).

## Acknowledgement

The implementation of Vision Transformer in [vit.py](./vit.py) is based on [HuggingFace Transformers](https://github.com/huggingface/transformers). And, the implementation of EDM is based on [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models).  
We are grateful for their work.

## Citation

To cite our work, please use the following BibTeX:

```bibtex
@inproceedings{shing2026diffusionblocks,
  title     = {DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation},
  author.   = {Makoto Shing and Masanori Koyama and Takuya Akiba},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=pwVSmK71cS}
}
```
