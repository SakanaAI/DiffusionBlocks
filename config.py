"""Run configuration: CLI enums and the typed `RunConfig` built from argparse (defaults live only here)."""

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from typing import Any

from transformers.trainer_utils import SchedulerType


class Stage(StrEnum):
    TRAIN = "train"
    TEST = "test"


class DatasetName(StrEnum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    TINY_IMAGENET = "tiny-imagenet"


class ModelType(StrEnum):
    VIT = "vit"
    DBLOCK = "dblock"
    DIT = "dit"


class SamplerType(StrEnum):
    EULER = "euler"
    HEUN = "heun"


@dataclass(frozen=True)
class RunConfig:
    stage: Stage
    data_name: DatasetName
    model_type: ModelType = ModelType.VIT
    devices: int = 1
    num_epochs: int = 500
    batch_size: int = 128
    add_rand_aug: bool = False
    eval_batch_size: int | None = None
    save_every_n_epochs: int = 5
    accumulate_grad_batches: int = 1
    gradient_checkpointing: bool = False
    scheduler_type: SchedulerType = SchedulerType.CONSTANT_WITH_WARMUP
    scheduler_specific_kwargs: dict[str, Any] | None = None
    lr: float = 0.001
    weight_decay: float = 0.01
    optimizer: str = "adamw"  # unused by the repo: AdamW is hard-coded
    num_warmup_steps: int = 0
    deepspeed: bool = False  # unused by the repo
    seed: int = 42
    save_top_k: int = 1
    num_workers: int = 8
    ckpt_path: str | None = None
    debug: bool = False
    postfix: str = ""
    # DiffusionBlocks
    num_blocks: int = 3
    gamma: float = 0.05
    num_inference_steps: int | None = None
    sampler: SamplerType = SamplerType.EULER
    cfg_scale: float = 0.0
    class_dropout_prob: float = 0.0
    # set from the DataModule in main.py
    image_size: int | None = None
    num_labels: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain-value dict for loggers and checkpoint hparams (enums become strings)."""
        return {
            k: v.value if isinstance(v, Enum) else v for k, v in asdict(self).items()
        }


def build_parser() -> argparse.ArgumentParser:
    # SUPPRESS: flags not given are left out, so RunConfig's defaults apply.
    p = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    p.add_argument("stage", type=Stage, choices=list(Stage))
    p.add_argument("data_name", type=DatasetName, choices=list(DatasetName))
    p.add_argument("--model_type", type=ModelType, choices=list(ModelType))
    p.add_argument("--scheduler_type", type=SchedulerType)
    p.add_argument(
        "--sampler",
        type=SamplerType,
        choices=list(SamplerType),
        help="ODE sampler for generation (--model_type dit only)",
    )
    p.add_argument(
        "--scheduler_specific_kwargs",
        type=json.loads,
        help="specific kwargs for the scheduler",
    )
    p.add_argument("--postfix", type=str, help="postfix for the experiment name")
    for name in ("add_rand_aug", "gradient_checkpointing", "deepspeed", "debug"):
        p.add_argument(f"--{name}", action="store_true")
    typed = {
        "devices": int,
        "num_epochs": int,
        "batch_size": int,
        "eval_batch_size": int,
        "save_every_n_epochs": int,
        "accumulate_grad_batches": int,
        "lr": float,
        "weight_decay": float,
        "optimizer": str,
        "num_warmup_steps": int,
        "seed": int,
        "save_top_k": int,
        "num_workers": int,
        "ckpt_path": str,
        "num_blocks": int,
        "gamma": float,
        "num_inference_steps": int,
        "cfg_scale": float,
        "class_dropout_prob": float,
    }
    for name, type_ in typed.items():
        p.add_argument(f"--{name}", type=type_)
    return p


def parse_args(argv: list[str] | None = None) -> RunConfig:
    return RunConfig(**vars(build_parser().parse_args(argv)))
