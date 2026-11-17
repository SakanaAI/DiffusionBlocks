"""Benchmark B=1 vs B=3 DiT-S/2: FLOPs per image, training memory, sampling speed.

The weights are random (cost does not depend on their values); the models differ
only in `num_blocks`.   uv run python -m scripts.benchmark --num_blocks 1 3
"""

import argparse
import gc
import json
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightning as L
import structlog
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from config import DatasetName, ModelType, RunConfig, SamplerType, Stage
from dit import DiTLayer
from model import DiTDBlockModel

MiB, GFLOP = 1024**2, 1e9
log = structlog.get_logger()


@dataclass(frozen=True)
class Flops:
    """GFLOPs to generate one image."""

    total: float
    layers: float  # the transformer layers only


@dataclass(frozen=True)
class StepMemory:
    """Bytes one training step keeps alive."""

    activations: int  # saved for backward, parameters excluded
    gradients: int
    allocator: int  # the allocator's highest reading (0 where there is none)


@dataclass(frozen=True)
class TrainingMemory:
    """MiB of a training step, the worst over the measured steps. The allocator peak
    is only a cross-check: it includes pooled buffers."""

    total: float
    weights: float
    gradients: float
    adamw: float
    activations: float
    allocator_peak: float | None


@dataclass(frozen=True)
class Throughput:
    images_per_s: float
    std: float


@dataclass(frozen=True)
class Result:
    num_blocks: int
    flops: Flops
    memory: TrainingMemory
    throughput: Throughput

    def metrics(self) -> dict[str, float]:
        """Every measured value under one name, e.g. `memory_total`."""
        parts = {"flops": self.flops, "memory": self.memory, "speed": self.throughput}
        return {
            f"{part}_{name}": value
            for part, values in parts.items()
            for name, value in asdict(values).items()
            if value is not None
        }


def rounded(values: dict[str, Any]) -> dict[str, float]:
    """Four significant digits, for readable log lines (the JSON keeps full values)."""
    return {k: float(f"{v:.4g}") for k, v in values.items() if v is not None}


def sync() -> None:
    """Wait for queued accelerator kernels, so clocks and memory readings see
    finished work (nothing to wait for on a CPU-only machine)."""
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


def allocated_bytes(device: torch.device) -> int:
    """The allocator's bytes in use (0 where the device has no such reading)."""
    sync()
    return torch.mps.current_allocated_memory() if device.type == "mps" else 0


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def build_model(num_blocks: int, args: argparse.Namespace) -> DiTDBlockModel:
    cfg = RunConfig(
        Stage.TRAIN,
        DatasetName.CIFAR10,
        ModelType.DIT,
        image_size=32,
        num_labels=10,
        num_blocks=num_blocks,
        num_inference_steps=args.num_inference_steps,
        sampler=args.sampler,
        cfg_scale=args.cfg_scale,
        class_dropout_prob=0.1,  # the paper's
    )
    model = DiTDBlockModel(cfg)
    model.configure_model()
    return model


def measure_flops(model: DiTDBlockModel) -> Flops:
    """FLOPs to generate one image. Math attention is forced: the counter misses
    fused SDPA."""
    model.eval()
    with (
        FlopCounterMode(depth=2, display=False) as flops,
        sdpa_kernel(SDPBackend.MATH),
    ):
        model.sample(torch.zeros(1, dtype=torch.long, device=model.device))
    layers = flops.get_flop_counts()[DiTLayer.__name__]
    return Flops(
        total=flops.get_total_flops() / GFLOP, layers=sum(layers.values()) / GFLOP
    )


def nbytes(tensors: Iterable[torch.Tensor]) -> int:
    """Bytes of the distinct storages behind `tensors` (views share one)."""
    storages = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors
    }
    return sum(storages.values())


def adamw_with_state(model: DiTDBlockModel) -> torch.optim.AdamW:
    """AdamW with its moments already created for every parameter (a real step only
    creates them for the block it trains). Learning rate 0: we need its memory, not
    its updates, so the weights stay as initialized."""
    params = list(model.model.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.0)
    for p in params:
        p.grad = torch.zeros_like(p)
    optimizer.step()
    return optimizer


def training_step(
    model: DiTDBlockModel, optimizer: torch.optim.AdamW, batch: dict[str, torch.Tensor]
) -> StepMemory:
    params = optimizer.param_groups[0]["params"]
    weights = {p.untyped_storage().data_ptr() for p in params}
    saved: list[torch.Tensor] = []

    def keep(tensor: torch.Tensor) -> torch.Tensor:
        saved.append(tensor)
        return tensor

    optimizer.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(keep, lambda t: t):
        loss, _ = model.shared_step(batch)
    after_forward = allocated_bytes(model.device)
    activations = nbytes(
        t for t in saved if t.untyped_storage().data_ptr() not in weights
    )
    saved.clear()  # autograd keeps what it needs; do not extend the lifetimes
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    optimizer.step()
    return StepMemory(
        activations=activations,
        gradients=nbytes(grads),
        allocator=max(after_forward, allocated_bytes(model.device)),
    )


def measure_training_memory(
    model: DiTDBlockModel, batch_size: int, steps: int
) -> TrainingMemory:
    """Memory of a training step, counted from the tensors it keeps alive."""
    model.train()
    optimizer = adamw_with_state(model)
    params = optimizer.param_groups[0]["params"]
    size = (batch_size, 3, model.image_size, model.image_size)
    batch = {
        "pixel_values": torch.randn(size, device=model.device).clamp(-1, 1),
        "labels": torch.randint(model.num_labels, (batch_size,), device=model.device),
    }
    free_memory(model.device)
    runs = [training_step(model, optimizer, batch) for _ in range(steps)]
    model.zero_grad(set_to_none=True)
    # a step trains one random block, so keep the worst step
    activations = max(r.activations for r in runs) / MiB
    gradients = max(r.gradients for r in runs) / MiB
    allocator = max(r.allocator for r in runs) / MiB
    state = (t for s in optimizer.state.values() for t in s.values())
    weights = nbytes(params) / MiB
    adamw = nbytes(t for t in state if torch.is_tensor(t)) / MiB
    return TrainingMemory(
        total=weights + gradients + adamw + activations,
        weights=weights,
        gradients=gradients,
        adamw=adamw,
        activations=activations,
        allocator_peak=allocator or None,
    )


def measure_throughput(
    models: dict[int, DiTDBlockModel], batch_size: int, repeats: int
) -> dict[int, Throughput]:
    """Images/s of full sampling runs. The models alternate, so drift hits them all."""
    first = next(iter(models.values()))
    labels = (torch.arange(batch_size) % first.num_labels).to(first.device)
    for num_blocks, model in models.items():
        log.info("warm-up sampling run (untimed)", blocks=num_blocks)
        model.eval()
        model.sample(labels)  # untimed: MPS compiles its kernels on first use
    rates: dict[int, list[float]] = {b: [] for b in models}
    for repeat in range(1, repeats + 1):
        for num_blocks, model in models.items():
            sync()
            start = time.perf_counter()
            model.sample(labels)
            sync()
            rates[num_blocks].append(batch_size / (time.perf_counter() - start))
        latest = {f"B{b}_images_per_s": r[-1] for b, r in rates.items()}
        log.info("throughput repeat", repeat=f"{repeat}/{repeats}", **rounded(latest))
    return {
        b: Throughput(
            images_per_s=statistics.fmean(r),
            std=statistics.stdev(r) if len(r) > 1 else 0.0,
        )
        for b, r in rates.items()
    }


def run(args: argparse.Namespace, device: torch.device) -> list[Result]:
    costs: dict[int, tuple[Flops, TrainingMemory]] = {}
    models: dict[int, DiTDBlockModel] = {}
    for num_blocks in args.num_blocks:
        L.seed_everything(args.seed, verbose=False)
        model = build_model(num_blocks, args)
        steps = model.num_inference_steps
        log.info("counting FLOPs of one image", blocks=num_blocks, steps=steps)
        flops = measure_flops(model)  # on the CPU: device-independent
        log.info("FLOPs counted", blocks=num_blocks, **rounded(asdict(flops)))
        model.to(device)
        log.info(
            "measuring training memory",
            blocks=num_blocks,
            steps=args.memory_steps,
            batch_size=args.train_batch_size,
        )
        memory = measure_training_memory(
            model, args.train_batch_size, args.memory_steps
        )
        log.info(
            "training memory measured", blocks=num_blocks, **rounded(asdict(memory))
        )
        costs[num_blocks] = (flops, memory)
        models[num_blocks] = model.cpu()  # keep the next model's reading clean
        free_memory(device)
    for model in models.values():
        model.to(device)
    log.info(
        "measuring sampling throughput",
        batch_size=args.sample_batch_size,
        repeats=args.repeats,
    )
    speeds = measure_throughput(models, args.sample_batch_size, args.repeats)
    return [Result(b, *costs[b], speeds[b]) for b in models]


def gains(results: list[Result]) -> dict[str, float]:
    """How many times better the model with the most blocks is than the one with
    the fewest (throughput is higher-is-better, every other metric lower)."""
    by_blocks = sorted(results, key=lambda r: r.num_blocks)
    fewest, most = by_blocks[0].metrics(), by_blocks[-1].metrics()
    return {
        k: most[k] / fewest[k] if k == "speed_images_per_s" else fewest[k] / most[k]
        for k in fewest
        if k in most and k != "speed_std"
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("--num_blocks", type=int, nargs="+", default=[1, 3])
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--sample_batch_size", type=int, default=100)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument(
        "--sampler",
        type=SamplerType,
        choices=list(SamplerType),
        default=SamplerType.EULER,
    )
    p.add_argument("--cfg_scale", type=float, default=2.0)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--memory_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("logs/benchmark.json"))
    args = p.parse_args()
    if len(set(args.num_blocks)) != len(args.num_blocks):
        p.error("--num_blocks must not repeat a value")
    return args


def main() -> None:
    args = parse_args()
    device = torch.accelerator.current_accelerator() or torch.device("cpu")
    log.info("benchmark started", device=str(device), torch=torch.__version__)
    results = run(args, device)
    gain = gains(results)
    blocks = sorted(r.num_blocks for r in results)
    for result in results:
        log.info("result", blocks=result.num_blocks, **rounded(result.metrics()))
    log.info(f"gain of B={blocks[-1]} over B={blocks[0]} (×)", **rounded(gain))
    payload = {
        "results": [asdict(r) for r in results],
        "gains": gain,
        "settings": vars(args),
        "device": str(device),
        "torch": torch.__version__,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    log.info("results written", path=str(args.output))


if __name__ == "__main__":
    main()
