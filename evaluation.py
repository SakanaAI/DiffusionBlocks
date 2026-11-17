"""Generation evaluation (paper App. E.2): clean-fid FID as a torchmetrics `Metric`, and
test-stage callbacks that save a class-conditional sample grid and the run's results."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import lightning as L
import numpy as np
import torch
from cleanfid.features import build_feature_extractor, get_reference_statistics
from cleanfid.fid import frechet_distance
from cleanfid.resize import build_resizer
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities import dim_zero_cat
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image


class FIDReference(StrEnum):
    """Which split's real-image statistics the generated images are compared against."""

    TRAIN = "train"
    TEST = "test"


@dataclass(frozen=True)
class FIDMode:
    name: str
    runs_on_mps: bool = True


# Train FID approximates the paper's TensorFlow ADM suite; test FID uses clean-fid's
# default mode, as in the paper. The TensorFlow-port Inception fails on MPS, so it
# falls back to the CPU there.
FID_MODES: dict[FIDReference, FIDMode] = {
    FIDReference.TRAIN: FIDMode("legacy_tensorflow", runs_on_mps=False),
    FIDReference.TEST: FIDMode("clean"),
}


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """[−1, 1] floats → (N, H, W, C) uint8 on the CPU, as a saved PNG stores them."""
    images = ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return images.permute(0, 2, 3, 1).cpu()


class CleanFID(Metric):
    """FID against clean-fid's precomputed statistics of a dataset split. `update` takes
    generated images in [−1, 1] and runs clean-fid's own per-image pipeline (uint8 →
    mode-specific resize → Inception), so it matches `cleanfid.fid.compute_fid` on the
    same images saved as PNGs."""

    full_state_update = False
    higher_is_better = False

    def __init__(self, dataset_name: str, dataset_res: int, reference: FIDReference):
        super().__init__()
        self.dataset_name, self.dataset_res = dataset_name, dataset_res
        self.reference, self.mode = reference, FID_MODES[reference]
        self.resize = build_resizer(self.mode.name)
        # built on first use: downloads the Inception weights, plain function (no state)
        self.extractor: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.add_state("features", default=[], dist_reduce_fx="cat")
        self.features: list[torch.Tensor]

    @cached_property
    def reference_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean and covariance of the real images' features (downloaded once, cached)."""
        return cast(
            tuple[np.ndarray, np.ndarray],
            get_reference_statistics(
                self.dataset_name,
                self.dataset_res,
                mode=self.mode.name,
                split=self.reference.value,
            ),
        )

    def load_reference(self) -> None:
        """Fetch the reference statistics now rather than at the first `compute`."""
        self.reference_stats  # noqa: B018 (evaluated for its caching side effect)

    def inception_device(self, images: torch.Tensor) -> torch.device:
        if images.device.type == "mps" and not self.mode.runs_on_mps:
            return torch.device("cpu")
        return images.device

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """[−1, 1] images → clean-fid's Inception input, as its ResizeDataset builds it:
        `to_tensor` keeps 0–255 for float32 resizes, and uint8 (no resize) is scaled
        back up to 0–255."""
        return torch.stack(
            [
                to_tensor(r) if r.dtype == np.float32 else to_tensor(r) * 255
                for r in map(self.resize, to_uint8(images).numpy())
            ]
        )

    def update(self, images: torch.Tensor) -> None:
        device = self.inception_device(images)
        if self.extractor is None:
            self.extractor = build_feature_extractor(
                self.mode.name, device, use_dataparallel=False
            )
        with torch.no_grad():
            features = self.extractor(self.preprocess(images).to(device))
        self.features.append(features.to(self.device))

    def compute(self) -> torch.Tensor:
        feats = dim_zero_cat(self.features).cpu().double().numpy()
        mu, sigma = self.reference_stats
        fid = frechet_distance(
            np.mean(feats, axis=0), np.cov(feats, rowvar=False), mu, sigma
        )
        return torch.tensor(float(fid))


TEST_PREFIX = "test/"
FID_PREFIX = f"{TEST_PREFIX}fid_"  # the logged names, e.g. "test/fid_train"


def build_fid_metrics(dataset_name: str, image_size: int) -> MetricCollection:
    """One FID per reference split, logged as `test/fid_<split>`."""
    return MetricCollection(
        {f"fid_{ref}": CleanFID(dataset_name, image_size, ref) for ref in FIDReference},
        prefix=TEST_PREFIX,
    )


@runtime_checkable
class ImageGenerator(Protocol):
    def sample(
        self, labels: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor: ...


class SaveResults(L.Callback):
    """At the end of the test stage, writes the logged FIDs and the run's metadata
    (config, number of generated images, device, versions, time) to a JSON file."""

    def __init__(self, path: Path, config: dict[str, Any]) -> None:
        self.path, self.config = path, config
        self.num_generated = 0

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.num_generated = 0

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: object,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.num_generated += len(batch["labels"])  # one image per label

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        results = {
            "fid": {
                name.removeprefix(TEST_PREFIX): value.item()
                for name, value in trainer.callback_metrics.items()
                if name.startswith(FID_PREFIX)
            },
            "num_generated": self.num_generated,
            "config": self.config,
            "device": str(pl_module.device),
            "torch": torch.__version__,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(results, indent=2))


class SampleGrid(L.Callback):
    """At the end of the test stage, saves a grid with one row per class. A fixed seed
    (0) gives every model the same initial noise, so grids can be compared side by side."""

    def __init__(self, path: Path, num_classes: int, per_class: int = 10) -> None:
        self.path, self.num_classes, self.per_class = path, num_classes, per_class

    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        assert isinstance(pl_module, ImageGenerator)
        labels = torch.arange(self.num_classes, device=pl_module.device)
        images = pl_module.sample(
            labels.repeat_interleave(self.per_class),
            torch.Generator().manual_seed(0),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        save_image(
            images,
            self.path,
            nrow=self.per_class,
            padding=1,
            pad_value=0.5,
            normalize=True,
            value_range=(-1, 1),
        )
