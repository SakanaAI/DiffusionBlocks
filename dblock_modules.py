"""DiffusionBlocks method core: noise partitioning, per-block σ sampling and EDM preconditioning.

Pure numpy/torch (no Lightning, no transformers), shared by classification and generation.
Paper refs: §3.3 (equi-probability partitioning), App. C (overlap γ), Eq. (6) (EDM loss), Karras et al. 2022.
"""

import random
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import torch
from scipy.stats import norm


@dataclass(frozen=True)
class NoiseConfig:
    """EDM defaults used by the paper: log σ ~ N(p_mean, p_std²) on [sigma_min, sigma_max]."""

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    p_mean: float = -1.2
    p_std: float = 1.2
    sigma_data: float = 0.5


def get_block_sigmas(
    num_layers: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> list[float]:
    """Equi-probability block boundaries σ_0 < … < σ_B (ascending), §3.3."""
    cdf_min = norm.cdf((np.log(sigma_min) - p_mean) / p_std)
    cdf_max = norm.cdf((np.log(sigma_max) - p_mean) / p_std)
    block_sigmas = []
    for i in range(num_layers + 1):
        p = cdf_min + (cdf_max - cdf_min) * (i / num_layers)
        sigma = np.exp(p_mean + p_std * norm.ppf(p))
        block_sigmas.append(sigma)
    return block_sigmas


def get_discrete_sigmas(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    dblock: bool = False,
) -> torch.Tensor:
    """Descending inference σ schedule: Karras ρ-schedule, or equi-probability spaced if `dblock` (App. H)."""
    if not dblock:
        ramp = torch.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas
    else:
        log_sigma_min = np.log(sigma_min)
        log_sigma_max = np.log(sigma_max)
        cdf_min = norm.cdf((log_sigma_min - p_mean) / p_std)
        cdf_max = norm.cdf((log_sigma_max - p_mean) / p_std)
        cdf_points = np.linspace(cdf_min, cdf_max, num_steps)
        sigmas = np.exp(p_mean + p_std * norm.ppf(cdf_points))
        sigmas = torch.tensor(sigmas, dtype=torch.float32)
        sigmas = torch.flip(sigmas, dims=[0])
        return sigmas


@dataclass(frozen=True)
class BlockPartition:
    """Partition of the noise range into `num_blocks` blocks. Block 0 owns the HIGHEST noise interval."""

    num_blocks: int
    gamma: float = 0.05
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    @cached_property
    def block_sigmas(self) -> list[float]:
        """Ascending block boundaries σ_0 < … < σ_B, length num_blocks + 1 (§3.3)."""
        n = self.noise
        return get_block_sigmas(
            self.num_blocks, n.sigma_min, n.sigma_max, n.p_mean, n.p_std
        )

    def get_block_sigma_range(self, block_idx: int) -> tuple[float, float]:
        """Training σ range of a block: its boundaries pushed out by γ·(log width)
        on each side, clamped to the global range (App. C)."""
        block_sigmas = self.block_sigmas
        i = self.num_blocks - 1 - block_idx
        sigma_min_block, sigma_max_block = block_sigmas[i], block_sigmas[i + 1]
        if self.gamma > 0.0:
            log_sigma_min = np.log(sigma_min_block)
            log_sigma_max = np.log(sigma_max_block)
            log_range = log_sigma_max - log_sigma_min
            sigma_min_block = max(
                np.exp(log_sigma_min - self.gamma * log_range), block_sigmas[0]
            )
            sigma_max_block = min(
                np.exp(log_sigma_max + self.gamma * log_range), block_sigmas[-1]
            )
        return sigma_min_block, sigma_max_block

    def sample_block_idx(self) -> int:
        """Uniform block per iteration (App. C)."""
        return self.num_blocks - 1 - random.choices(range(self.num_blocks), k=1)[0]

    def get_sigmas(self, block_idx: int, n_samples: int) -> torch.Tensor:
        """Log-normal σ truncated to the block's range, by inverse CDF (float64)."""
        sigma_min_block, sigma_max_block = self.get_block_sigma_range(block_idx)
        p_mean, p_std = self.noise.p_mean, self.noise.p_std
        cdf_min_block = norm.cdf((np.log(sigma_min_block) - p_mean) / p_std)
        cdf_max_block = norm.cdf((np.log(sigma_max_block) - p_mean) / p_std)
        rand = np.random.uniform(cdf_min_block, cdf_max_block, n_samples)
        return torch.from_numpy(np.exp(p_mean + p_std * norm.ppf(rand)))

    def get_block_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        """Owning block of each σ, using the un-expanded boundaries."""
        block_sigmas = torch.tensor(self.block_sigmas, device=sigma.device)
        block_idx = torch.bucketize(sigma, block_sigmas, right=True) - 1
        block_idx = (self.num_blocks - 1) - block_idx
        return torch.clamp(block_idx, 0, self.num_blocks - 1).long()

    def estimate_target_layer(self, sigma: torch.Tensor) -> int:
        """Most common owning block over a batch (the repo runs one block per batch)."""
        values, counts = self.get_block_idx(sigma).unique(return_counts=True)
        return int(values[counts.argmax()].item())

    def get_discrete_sigmas(self, num_steps: int) -> torch.Tensor:
        """Descending equi-probability inference schedule (App. H)."""
        n = self.noise
        return get_discrete_sigmas(
            num_steps,
            n.sigma_min,
            n.sigma_max,
            p_mean=n.p_mean,
            p_std=n.p_std,
            dblock=True,
        )

    def layer_assignment(self, num_layers: int) -> list[list[int]]:
        """Contiguous layer split; block b gets layers [b·L/B, (b+1)·L/B)."""
        size = num_layers // self.num_blocks
        return [list(range(i * size, (i + 1) * size)) for i in range(self.num_blocks)]


@dataclass(frozen=True)
class EDMCoefficients:
    """EDM preconditioning: D(z; σ) = c_skip·z + c_out·F(c_in·z; c_noise)."""

    c_skip: torch.Tensor
    c_out: torch.Tensor
    c_in: torch.Tensor
    c_noise: torch.Tensor

    @classmethod
    def from_sigma(
        cls, sigma: torch.Tensor, sigma_data: float = 0.5
    ) -> "EDMCoefficients":
        return cls(
            c_skip=sigma_data**2 / (sigma**2 + sigma_data**2),
            c_out=sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5,
            c_in=1 / (sigma**2 + sigma_data**2) ** 0.5,
            c_noise=0.25 * sigma.log(),
        )


def edm_loss_weight(sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
    """λ(σ) = (σ² + σ_data²) / (σ·σ_data)², which equals 1 / c_out²."""
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
