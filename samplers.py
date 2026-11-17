"""Samplers for the probability-flow ODE dz/dσ = (z − D(z; σ))/σ (Karras et al. 2022)."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from dblock_modules import BlockPartition

# denoise(z, σ, i): D(z; σ) for a per-sample σ (N,), run by the block of step i
DenoiseFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


class Sampler(Protocol):
    def get_block_idxs(
        self, partition: BlockPartition, sigmas: torch.Tensor
    ) -> list[int]:
        """Block of each step σ_i → σ_{i+1}, then of the final denoise."""
        ...

    def diffusion_step(
        self,
        denoise: DenoiseFn,
        z: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        i: int,
    ) -> torch.Tensor: ...


def sample(
    sampler: Sampler, denoise: DenoiseFn, sigmas: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Integrate from noise `z` at sigmas[0] down to sigmas[-1], then denoise there."""
    s_in = z.new_ones(z.shape[0])
    for i in range(sigmas.shape[0] - 1):
        z = sampler.diffusion_step(
            denoise, z, sigmas[i] * s_in, sigmas[i + 1] * s_in, i
        )
    return denoise(z, sigmas[-1] * s_in, sigmas.shape[0] - 1)


@dataclass(frozen=True)
class EulerSampler:
    """One evaluation per step, by the block that owns the step's starting σ."""

    def get_block_idxs(
        self, partition: BlockPartition, sigmas: torch.Tensor
    ) -> list[int]:
        return partition.get_block_idx(sigmas).tolist()

    def diffusion_step(
        self,
        denoise: DenoiseFn,
        z: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        i: int,
    ) -> torch.Tensor:
        d = (z - denoise(z, sigma, i)) / _expand(sigma, z)
        return z + _expand(next_sigma - sigma, z) * d


@dataclass(frozen=True)
class HeunSampler:
    """Heun, two evaluations per step, applied to the ODE in x = z/σ, u = 1/σ (dx/du = D):
    the predictor is the Euler step and the corrector averages the two denoiser outputs.
    Karras et al.'s Alg. 1 averages slopes in σ instead and overshoots on this schedule's
    first step (σ = 80 → ~2). Both evaluations of a step run one block, so no step may
    cross a block boundary."""

    def get_block_idxs(
        self, partition: BlockPartition, sigmas: torch.Tensor
    ) -> list[int]:
        s = sigmas.double().cpu()
        hi, lo = s[:-1, None], s[1:, None]
        bounds = torch.tensor(partition.block_sigmas[1:-1], dtype=torch.float64)
        tol = 1e-4  # boundaries match grid points only up to float32 rounding
        if ((bounds < hi * (1 - tol)) & (bounds > lo * (1 + tol))).any():
            raise ValueError(
                f"a Heun step crosses a block boundary: with {len(s)} points and "
                f"{partition.num_blocks} blocks, use (points − 1) divisible by the blocks"
            )
        midpoints = (s[:-1] * s[1:]).sqrt()  # log-space, away from the boundaries
        return partition.get_block_idx(torch.cat([midpoints, s[-1:]])).tolist()

    def diffusion_step(
        self,
        denoise: DenoiseFn,
        z: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        i: int,
    ) -> torch.Tensor:
        ratio = _expand(next_sigma / sigma, z)
        denoised = denoise(z, sigma, i)
        z_pred = denoised + ratio * (z - denoised)  # the Euler step
        mean = (denoised + denoise(z_pred, next_sigma, i)) / 2
        return mean + ratio * (z - mean)


def _expand(v: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """(N,) → (N, 1, …) to broadcast against `like`."""
    return v.view(-1, *([1] * (like.dim() - 1)))
