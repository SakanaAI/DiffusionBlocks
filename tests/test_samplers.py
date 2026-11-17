import pytest
import torch
from conftest import make_model

from config import SamplerType
from dblock_modules import BlockPartition
from samplers import EulerSampler, HeunSampler, sample


def test_heun_is_second_order_on_gaussian_data() -> None:
    """Data N(0, s²): D(z; σ) = z·s²/(s²+σ²); the ODE scales z by √((s²+σ²)/(s²+σ₀²))."""
    s = 0.5

    def denoise(z: torch.Tensor, sigma: torch.Tensor, _: int) -> torch.Tensor:
        return z * s**2 / (s**2 + sigma[:, None] ** 2)

    def error(sampler: EulerSampler | HeunSampler, points: int) -> float:
        sigmas = BlockPartition(3).get_discrete_sigmas(points).double()
        z = torch.randn(100, 1, dtype=torch.float64) * (s**2 + sigmas[0] ** 2).sqrt()
        end = z * ((s**2 + sigmas[-1] ** 2) / (s**2 + sigmas[0] ** 2)).sqrt()
        exact = denoise(end, sigmas[-1:], 0)
        return float((sample(sampler, denoise, sigmas, z) - exact).abs().max())

    assert (
        error(HeunSampler(), 25) < error(EulerSampler(), 50) / 4
    )  # matched evaluations
    assert error(HeunSampler(), 49) < error(HeunSampler(), 25) / 2.5  # ≈ 1/4: order 2


def test_heun_steps_stay_inside_one_block() -> None:
    partition = BlockPartition(3)
    blocks = HeunSampler().get_block_idxs(partition, partition.get_discrete_sigmas(7))
    assert blocks == [0, 0, 1, 1, 2, 2, 2]
    with pytest.raises(ValueError, match="crosses a block boundary"):
        HeunSampler().get_block_idxs(partition, partition.get_discrete_sigmas(8))


def test_heun_sample_is_deterministic() -> None:
    model = make_model(sampler=SamplerType.HEUN, num_inference_steps=7).eval()
    a = model.sample(torch.arange(3), torch.Generator().manual_seed(0))
    b = model.sample(torch.arange(3), torch.Generator().manual_seed(0))
    assert a.shape == (3, 3, 8, 8) and torch.equal(a, b)
