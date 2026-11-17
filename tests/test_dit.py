import torch
from conftest import TINY

from dblock_modules import BlockPartition
from dit import DiT, get_2d_sincos_pos_embed
from samplers import EulerSampler, sample


def make_inputs(
    n: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(n, 3, 8, 8), torch.randn(n), torch.arange(n), torch.ones(n)


def test_output_is_zero_at_init_and_image_shaped() -> None:  # AdaLN-Zero
    x, c_noise, y, keep = make_inputs()
    out = DiT(TINY).forward_block(x, c_noise, y, keep, [0, 1, 2])
    assert out.shape == x.shape and out.abs().max() == 0


def test_only_the_block_layers_get_gradients() -> None:
    model = DiT(TINY)
    torch.nn.init.normal_(model.final_layer.linear.weight)  # leave zero init
    model.forward_block(*make_inputs(), [1]).sum().backward()
    grads = [any(p.grad is not None for p in m.parameters()) for m in model.layers]
    assert grads == [False, True, False]


def test_null_condition_ignores_the_label() -> None:
    model = DiT(TINY)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x, c_noise, _, _ = make_inputs()
    null = torch.zeros(4)
    a = model.forward_block(x, c_noise, torch.tensor([0, 1, 2, 3]), null, [0])
    b = model.forward_block(x, c_noise, torch.tensor([9, 8, 7, 6]), null, [0])
    torch.testing.assert_close(a, b)


def test_pos_embed_is_unique_per_patch() -> None:
    assert get_2d_sincos_pos_embed(32, 4).unique(dim=0).shape == (16, 32)


def test_euler_sampler_recovers_two_point_data() -> None:
    """Exact denoiser for data ±1: D(z; σ) = tanh(z/σ²)."""
    sigmas = BlockPartition(3).get_discrete_sigmas(50)
    z = torch.randn(2000, 1) * (1 + sigmas[0] ** 2).sqrt()
    out = sample(
        EulerSampler(), lambda z, s, _: torch.tanh(z / s[:, None] ** 2), sigmas, z
    )
    assert torch.allclose(out.abs(), torch.ones_like(out), atol=1e-3)
    assert abs(out.mean()) < 0.1
