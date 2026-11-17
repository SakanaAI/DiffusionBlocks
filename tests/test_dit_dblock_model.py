import numpy as np
import torch
from conftest import make_model
from PIL import Image

from data import CIFAR10DataModule


def test_training_step_updates_only_the_sampled_block() -> None:
    model = make_model()
    batch = {"pixel_values": torch.randn(4, 3, 8, 8), "labels": torch.arange(4)}
    loss, logs = model.shared_step(batch)
    loss.backward()
    (block_key,) = [k for k in logs if k.startswith("train/loss_")]
    block = int(block_key[-1])
    grads = [
        any(p.grad is not None for p in m.parameters()) for m in model.model.layers
    ]
    assert torch.isfinite(loss) and grads == [i == block for i in range(3)]
    assert model.num_blocks == 3  # max_epochs multiplier


def test_sample_is_deterministic_and_cfg_zero_is_conditional() -> None:
    model = make_model(cfg_scale=2.0).eval()
    labels = torch.arange(3)
    a = model.sample(labels, torch.Generator().manual_seed(0))
    b = model.sample(labels, torch.Generator().manual_seed(0))
    assert a.shape == (3, 3, 8, 8) and torch.equal(a, b)
    z, sigma = torch.randn(3, 3, 8, 8), torch.full((3,), 1.0)
    model.cfg_scale = 0.0
    no_cfg = model.denoise(z, sigma, labels)
    model.train()
    torch.testing.assert_close(no_cfg, model.denoise(z, sigma, labels))


def test_cfg_scale_one_is_conditional_and_guidance_is_linear() -> None:
    """D_u + s·(D_c − D_u): s = 1 gives D_c (s = 0 disables CFG) and D is linear in s."""
    model = make_model().eval()
    z, sigma, labels = torch.randn(3, 3, 8, 8), torch.full((3,), 1.0), torch.arange(3)
    for w in (  # break AdaLN-Zero so the label changes the output (cond ≠ uncond)
        model.model.y_embedder.weight,
        model.model.final_layer.adaLN_modulation.linear.weight,
        model.model.final_layer.linear.weight,
    ):
        torch.nn.init.normal_(w)
    out = {}
    for s in (0.0, 1.0, 2.0, 3.0):
        model.cfg_scale = s
        out[s] = model.denoise(z, sigma, labels)
    torch.testing.assert_close(out[1.0], out[0.0])
    torch.testing.assert_close(out[3.0] - out[1.0], 2 * (out[2.0] - out[1.0]))
    assert not torch.allclose(out[2.0], out[1.0])


def test_sampling_schedule_is_rebuilt_not_loaded() -> None:
    model = make_model()
    assert "sigmas" not in model.state_dict()
    ckpt = {"state_dict": {"sigmas": torch.zeros(6)}}
    model.on_load_checkpoint(ckpt)
    assert ckpt["state_dict"] == {}
    assert torch.equal(model.sigmas, model.partition.get_discrete_sigmas(6))


def test_cifar10_transform_is_flip_only_in_minus_one_one() -> None:
    img = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
    out = CIFAR10DataModule(num_workers=0).train_transforms(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32) and torch.all(out == 1.0)
