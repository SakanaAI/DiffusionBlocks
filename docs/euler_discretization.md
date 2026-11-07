# Euler discretization and inference indices

This note makes the mathematical convention explicit for the existing sampler
in `ViTDBlockModel.diffusion_step`. It does not change the sampling algorithm.

## Signed versus positive step sizes

Equation (1) of [arXiv:2506.14202v4](https://arxiv.org/pdf/2506.14202v4) gives

```text
dz/dsigma = -sigma * score(z, sigma).
score(z, sigma) approximately equals (D(z, sigma) - z) / sigma^2.
```

Substituting the denoiser gives the approximate derivative `(z - D) / sigma`.
For a descending step from `s` to `t`, with `s > 0` and `0 <= t <= s`, Euler's
method therefore gives

```text
z_next = z + (t - s) * (z - D) / s.
```

This is the convention already implemented in `model.py`:

```python
d = (z - denoised) / sigma[:, None]
dt = next_sigma - sigma
euler_step = z + dt[:, None] * d
```

If the step size is instead defined as the nonnegative quantity `Delta = s - t`,
the equivalent expression is

```text
z_next = z + (Delta / s) * (D - z)
       = (t / s) * z + (1 - t / s) * D.
```

With the paper's definition `Delta_sigma = sigma_old - sigma_new > 0`, equations
(3)-(5) have the opposite sign from the Euler discretization of equation (1).
In equation (3), the score contribution should be `+ Delta_sigma * sigma_old * score`.
In equations (4) and (5), the denoiser residual should be `D - z_old`, rather than
`z_old - D`, when the preceding sign is `+ Delta_sigma / sigma_old`.

A scalar check distinguishes the two conventions: take `z=1`, `D=0`, `s=1`, and
`t=1/2`. The implemented step gives `1/2`; using positive `Delta=1/2` with
`z + (Delta/s)*(z-D)` gives `3/2` instead. The implemented sign should not be
reversed to match the printed equation.

## Consistent zero-based pseudocode

For a schedule `sigma[0], ..., sigma[T]` and an initialized state `z[0]`, a
consistent version of the mathematical inference loop is

```text
for i = 0, ..., T-1:
    choose the block using sigma[i]
    denoised = D_block(x, z[i], sigma[i])
    d = (z[i] - denoised) / sigma[i]
    z[i+1] = z[i] + (sigma[i+1] - sigma[i]) * d
return z[T]
```

Each denominator `sigma[i]` must be positive. A terminal `sigma[T]=0` is valid
for this Euler update because it is not used as a denominator. This does not
license evaluating a model's `log(sigma)` conditioning at zero.

Figure 3 instead starts at `i=0` while reading `z[i-1]` and `sigma[i-1]`, writes
only through `z[T-1]`, and returns `z[T]`. The version above avoids both the
negative initial index and the unassigned output. Selecting a block at a shared
interval endpoint also requires a consistent tie rule; the implementation uses
`bucketize` and clamping.

The public ViT implementation uses adjacent entries of `self.sigmas` and updates
one state variable in place. After its Euler loop, it performs an additional
denoiser evaluation at the final noise level to obtain classification logits.
The pseudocode here describes the state updates, not that final readout.

## Checking the examples

Run the dependency-free exact-arithmetic checks from the repository root:

```sh
python -m unittest discover -s tests -p 'test_euler_discretization.py' -v
```

The tests check the formulas and indexing above using rational arithmetic. They
do not import the training stack, run a model, or establish model quality,
convergence, or the accuracy of an approximate score estimate.
