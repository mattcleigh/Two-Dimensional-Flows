import torch as T
from torch.optim import Optimizer

from torch.optim.lr_scheduler import LambdaLR
import math
from torch import nn


@T.no_grad()
def ema_param_sync(source: nn.Module, target: nn.Module, gamma: float) -> None:
    for s, t in zip(source.parameters(), target.parameters(), strict=False):
        t.data.copy_(t.data.lerp_(s.data, 1 - gamma))


def linear_warmup_cosine_decay(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    total_steps: int = 10000,
    final_factor: float = 1e-3,
    init_factor: float = 1e-5,
) -> LambdaLR:
    """Return a scheduler with a linear warmup and a cosine decay."""

    warmup_steps = max(1, warmup_steps)  # Avoid division by zero
    assert 0 < final_factor < 1, "Final factor must be less than 1"
    assert 0 < init_factor < 1, "Initial factor must be less than 1"
    assert 0 < warmup_steps < total_steps, "Total steps must be greater than warmup"

    def fn(x: int) -> float:
        if x <= warmup_steps:
            return init_factor + x * (1 - init_factor) / warmup_steps
        if x >= total_steps:
            return final_factor
        t = (x - warmup_steps) / (total_steps - warmup_steps) * math.pi
        return (1 + math.cos(t)) * (1 - final_factor) / 2 + final_factor

    return LambdaLR(optimizer, fn)


def append_dims(x: T.Tensor, target_dims: int, dim=-1) -> T.Tensor:
    """Append dimensions of size 1 to tensor until it has target_dims."""
    if (dim_diff := target_dims - x.dim()) < 0:
        raise ValueError(f"x has more dims ({x.ndim}) than target ({target_dims})")

    # Fast exit conditions
    if dim_diff == 0:
        return x
    if dim_diff == 1:
        return x.unsqueeze(dim)
    if dim == -1:
        return x[(...,) + (None,) * dim_diff]
    if dim == 0:
        return x[(None,) * dim_diff + (...)]

    # Check if the dimension is in range
    allow = [-x.dim() - 1, x.dim()]
    if not allow[0] <= dim <= allow[1]:
        raise IndexError(
            f"Dimension out of range (expected to be in {allow} but got {dim})"
        )

    # Following only works for a positive index
    if dim < 0:
        dim += x.dim() + 1
    return x.view(*x.shape[:dim], *dim_diff * (1,), *x.shape[dim:])


def sample_heun(
    vel_fn: callable,
    x: T.Tensor,
    times: T.Tensor,
    save_all: bool = False,
    **kwargs,
) -> tuple:
    num_steps = len(times) - 1
    if save_all:
        all_stages = [x]
    time_shape = x.new_ones([x.shape[0]])
    for i in range(num_steps):
        d = vel_fn(x, times[i] * time_shape, **kwargs)
        dt = times[i + 1] - times[i]
        x_2 = x + d * dt
        d_2 = vel_fn(x_2, times[i + 1] * time_shape, **kwargs)
        d_prime = (d + d_2) / 2
        x = x + d_prime * dt
        if save_all:
            all_stages.append(x)
    return x, all_stages
