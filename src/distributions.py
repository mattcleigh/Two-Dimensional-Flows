"""Collection of toy data distributions for the generative models.

All distributions should have mean of 0 and variance of 1.
"""

import torch as T
import math


class Normal:
    def __init__(self, dim: int = 2, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self.minmax = 3

    def sample(self, n: int = 1):
        return T.randn(n, self.dim, device=self.device)


class Uniform:
    def __init__(self, dim: int = 2, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self.minmax = math.sqrt(3)

    def sample(self, n: int = 1):
        x = T.rand(n, self.dim, device=self.device)
        x = x * 2 - 1
        return x * math.sqrt(3)


class Moons:
    def __init__(self, noise=0.15, device: str = "cpu"):
        self.noise = noise
        self.device = device
        self.mean = T.tensor([0.0750, 0.3250], device=self.device)
        self.std = T.tensor([0.5790, 0.4961], device=self.device)
        self.minmax = 1.8

    def sample(self, n: int):
        na = n // 2
        nb = n - na
        outer_circ_x = T.cos(T.linspace(0, math.pi, na, device=self.device)) - 0.5
        outer_circ_y = T.sin(T.linspace(0, math.pi, na, device=self.device))
        inner_circ_x = 1 - T.cos(T.linspace(0, math.pi, nb, device=self.device)) - 0.5
        inner_circ_y = 1 - T.sin(T.linspace(0, math.pi, nb, device=self.device)) - 0.5
        x = T.stack([outer_circ_x / 1.5, outer_circ_y], dim=1)
        y = T.stack([inner_circ_x / 1.5, inner_circ_y], dim=1)
        x = T.vstack([x, y])
        x += T.rand_like(x) * self.noise
        return (x - self.mean) / self.std


class Squares:
    def __init__(self, s: int = 2, device: str = "cpu"):
        self.device = device
        self.n_squares = s
        self.mean = s - 0.5
        self.std = math.sqrt((s**2 - 1) / 3 + 1 / 12)
        self.minmax = 1.6

    def sample(self, n: int):
        b = T.randint(0, self.n_squares, (n, 2), device=self.device) * 2
        a = T.rand((n, 2), device=self.device)
        x = a + b
        return (x - self.mean) / self.std


class Deltas:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.minmax = 1

    def sample(self, n: int):
        return T.randint(0, 2, (n, 2), device=self.device, dtype=T.float32) * 2 - 1


DISTRIBUTIONS_ = {
    "normal": Normal,
    "uniform": Uniform,
    "moons": Moons,
    "squares": Squares,
    "deltas": Deltas,
}


def get_distribution(name: str, *args, **kwargs):
    """Get the data distribution by name."""
    return DISTRIBUTIONS_[name](*args, **kwargs)
