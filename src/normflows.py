import math
import numpy as np
import torch as T
from torch import nn
from torch.nn import functional as F

from src.mlp import MLP


def rational_quadratic_spline(
    x: T.Tensor,
    raw_widths: T.Tensor,
    raw_heights: T.Tensor,
    raw_derivs: T.Tensor,
    right: float = 1.0,
    left: float = -1.0,
    min_width: float = 1.0e-4,
    min_height: float = 1.0e-4,
    min_deriv: float = 1.0e-4,
    return_logdet: bool = True,
    inverse=False,
) -> tuple:
    """Rational quadratic spline function.

    Adapted from normflows library.
    Which itself is taken from https://arxiv.org/pdf/1906.04032.

    Parameters
    ----------
    x : T.Tensor
        Input tensor for which the spline interpolation is computed.
    raw_widths : T.Tensor
        Tensor containing the UNNORMALIZED widths of the bins.
    raw_heights : T.Tensor
        Tensor containing the UNNORMALIZED heights of the bins.
    raw_derivs : T.Tensor
        Tensor containing the RAW (pos or neg) derivatives at the bin edges.
    limit : float, optional
        The spline is defined over [-limit, limit], in both x and y, by default 1.0.
    min_width : float, optional
        Minimum width of the bins to avoid numerical issues, by default 1.0e-3.
    min_height : float, optional
        Minimum height of the bins to avoid numerical issues, by default 1.0e-3.
    min_derivative : float, optional
        Minimum derivative to avoid numerical issues, by default 1.0e-3.
    return_logdet : bool, optional
        If True, returns the log-determinant of the Jacobian, by default True.
    inverse : bool, optional
        If True, computes the inverse of the spline, by default False.

    Returns
    -------
    tuple
        A tuple containing the output values and the log-determinant of the Jacobian.
    """

    # Check that the inputs are valid, too many bins means we cant fit the spline
    num_bins = raw_widths.shape[-1]
    span = right - left
    assert raw_widths.shape == raw_heights.shape, "Invalid input shapes"
    assert raw_derivs.shape[-1] + 1 == num_bins, "Invalid derivative shape"
    assert num_bins * min_width < span, "Too many bins given the minimal width"
    assert num_bins * min_height < span, "Too many bins given the minimal height"

    # Normalise the widths and heights, make all derivatives positive
    # If the inputs are all zero, this ensures that the spline is the identity
    const = math.log(math.exp(1 - min_deriv) - 1)
    widths = F.softmax(raw_widths, dim=-1)
    heights = F.softmax(raw_heights, dim=-1)
    derivs = F.softplus(raw_derivs + const) + min_deriv

    # Add a small value while keeping sum=1
    widths = min_width + (1 - min_width * num_bins) * widths
    heights = min_height + (1 - min_height * num_bins) * heights

    # Scale to the correct range
    widths = span * widths
    heights = span * heights

    # The cumulative widths and heights - used for the edges along each dimension
    edge_x = T.cumsum(widths, dim=-1)
    edge_y = T.cumsum(heights, dim=-1)
    edge_x = F.pad(edge_x, pad=(1, 0), mode="constant", value=0.0)  # Start at zero
    edge_y = F.pad(edge_y, pad=(1, 0), mode="constant", value=0.0)
    edge_x = edge_x + left  # Shift to the correct range
    edge_y = edge_y + left

    # Pad the derivatives with 1 on either end to match the linear interpolation
    derivs = F.pad(derivs, pad=(1, 1), value=1)

    # Create a mask which will checks if the values are in the range of the spline
    # If all values are outside the range, the function should be the identity
    blim = left + 1e-3  # We are overly strict with the limits to avoid issues
    ulim = right - 1e-3  # when counting the bin locations
    out_mask = (x <= blim) | (x >= ulim)  # False for values within the range
    if T.all(out_mask):
        return x.clone(), x.new_zeros(x.shape[0])

    # Zero out the x values outside the range so they dont cause issues in the
    # spline calculations.
    # They will not be used as they have already been cloned!
    x_in = T.where(out_mask, 0, x)  # Use where to not do it in place

    # The bin index are defined by the edges, x for forward, y for inverse
    bin_edges = edge_y if inverse else edge_x
    bin_idxes = T.searchsorted(bin_edges, x_in.unsqueeze(-1), right=True) - 1

    # Get the values for the edge locations for each bin idx
    xk = edge_x.gather(-1, bin_idxes).squeeze(-1)
    yk = edge_y.gather(-1, bin_idxes).squeeze(-1)
    wk = widths.gather(-1, bin_idxes).squeeze(-1)
    hk = heights.gather(-1, bin_idxes).squeeze(-1)
    dk = derivs.gather(-1, bin_idxes).squeeze(-1)
    dk1 = derivs[..., 1:].gather(-1, bin_idxes).squeeze(-1)
    sk = hk / wk

    # Some useful terms in the upcoming equations
    eps = (x_in - xk) / wk
    eps_term = eps * (1 - eps)
    eps2 = eps.square()
    beta = sk + (dk1 + dk - 2 * sk) * eps_term

    if inverse:
        # More useful terms
        shift = x - yk
        theta = shift * (dk1 + dk - 2 * sk)

        # Equations (6), (7) and (8)
        a = hk * (sk - dk) + theta
        b = hk * dk - theta
        c = -sk * shift

        # Solve the quadratic equation
        quad = b.square() - 4 * a * c
        ep = 2 * c / (-b - quad.sqrt())
        output = ep * wk + xk

    else:
        # Equation (4)
        alpha = hk * (sk * eps2 + dk * eps_term)
        output = yk + alpha / beta

    output[out_mask] = x[out_mask]  # Reinsert the values outside the range
    logdet = None  # Placeholder

    if return_logdet:
        # Equation (5)
        dxa = 2 * T.log(sk)
        dxb = T.log(dk1 * eps2 + 2 * sk * eps_term + dk * (1 - eps).square())
        dxc = 2 * T.log(beta)
        logdet = dxa + dxb - dxc

        # Insert at the mask locations
        if inverse:
            logdet = -logdet
        logdet[out_mask] = 0  # Reinsert the values outside the range

    return output, logdet


class CouplingSplineLayer(nn.Module):
    """A couling rational quadratic spline layer with MLP derived parameters."""

    def __init__(
        self,
        input_dim: int,
        mask: T.BoolTensor,
        num_bins: int = 8,
        context_dim: int = 0,
        mlp_dim: int = 256,
        mlp_layers: int = 2,
        limit: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.num_bins = num_bins
        self.mask = mask  # True if transformed, False if not
        self.limit = limit
        self.trans_dim = (mask).sum()

        # The inputs to the mlp are the non-masked inputs and the context
        self.mlp = MLP(
            input_dim=(~mask).sum(),
            output_dim=(mask).sum() * (3 * num_bins - 1),
            context_dim=context_dim,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
        )

    def _shared_pass(
        self,
        x: T.Tensor,
        context: T.Tensor | None = None,
        inverse: bool = False,
    ) -> tuple:
        # Split the input into the masked and non-masked parts
        x1 = x[..., self.mask]
        x2 = x[..., ~self.mask]

        # Pass the non-masked part through the mlp and split the output
        raw = self.mlp(x2, context)
        raw = raw.view(*x1.shape[:-1], self.trans_dim, 3 * self.num_bins - 1)
        raw_w, raw_h, raw_d = T.split(
            raw, [self.num_bins, self.num_bins, self.num_bins - 1], dim=-1
        )

        # Compute the spline
        x1, logdet = rational_quadratic_spline(
            x1,
            raw_w,
            raw_h,
            raw_d,
            right=self.limit,
            left=-self.limit,
            inverse=inverse,
        )

        # Build the output tensor
        out = T.empty_like(x)
        out[..., self.mask] = x1
        out[..., ~self.mask] = x2

        return out, logdet.flatten(1).sum(-1)

    def forward(self, x: T.Tensor, context: T.Tensor | None = None) -> tuple:
        return self._shared_pass(x, context, inverse=False)

    def inverse(self, y: T.Tensor, context: T.Tensor | None = None) -> tuple:
        return self._shared_pass(y, context, inverse=True)


class Scaler(nn.Module):
    """Scale the input tensor by a learnable parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(T.zeros(1))

    def forward(self, x: T.Tensor, context: T.Tensor | None = None) -> tuple:
        scale = self.scale.exp()
        return x * scale, self.scale.expand(x.shape[0])

    def inverse(self, z: T.Tensor, context: T.Tensor | None = None) -> tuple:
        scale = self.scale.exp()
        return z / scale, -self.scale.expand(z.shape[0])


class UniformDist(nn.Module):
    """Multivariate uniform distribution."""

    def __init__(self, dim: int, low=-1.0, high=1.0):
        super().__init__()
        self.dim = dim
        self.low = low
        self.high = high
        self.range = high - low
        self.log_prob_val = -self.dim * np.log(self.range)
        self.register_buffer("dev_tensor", T.tensor([0.0]))

    def forward(self, num_samples: int = 1) -> tuple:
        eps = T.rand((num_samples, self.dim), device=self.dev_tensor.device)
        z = self.low + self.range * eps
        log_p = T.full((num_samples,), self.log_prob_val, device=z.device)
        return z, log_p

    def log_prob(self, z: T.Tensor) -> T.Tensor:
        log_p = T.full((z.shape[0],), self.log_prob_val, device=z.device)
        out_range = (z < self.low) | (self.high < z)
        ind_inf = T.any(out_range.view(z.shape[0], -1), dim=-1)
        log_p[ind_inf] = -T.inf
        return log_p


class NormalDist(nn.Module):
    """Multivariate standard normal distribution."""

    def __init__(self, dim: tuple):
        super().__init__()
        self.dim = dim
        self.register_buffer("dev_tensor", T.tensor([0.0]))

    def forward(self, num_samples: int = 1) -> tuple:
        z = T.randn((num_samples, self.dim), device=self.dev_tensor.device)
        return z, self.log_prob(z)

    def log_prob(self, z: T.Tensor) -> T.Tensor:
        return -0.5 * self.dim * np.log(2 * np.pi) - 0.5 * z.square().sum(-1)


class CouplingSplineFlow(nn.Module):
    """A stack of conditional spline layers."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int = 0,
        num_layers: int = 2,
        base: str = "normal",
        do_scales: bool = False,
        layer_config: dict | None = None,
    ) -> None:
        super().__init__()
        layer_config = layer_config or {}

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                CouplingSplineLayer(
                    input_dim=input_dim,
                    context_dim=context_dim,
                    mask=T.arange(input_dim) % 2 == (i % 2),  # Alternating even/odd
                    **layer_config,
                )
            )
            if do_scales:
                self.layers.append(Scaler())

        if base == "normal":
            self.base_dist = NormalDist(dim=input_dim)
        elif base == "uniform":
            self.base_dist = UniformDist(dim=input_dim)
        else:
            raise ValueError(f"Unknown base distribution {base}")

    def forward(self, x: T.Tensor, context: T.Tensor | None = None) -> tuple:
        logprob = x.new_zeros(x.shape[0])
        change = 0
        for layer in self.layers:
            x_new, ld = layer(x, context)
            logprob += ld
            change += (x_new - x).abs().mean()
            x = x_new
        logprob += self.base_dist.log_prob(x)
        return x, logprob, change

    def inverse(self, z: T.Tensor, context: T.Tensor | None = None) -> tuple:
        logdet = z.new_zeros(z.shape[0])
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z, context)
            logdet += ld
        return z, logdet

    def sample(self, num_samples: int = 1, context: T.Tensor | None = None) -> tuple:
        z, logprob = self.base_dist(num_samples)
        z, logdet = self.inverse(z, context)
        return z, logprob + logdet
