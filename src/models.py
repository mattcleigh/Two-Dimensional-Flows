from copy import deepcopy
import torch as T
from torch import nn
import math
from src.normflows import CouplingSplineFlow
from src.mlp import MLP
from src.utils import append_dims, ema_param_sync, sample_heun


class Fourier(nn.Module):
    def __init__(self, outp_dim: int):
        super().__init__()
        self.outp_dim = outp_dim
        self.register_buffer("freqs", 2 * math.pi * T.randn(outp_dim))
        self.register_buffer("phases", 2 * math.pi * T.rand(outp_dim))

    def forward(self, x: T.Tensor) -> T.Tensor:
        y = x.float().ger(self.freqs) + self.phases  # Ger is outer product
        return y.cos().to(x.dtype) * math.sqrt(2)  # Sqrt(2) ensures variance is 1


def cosine_decay(step: int, total_steps: int) -> float:
    if step >= total_steps:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * step / total_steps))


class Flow(nn.Module):
    def __init__(
        self,
        input_dim=2,
        context_dim=0,
        num_layers=4,
        base="normal",
        layer_config=None,
        max_steps=1_000,
    ):
        super().__init__()
        self.flow = CouplingSplineFlow(
            input_dim=input_dim,
            context_dim=context_dim,
            num_layers=num_layers,
            base=base,
            layer_config=layer_config,
        )
        self.max_steps = max_steps

    def train_step(self, x0: T.Tensor, x1: T.Tensor, it: int) -> T.Tensor:
        z, logprob, change = self.flow(x0)
        change_weight = cosine_decay(it, self.max_steps)
        return -logprob.mean() + change_weight * change.mean()

    def get_density(self, x0: T.Tensor) -> T.Tensor:
        return self.flow(x0)[1].exp()

    def gen_stages(self, x1: T.Tensor):
        stages = [x1]
        for layer in reversed(self.flow.layers):
            x1, _ = layer.inverse(x1)
            stages.append(x1)
        return stages


class Diffuser(nn.Module):
    def __init__(
        self,
        schedule: str = "linear",
        time_sampling: str = "uniform",
    ) -> None:
        super().__init__()
        self.schedule = schedule
        self.time_sampling = time_sampling

        self.time_enc = Fourier(8)
        self.mlp = MLP(
            input_dim=2,
            output_dim=4,
            context_dim=8,
            hidden_dim=256,
            num_layers=3,
        )
        self.ema_mlp = deepcopy(self.mlp)
        self.ema_mlp.requires_grad_(False)

        self.n_times = 65
        self.times = T.linspace(1, 0, self.n_times)

    def get_velocity(self, xt: T.Tensor, t: T.Tensor):  # -> Any:
        """Get the predicted velocity."""
        model = self.mlp if self.training else self.ema_mlp
        out = model(xt, context=self.time_enc(t))
        return out.split(2, dim=-1)

    def forward(self, xt: T.Tensor, t: T.Tensor):  # -> Any:
        """Sample the predicted velocity."""
        return self.get_velocity(xt, t)[0]

    def train_step(self, x0: T.Tensor, x1: T.Tensor, it: int) -> T.Tensor:
        self.train()
        ema_param_sync(self.mlp, self.ema_mlp, 0.999)
        B = x0.shape[0]  # Batch size

        # Sample time values
        if self.time_sampling == "uniform":
            t = T.rand(B, device=x0.device)
        elif self.time_sampling == "signorm":
            t = T.randn(B, device=x0.device)
            t = T.sigmoid(t)
        td = append_dims(t, x0.ndim)

        # Schedule functions
        if self.schedule == "linear":
            a = 1 - td
            b = td
        elif self.schedule == "vp":
            a = T.cos(td * math.pi / 2)
            b = T.sin(td * math.pi / 2)

        # Get the target velocity
        if self.schedule == "linear":
            v = x1 - x0
        elif self.schedule == "vp":
            i = td * math.pi / 2
            v = math.pi / 2 * (T.cos(i) * x1 - T.sin(i) * x0)

        # Mix the distributions - calculate the loss
        xt = a * x0 + b * x1
        v_hat, log_var = self.get_velocity(xt, t)
        return ((v - v_hat).square() / log_var.exp() + log_var).mean()

    def gen_stages(self, x1):
        self.eval()
        return sample_heun(self, x1, self.times, save_all=True)[1]


def get_model(
    model_name: str,
    input_dim: int,
    base_dist: str,
    minmax: float,
    max_steps: int,
    **kwargs,
):
    if model_name == "flow":
        assert base_dist in {"normal", "uniform"}, "Flow can only do normal and uniform"
        assert input_dim > 1, "Flow needs input_dim > 1"
        return Flow(
            input_dim=input_dim,
            base=base_dist,
            layer_config={"limit": 1.1 * minmax},
            max_steps=max_steps,
            **kwargs,
        )
    if model_name == "linear_uniform":
        return Diffuser(schedule="linear", time_sampling="uniform")
    if model_name == "linear_signorm":
        return Diffuser(schedule="linear", time_sampling="signorm")
    if model_name == "vp_uniform":
        return Diffuser(schedule="vp", time_sampling="uniform")
    if model_name == "vp_signorm":
        return Diffuser(schedule="vp", time_sampling="signorm")
