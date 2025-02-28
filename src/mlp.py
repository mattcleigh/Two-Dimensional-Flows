import torch as T
import torch.nn as nn

from src.utils import append_dims


class MLP(nn.Module):
    """Very simple multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim + context_dim, hidden_dim))
        self.layers.append(nn.SiLU())
        self.layers.append(nn.RMSNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.SiLU())
            self.layers.append(nn.RMSNorm(hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # Initialize the weights of the final layer to zero
        self.layers[-1].weight.data.zero_()
        self.layers[-1].bias.data.zero_()

    def forward(self, x: T.Tensor, context: T.Tensor = None) -> T.Tensor:
        if self.context_dim:
            context = append_dims(context, x.ndim)
            x = T.cat([x, context], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x
