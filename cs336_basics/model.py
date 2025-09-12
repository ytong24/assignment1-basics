import torch
import torch.nn as nn
import einops
import math


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Weights has the shape of (d_out x d_in)
        self.weights = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Linear wieghts: mean: 0, std: sqrt(2 / (d_in + d_out)), a: -3*std, b: 3*std
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = x @ W.T
        y = einops.einsum(x, self.weights, "... d_int, d_out d_int -> ... d_out")
        return y
