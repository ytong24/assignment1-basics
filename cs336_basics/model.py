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


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.vocab_size = num_embeddings

        # Weights has the shape of (vocab_size x d_model)
        self.weights = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # Embedding wieghts: mean: 0, std: 1, a: -3, b: 3
        nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch_size, sequence_length]. weights: [vocab_size x d_model]. return: [..., d_model]

        # convert token_ids into [batch_size, sequence_length, vocab_size], where each integer token id is converted to a vector that have zeros everywhere except where the index of last dimension matches the corresponding value of the input tnesor, in which case it will be 1.
        # for example, given token id 2 and vocab_size 4, it will convert token id into [0,0,1,0]

        # one_hot = torch.nn.functional.one_hot(token_ids, num_classes=self.vocab_size).float()
        # y = einops.einsum(
        #     one_hot, self.weights, "... vocab_size, vocab_size d_model -> ... d_model"
        # )
        # return y

        return self.weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # g: (d_model)
        self.weights = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: (batch_size, sequence_length, d_model)
        # return: (batch_size, sequence_length, d_model)

        # prevent overflow when sqaure the input
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMSNorm
        rms = torch.sqrt(
            (x**2).mean(dim=-1, keepdim=True) + self.eps
        )  # (batch_size, sequence_length, 1)
        rms_norm = x / rms * self.weights

        return rms_norm.to(in_dtype)
