import torch
import torch.nn as nn


def compute_dimension(kernel_size, stride, hidden_size):
    return ((hidden_size - kernel_size) // stride) + 1


class MeanPooler(nn.Module):
    def __init__(self, kernel_size: int, stride: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling_layer = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        pooled_output = self.pooling_layer(logits)
        return pooled_output


class MaxPooler(nn.Module):
    def __init__(self, kernel_size: int, stride: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling_layer = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        pooled_output = self.pooling_layer(logits)
        return pooled_output


class CLSPooler(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kernel_size = 1
        self.stride = 1

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        sequence_length = logits.size(1)
        pooled_output = logits[:, 0, :]
        pooled_output = pooled_output.repeat(1, sequence_length, 1)
        return pooled_output


class IdentityPooler(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kernel_size = 1
        self.stride = 1

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits
