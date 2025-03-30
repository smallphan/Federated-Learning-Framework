import torch
import torch.nn as nn

class mlp(nn.Module):

    def __init__(self,
        input_size:  int,
        middle_size: int,
        output_size: int
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, output_size)
        )

    def forward(self,
        input: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input)