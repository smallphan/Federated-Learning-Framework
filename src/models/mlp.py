import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class mlp(nn.Module):
  """
  Multi-Layer Perceptron (MLP) model for federated learning.
  A simple feed-forward neural network with configurable input, hidden, and output dimensions.
  """

  def __init__(self, 
    input_dim = 3, 
    hidden_dim = 64, 
    output_dim = 1
  ) -> None:
    """
    Initialize the MLP model.

    Args:
      input_dim (int): Dimension of input features
      hidden_dim (int): Dimension of hidden layers
      output_dim (int): Dimension of output
    """
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, output_dim)
    )
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the model.

    Args:
      x (torch.Tensor): Input tensor

    Returns:
      torch.Tensor: Output predictions
    """
    return self.network(x)
