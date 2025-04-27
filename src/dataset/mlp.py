import torch
from torch.utils.data import Dataset
from typing import Callable
from random import randint

def default_function(
  input: torch.Tensor
) -> torch.Tensor:
  """
  Default function for generating synthetic dataset labels.
  Computes the sum of squares of the input tensor.

  Args:
      input (torch.Tensor): Input tensor of shape [1, input_size]

  Returns:
      torch.Tensor: Scalar output representing the sum of squares
  """
  return (input * input).sum()


class mlpDataset(Dataset):
  """
  Synthetic dataset generator for MLP training.
  Generates random input tensors and computes corresponding outputs using a specified function.
  Useful for testing and benchmarking MLP models in federated learning scenarios.
  """

  def __init__(self,
    input_size:   int = 3,
    num_samples:  int = randint(100, 200),
    gen_function: Callable[[torch.Tensor], torch.Tensor] = default_function
  ) -> None:
    """
    Initialize the MLP dataset.

    Args:
        input_size (int, optional): Dimension of input features. Defaults to 3.
        num_samples (int, optional): Number of samples in the dataset. 
            Defaults to random number between 100 and 200.
        gen_function (Callable, optional): Function to generate output from input. 
            Defaults to sum of squares function.
    """
    super().__init__()
    self.num_samples  = num_samples
    self.gen_function = gen_function
    self.input_size   = input_size


  def __len__(self) -> int:
    """
    Get the number of samples in the dataset.

    Returns:
        int: Number of samples
    """
    return self.num_samples
  

  def __getitem__(self, 
    index: int
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a sample from the dataset.

    Args:
        index (int): Index of the sample to get (unused since samples are generated randomly)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing:
            - Input tensor of shape [1, input_size]
            - Output tensor (scalar value)
    """
    input = torch.randn(size = (1, self.input_size))
    ouput = self.gen_function(input)
    return input, ouput
