import torch
from torch.utils.data import Dataset
from typing import Callable
from random import randint

def default_function(
  input: torch.Tensor
) -> torch.Tensor:
  
  return (input * input).sum()


class linearDataset(Dataset):

  def __init__(self,
    input_size:   int = 3,
    num_samples:  int = randint(100, 200),
    gen_function: Callable[[torch.Tensor], torch.Tensor] = default_function
  ) -> None:
    
    super().__init__()
    self.num_samples  = num_samples
    self.gen_function = gen_function
    self.input_size   = input_size


  def __len__(self) -> int:
    return self.num_samples
  

  def __getitem__(self, 
    index: int
  ) -> tuple[torch.Tensor, torch.Tensor]:
    
    input = torch.randn(size = (1, self.input_size))
    ouput = self.gen_function(input)
    return input, ouput
  