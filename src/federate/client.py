import torch
import torch.nn as nn
from models.mlp import mlp

class Client():
    
  def __init__(
    self,
    model: nn.Module
  ) -> None:
    self.model = model

  def train(
    self,
      
  ) -> None:
    
