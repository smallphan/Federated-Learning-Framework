import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
from typing import Iterator

"""
Torch optimizers supported
"""
OPTIMIZER_MAPPING = {
  'Adam': optim.Adam,
  'SGD': optim.SGD,
  'RMSprop': optim.RMSprop,
  'Adagrad': optim.Adagrad,
  'Adadelta': optim.Adadelta,
  'AdamW': optim.AdamW,
  'ASGD': optim.ASGD,
  'LBFGS': optim.LBFGS
}

"""
Torch loss functions supported
"""
CRITERION_MAPPING = {
  'MSELoss': nn.MSELoss,
  'CrossEntropyLoss': nn.CrossEntropyLoss,
  'BCELoss': nn.BCELoss,
  'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
  'L1Loss': nn.L1Loss,
  'NLLLoss': nn.NLLLoss,
  'KLDivLoss': nn.KLDivLoss,
  'SmoothL1Loss': nn.SmoothL1Loss,
  'HuberLoss': nn.HuberLoss,
  'CTCLoss': nn.CTCLoss,
  'CosineEmbeddingLoss': nn.CosineEmbeddingLoss,
  'MarginRankingLoss': nn.MarginRankingLoss,
  'MultiMarginLoss': nn.MultiMarginLoss,
  'TripletMarginLoss': nn.TripletMarginLoss,
  'PoissonNLLLoss': nn.PoissonNLLLoss
}

def gen_optimizer(
  optimizer_name: str,
  model_params: Iterator[Parameter],
  **kwargs
) -> Optimizer:
  """
  Safely create an optimizer instance with validation.
  Supports common PyTorch optimizers like Adam, SGD, RMSprop, etc.
  
  Args:
    optimizer_name (str): Name of the optimizer to use, must be one of:
        'Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW', 'ASGD', 'LBFGS'
    model_params (Iterator[Parameter]): Model parameters to optimize
    **kwargs: Additional optimizer-specific parameters such as:
        - lr: learning rate
        - weight_decay: L2 penalty
        - momentum: for SGD
        - beta1, beta2: for Adam
  
  Returns:
    Optimizer: Instantiated PyTorch optimizer
  
  Raises:
    ValueError: If optimizer_name is not supported
  """
  if optimizer_name not in OPTIMIZER_MAPPING:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
      
  optimizer_class = OPTIMIZER_MAPPING[optimizer_name]
  
  return optimizer_class(params = model_params, **kwargs)


def gen_criterion(
  criterion_name: str,
  **kwargs
) -> nn.Module:
  """
  Safely create a loss function instance with validation.
  Supports common PyTorch loss functions like MSE, CrossEntropy, BCE, etc.
  
  Args:
    criterion_name (str): Name of the loss function to use, must be one of:
        'MSELoss', 'CrossEntropyLoss', 'BCELoss', 'BCEWithLogitsLoss',
        'L1Loss', 'NLLLoss', 'KLDivLoss', 'SmoothL1Loss', 'HuberLoss',
        'CTCLoss', 'CosineEmbeddingLoss', 'MarginRankingLoss',
        'MultiMarginLoss', 'TripletMarginLoss', 'PoissonNLLLoss'
    **kwargs: Loss function specific parameters such as:
        - reduction: 'none', 'mean', 'sum'
        - weight: class weights for imbalanced datasets
        - ignore_index: for tasks with masked targets
  
  Returns:
    nn.Module: Instantiated PyTorch loss function
  
  Raises:
    ValueError: If criterion_name is not supported
  """
  if criterion_name not in CRITERION_MAPPING:
    raise ValueError(f"Unsupported criterion: {criterion_name}")
      
  criterion_class = CRITERION_MAPPING[criterion_name]
  return criterion_class(**kwargs)

