import torch
import torch.nn as nn
import torch.optim as optim
from models.mlp import mlp
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
from typing import Iterator

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
    Safely create an optimizer instance with validation
    
    Args:
        optimizer_name: Name of the optimizer to use
        model_params: Model parameters to optimize
        **kwargs: Optimizer specific parameters
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
    Safely create a loss function instance with validation
    
    Args:
        criterion_name: Name of the loss function to use
        **kwargs: Loss function specific parameters
    Returns:
        nn.Module: Instantiated loss function
    """
    if criterion_name not in CRITERION_MAPPING:
        raise ValueError(f"Unsupported criterion: {criterion_name}")
        
    criterion_class = CRITERION_MAPPING[criterion_name]
    return criterion_class(**kwargs)

