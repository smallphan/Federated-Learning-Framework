import torch
import copy

def state_dict_zeros_like(
  state_dict: dict
) -> dict:
  """
  Create a new state dictionary with the same structure but filled with zeros.
  
  Args:
    state_dict (dict): Reference state dictionary
  
  Returns:
    dict: A new state dictionary with same structure but zero values
  """
  new_dict = {}
  for key, value in state_dict.items():
    if isinstance(value, torch.Tensor):
      new_dict[key] = torch.zeros_like(value)
    else:
      new_dict[key] = copy.deepcopy(value)
  return new_dict


def aggregate_state_dicts(
  state_dicts: list[dict], 
  weights: list[float] = None
) -> dict:
    """
    Aggregate multiple model state dictionaries using weighted averaging.
    
    Args:
      state_dicts (list[dict]): List of model state dictionaries to aggregate
      weights (list[float], optional): List of weights for each state dict. 
        If None, equal weights are used.
    
    Returns:
        dict: Aggregated state dictionary
    """
    if weights is None:
      weights = [1.0 / len(state_dicts)] * len(state_dicts)
    
    result = state_dict_zeros_like(state_dicts[0])
    
    for sd, w in zip(state_dicts, weights):
      for key in result:
        if isinstance(result[key], torch.Tensor):
          result[key] += sd[key] * w
                
    return result


def model_quantization(
  state_dict: dict
) -> dict:
  """
  Quantize model parameters from float32 to float16 to reduce communication overhead.
  
  Args:
      state_dict (dict): Model state dictionary with float32 parameters
  
  Returns:
      dict: Quantized state dictionary with float16 parameters
  """
  quantization_dict = {}
  for key, tensor in state_dict.items():
    tensor: torch.Tensor
    quantization_dict[key] = tensor.to(torch.float16)
  return quantization_dict


def model_dequantization(
  state_dict: dict
) -> dict:
  """
  Dequantize model parameters from float16 back to float32 for computation.
  
  Args:
      state_dict (dict): Model state dictionary with float16 parameters
  
  Returns:
      dict: Dequantized state dictionary with float32 parameters
  """
  dequantization_dict = {}
  for key, tensor in state_dict.items():
    tensor: torch.Tensor
    dequantization_dict[key] = tensor.to(torch.float32)
  return dequantization_dict


