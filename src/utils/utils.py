import torch
import torch.nn as nn

class utils:
    
  def __init__(
    self
  ) -> None:
    pass

  def initial_state_dict(
    state_dict: dict
  ) -> dict:
      
    # Initialize a state_dict of nn.Module:
    # - Layers shape same as the param 'state_dict'
    # - Layers weight and bias set to zeros tensor

    for item in state_dict:
      if (type(state_dict[item]) != torch.Tensor):
        raise TypeError('Dict values should be torch.Tensor.')
      state_dict[item] = torch.zeros_like(state_dict[item])
    
    return state_dict

  def addup_state_dict(
    dict_a: dict,
    dict_b: dict
  ) -> dict:
      
    # Addup two state_dicts.

    for item in dict_a:
      if (type(dict_a[item]) != torch.Tensor or type(dict_b[item]) != torch.Tensor):
        raise TypeError('Dict values should be torch.Tensor.')
      dict_a[item] += dict_b[item]

    return dict_a
  
  def div_state_dict(
    state_dict: dict,
    divnumber:  int
  ) -> dict:

    # Div state_dict with number.

    for item in state_dict:
      if (type(state_dict[item]) != torch.Tensor):
        raise TypeError('Dict values should be torch.Tensor.')
      state_dict[item] /= divnumber
    
    return state_dict
    