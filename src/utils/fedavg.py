import torch
import copy
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

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


def _get_key(
  psword: str
) -> bytes:
  """
  Generate an encryption key from a password using PBKDF2.
  
  Args:
      psword (str): Password string for key generation
  
  Returns:
      bytes: 32-byte key suitable for Fernet encryption
  """
  kdf = PBKDF2HMAC(
    algorithm   = hashes.SHA256(),
    length      = 32,
    salt        = b'federated_learning', 
    iterations  = 100000,
  )
  key = base64.urlsafe_b64encode(kdf.derive(psword.encode()))
  return key


def model_encode(
  data: bytes, 
  psword: str
) -> bytes:
  """
  Encrypt serialized model data using Fernet symmetric encryption.
  
  Args:
      data (bytes): Serialized model data (from pickle.dumps)
      psword (str): Password for encryption
  
  Returns:
      bytes: Encrypted data
  """
  f = Fernet(_get_key(psword))
  return f.encrypt(data)


def model_decode(
  encrypted_data: bytes, 
  psword: str
) -> bytes:
  """
  Decrypt encrypted model data using Fernet symmetric encryption.
  
  Args:
      encrypted_data (bytes): Encrypted model data
      psword (str): Password for decryption
  
  Returns:
      bytes: Decrypted serialized data (for pickle.loads)
  """
  f = Fernet(_get_key(psword))
  return f.decrypt(encrypted_data)