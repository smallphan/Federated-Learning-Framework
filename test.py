import torch
import torch.nn as nn
from utils.utils import utils
from models.mlp import mlp

a = {
    "asdfasdf": torch.tensor([1, 2, 3, 4]),
    "sadf": 55,
    "ffff": torch.zeros([10, 2])
}

print(utils.initial_state_dict(a))

# print(model.state_dict())
# aggregate_state_dict = model.state_dict()
# print(aggregate_state_dict)
# for layer in aggregate_state_dict:
#     aggregate_state_dict[layer] = torch.zeros_like(aggregate_state_dict[layer])
# print(aggregate_state_dict)