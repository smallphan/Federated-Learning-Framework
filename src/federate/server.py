import torch
import torch.nn as nn
from models.mlp import mlp
from client import Client
from utils.utils import utils

class Server():
    
  def __init__(
    self,
    model:       nn.Module,
    num_clients: int
  ) -> None:
    self.model = model
    self.clients = []
    for _ in range(num_clients):
      self.clients.append(Client())
      
  def send_model_params(
    self,
    client: Client,
    params: dict
  ) -> None:
    client.model.load_state_dict(params)

  def recv_model_params(
    self,
    client: Client
  ) -> dict:
    return client.model.state_dict()

  def distribute_model(
    self
  ) -> None:
    for client in self.clients:
      self.send_model_params(client, self.model.state_dict())
  
  def aggregate_model(
    self
  ) -> None:
    state_dict = utils.initial_state_dict(self.model.state_dict())
    for client in self.clients:
      state_dict += utils.addup_state_dict(state_dict, self.recv_model_params(client))
    state_dict = utils.div_state_dict(len(self.clients))
    self.model.load_state_dict(state_dict)
  

