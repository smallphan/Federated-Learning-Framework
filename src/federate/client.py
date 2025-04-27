import yaml
import asyncio
import pickle
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from utils import conf
import utils.fedavg as fedavg
from typing import Callable

class Client():
  """
  Federated Learning Client class that handles local training and model updates.
  """

  def __init__(self,
    model: nn.Module,
    dataset: Dataset,
    train_kernel: Callable[[nn.Module, int, DataLoader, str, nn.Module, Optimizer], None],
    config_file: str = './config/client_config.yaml'
  ) -> None:
    """
    Initialize the FL client.

    Args:
      model (nn.Module): The PyTorch model to be trained
      config_file (str): Path to configuration file
    """
    with open(config_file, 'r') as file:
      config = yaml.safe_load(file)

    self.device = 'cuda' if (config['train']['device'] == 'cuda' and torch.cuda.is_available()) else 'cpu'
    self.model = model.to(self.device)
    
    self.server_host:   str       = config['server']['host']
    self.server_port:   int       = config['server']['port']

    self.num_epochs:    int       = config['train']['num_epochs']
    self.learning_rate: float     = config['train']['learning_rate']
    self.batch_size:    int       = config['train']['batch_size']
    self.optimizer:     Optimizer = conf.gen_optimizer(config['train']['optimizer'], self.model.parameters(), lr = self.learning_rate)
    self.criterion:     nn.Module = conf.gen_criterion(config['train']['criterion'])

    self.dataset = dataset
    self.loader = DataLoader(self.dataset, self.batch_size, shuffle = True)
    self.train_kernel = train_kernel
  

  async def send_model_params(self,
    writer: asyncio.StreamWriter,
    state_dict: dict
  ) -> None:
    """
    Send model parameters to the server.

    Args:
      writer (asyncio.StreamWriter): Stream writer for sending data
      state_dict (dict): Model state dictionary
    """
    try:
      serialized_data = pickle.dumps(fedavg.model_quantization(state_dict))
      writer.write(len(serialized_data).to_bytes(4, 'big'))
      await writer.drain()
      writer.write(serialized_data)
      await writer.drain()

    except Exception as error:
      print(f'Exception: {error}')


  async def recv_model_params(self,
    reader: asyncio.StreamReader
  ) -> dict:
    """
    Receive model parameters from the server.

    Args:
      reader (asyncio.StreamReader): Stream reader for receiving data

    Returns:
      dict: Model state dictionary
    """
    try:    
      stream_length = await reader.readexactly(4)
      if not stream_length:
        raise ConnectionError

      stream_length = int.from_bytes(stream_length, 'big')

      serialized_data = await reader.readexactly(stream_length)
      return fedavg.model_dequantization(pickle.loads(serialized_data))

    except Exception as error:
      print(f'Exception: {error}')


  async def train(self) -> None:
    """
    Train the model for a specified number of epochs.
    """
    self.train_kernel(
      self.model,
      self.num_epochs,
      self.loader,
      self.device,
      self.criterion,
      self.optimizer
    )


  async def start(self) -> None:
    """
    Start the client training process and communicate with the server.
    """
    while True:
      reader, writer = await asyncio.open_connection(self.server_host, self.server_port)
      print(f'Successfully connected to server {self.server_host}:{self.server_port}')
    
      try: 
        while True:
          state_dict = await self.recv_model_params(reader)
          print(state_dict)
          self.model.load_state_dict(state_dict)
          await self.train()
          await self.send_model_params(writer, self.model.state_dict())

      except asyncio.CancelledError as error:
        print(f'Exception: {error}')

      except Exception as error:
        print(f'Exception: {error}')
