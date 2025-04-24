import yaml
import asyncio
import pickle
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from dataset.linear import linearDataset
from tqdm import tqdm
from utils import conf

class Client():
  """
  Federated Learning Client class that handles local training and model updates.
  """

  def __init__(self,
    model: nn.Module,
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

    self.dataset = self.get_dataset()
    self.loader = DataLoader(self.dataset, self.batch_size, shuffle = True)
  

  def get_dataset(self) -> Dataset:
    """
    Get the dataset for training.

    Returns:
        DataLoader: The dataset loader
    """
    return linearDataset()

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
      serialized_data = pickle.dumps(state_dict)
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
      stream_length = await reader.read(4)
      if not stream_length:
        raise ConnectionError

      stream_length = int.from_bytes(stream_length, 'big')

      serialized_data = await reader.read(stream_length)
      return pickle.loads(serialized_data)

    except Exception as error:
      print(f'Exception: {error}')


  async def train(self) -> None:
    """
    Train the model for a specified number of epochs.
    """
    self.model.train()

    for epoch in range(self.num_epochs):
      epoch_loss = 0.0
      with tqdm(self.loader, desc = f'{epoch + 1:02d}/{self.num_epochs}',total = len(self.loader)) as bar:
        for input, ouput in bar:
          input: torch.Tensor; ouput: torch.Tensor
          input, ouput = input.to(self.device), ouput.to(self.device)
          _ouput_: torch.Tensor = self.model(input)
          _ouput_ = _ouput_.squeeze(1, 2)
          loss: torch.Tensor = self.criterion(_ouput_, ouput)
          epoch_loss += loss.item() * input.size(0)
          loss.backward()
          self.optimizer.step()
          self.optimizer.zero_grad()
        
      print(f'Epoch: {epoch}, Loss: {epoch_loss / len(self.dataset)}')


  async def start(self) -> None:
    """
    Start the client training process and communicate with the server.
    """
    reader, writer = await asyncio.open_connection(self.server_host, self.server_port)
    print(f'Successfully connected to server {self.server_host}:{self.server_port}')
    
    try: 
      while True:
        self.model.load_state_dict(await self.recv_model_params(reader))
        await self.train()
        await self.send_model_params(writer, self.model.state_dict())

    except asyncio.CancelledError as error:
      print(f'Exception: {error}')
