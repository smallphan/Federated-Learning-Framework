import asyncio
from federate.client import Client
import torch
import torch.nn as nn
from models.mlp import mlp
from dataset.mlp import mlpDataset
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm

def train(
  model: nn.Module,
  num_epochs: int,
  loader: DataLoader,
  device: str,
  criterion: nn.Module,
  optimizer: Optimizer
) -> None:
  model.train()

  for epoch in range(num_epochs):
    epoch_loss = 0.0
    with tqdm(loader, desc = f'{epoch + 1:02d}/{num_epochs}', total = len(loader)) as bar:
      for input, ouput in bar:
        input: torch.Tensor; ouput: torch.Tensor
        input, ouput = input.to(device), ouput.to(device)
        _ouput_: torch.Tensor = model(input)
        _ouput_ = _ouput_.squeeze(1, 2)
        loss: torch.Tensor = criterion(_ouput_, ouput)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f'Epoch: {epoch}, Loss: {epoch_loss / len(loader)}')


client = Client(mlp(), mlpDataset(), train)
asyncio.run(client.start())