import asyncio
from federate.client import Client
import torch
import torch.nn as nn
from models.resnet import ResNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Optimizer
from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

dataset = datasets.MNIST(
  root      = './data',
  train     = True,
  transform = transform,
  download  = True
)

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
    with tqdm(loader, desc = f'Epoch [{epoch + 1:02d}/{num_epochs}]', total = len(loader)) as bar:
      for batch_image, batch_label in bar:
        batch_image: torch.Tensor; batch_label: torch.Tensor
        batch_image, batch_label = batch_image.to(device), batch_label.to(device)

        output = model(batch_image)
        loss: torch.Tensor = criterion(output, batch_label)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        bar.set_postfix({'Batch Loss': f'{loss.item():.6f}'})

    print(f'    ---- Epoch Loss: {epoch_loss / len(dataset):.6f}')

client = Client(ResNet(10), dataset, train)
asyncio.run(client.start())