import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class ConvbnBlock(nn.Module):
    """
    Convolution-BatchNorm block used in ResNet.
    Combines a 2D convolution layer with batch normalization.
    """

    def __init__(self, 
        in_channels:    int, 
        out_channels:   int, 
        kernel_size:    int, 
        stride:         int = 1
    ) -> None:
        """
        Initialize the ConvbnBlock.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolution kernel
            stride (int, optional): Stride of the convolution. Defaults to 1
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels     = in_channels,
                out_channels    = out_channels,
                kernel_size     = kernel_size,
                stride          = stride,
                padding         = 1,
                bias            = False
            ),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self,
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the ConvbnBlock.

        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, height', width']
        """
        return self.model(input)


class ResidualBlock(nn.Module):
    """
    Residual Block implementation for ResNet.
    Contains two convolutional layers with batch normalization and a skip connection.
    """

    def __init__(self,
        in_channels:    int,
        out_channels:   int,
        stride:         int = 1
    ) -> None:
        """
        Initialize the ResidualBlock.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Stride for first convolution. Affects output size. Defaults to 1
        """
        super().__init__()
        self.convbn1 = ConvbnBlock(
            in_channels     = in_channels,
            out_channels    = out_channels,
            kernel_size     = 3,
            stride          = stride
        )
        self.convbn2 = ConvbnBlock(
            in_channels     = out_channels,
            out_channels    = out_channels,
            kernel_size     = 3
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels     = in_channels,
                    out_channels    = out_channels,
                    kernel_size     = 1,
                    stride          = stride,
                    bias            = False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the ResidualBlock.

        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, height', width']
        """
        output = F.relu(self.convbn1(input))
        output = self.convbn2(output)
        output = F.relu(output + self.shortcut(input))
        return output
    

class ResNet(nn.Module):
    """
    ResNet architecture implementation for MNIST classification.
    A simplified ResNet variant with fewer layers, suitable for simpler tasks.
    Architecture:
    1. Initial conv+bn layer
    2. 3 residual blocks (with increasing channels: 16->32->64)
    3. Global average pooling
    4. Fully connected layer to num_classes
    """

    def __init__(self,
        num_class: int  = 10
    ) -> None:
        """
        Initialize the ResNet model.

        Args:
            num_class (int, optional): Number of output classes. Defaults to 10 for MNIST
        """
        super().__init__()
        self.convbn = ConvbnBlock(
            in_channels     = 1,
            out_channels    = 16,
            kernel_size     = 3
        )
        self.residual = nn.Sequential(
            ResidualBlock(16, 16, 1),
            ResidualBlock(16, 32, 2),
            ResidualBlock(32, 64, 2)
        )
        self.fc = nn.Linear(64, num_class)
    
    def forward(self,
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the ResNet model.

        Args:
            input (torch.Tensor): Input image tensor of shape [batch_size, 1, height, width]

        Returns:
            torch.Tensor: Class logits of shape [batch_size, num_class]
        """
        output = F.relu(self.convbn(input))
        output = self.residual(output)
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
    

if __name__ == '__main__':
    
    # Hyperparameters
    batch_size = 128
    epochs = 15
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    train_dataset = datasets.MNIST(
        root        = '../.././dataset',
        train       = True,
        transform   = transform,
        download    = True
    )

    valid_dataset = datasets.MNIST(
        root        = '../.././dataset',
        train       = False,
        transform   = transform,
        download    = True
    )

    train_loader = DataLoader(
        dataset     = train_dataset,
        batch_size  = batch_size,
        shuffle     = True
    )

    valid_loader = DataLoader(
        dataset     = valid_dataset,
        batch_size  = batch_size,
        shuffle     = False
    )
    # Network, Optimizer and LossFn
    net = ResNet(10).to(device)
    optimizer = optim.Adam(net.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(epochs):
        net.train() 
        train_loss = 0.0
        with tqdm(train_loader, total = len(train_loader), desc = f'Epoch [{epoch + 1:02d}/{epochs}]') as pbar:
            for batch_image, batch_label in pbar:
                batch_image: torch.Tensor
                batch_label: torch.Tensor
                batch_image, batch_label = batch_image.to(device), batch_label.to(device)
                optimizer.zero_grad()
                output = net(batch_image)
                loss: torch.Tensor = criterion(output, batch_label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({'Batch Loss': f'{loss.item():.6f}'})

        print(f'    ---- Epoch Loss: {train_loss / len(train_dataset):.6f}')

    # Saving weights   
    torch.save(net.state_dict(), './weight/resnet.MNIST.pth')
    print("Successfully saved.")

    # Validation
    net.eval()
    count = 0
    with torch.no_grad():
        for image, label in valid_loader:
            image: torch.Tensor
            label: torch.Tensor
            image, label = image.to(device), label.to(device)
            output: torch.Tensor = net(image)
            count += (output.max(1)[1] == label).sum().item()

    print(f'Correct Rate: {count/ len(valid_dataset):.6f}')
