# Federated Learning Framework

A flexible and extensible asynchronous federated learning framework implemented in Python using PyTorch and asyncio. The framework supports different model architectures and is demonstrated with MLP and ResNet examples.

## Features

- Asynchronous federated learning implementation
- Support for multiple model architectures (MLP, ResNet)
- Configurable server and client settings
- CUDA support for GPU acceleration
- Modular design for easy extension
- Example implementations with synthetic data (MLP) and MNIST dataset (ResNet)

## Project Structure

```
src/
├── dataset/         # Dataset implementations
├── federate/        # Core federated learning components
├── models/          # Neural network models
├── utils/           # Utility functions and configurations
test/
├── mlp/            # MLP model test example
└── resnet/         # ResNet model test example with MNIST
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision (for ResNet MNIST example)
- PyYAML
- tqdm

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Federated-SleepNet
```

2. Install the required packages:
```bash
pip install torch torchvision tqdm pyyaml
```

## Running the Examples

The framework includes two example implementations: MLP with synthetic data and ResNet with MNIST dataset.

### Running MLP Example

1. Start the server:
```bash
cd test/mlp
python run_server.py
```

2. Start one or more clients (in separate terminals):
```bash
cd test/mlp
python run_client.py
```

### Running ResNet Example

1. Start the server:
```bash
cd test/resnet
python run_server.py
```

2. Start one or more clients (in separate terminals):
```bash
cd test/resnet
python run_client.py
```

## Configuration

Both server and client configurations can be customized through YAML files located in the `test/[model]/config/` directories.

### Server Configuration (`server_config.yaml`)
```yaml
server:
  host: 127.0.0.1
  port: 9000

train:
  device: cuda        # Use 'cuda' for GPU, 'cpu' for CPU
  num_clients: 5      # Number of clients to wait for
  timeout: 90        # Timeout for client responses
  num_rounds: 100    # Number of training rounds
  round_sep: 5       # Separation time between rounds
```

### Client Configuration (`client_config.yaml`)
```yaml
server:
  host: 127.0.0.1
  port: 9000

train:
  device: cuda           # Use 'cuda' for GPU, 'cpu' for CPU
  num_epochs: 40        # Number of local training epochs
  learning_rate: 0.001  # Learning rate for optimization
  batch_size: 30        # Batch size for training
  optimizer: Adam       # Optimizer (Adam, SGD, etc.)
  criterion: MSELoss    # Loss function
```

## Extending the Framework

The framework is designed to be easily extensible:

1. Add new models in the `src/models/` directory
2. Create custom datasets in the `src/dataset/` directory
3. Modify federated learning algorithms in `src/utils/fedavg.py`
4. Configure training parameters in the YAML configuration files

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]