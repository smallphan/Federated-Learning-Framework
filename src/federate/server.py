import torch
import torch.nn as nn
import asyncio
import pickle
from utils import fedavg
import yaml

class Server():
  """
  Asynchronous Federated Learning Server class that manages model distribution and aggregation.
  Handles multiple client connections and performs asynchronous model updates.
  """

  def __init__(self,
    model: nn.Module,
    config_file: str = './config/server_config.yaml'
  ) -> None:
    """
    Initialize the FL server.
    
    Args:
      model (nn.Module): The PyTorch model to be trained
      host (str): Server host address
      port (int): Server port number
    """
    with open(config_file, 'r') as file:       
      config = yaml.safe_load(file)

    self.device = 'cuda' if (config['train']['device'] == 'cuda' and torch.cuda.is_available()) else 'cpu'
    self.model = model.to(self.device)

    self.host:        str = config['server']['host']
    self.port:        int = config['server']['port']
    self.psword:      str = str(config['server']['psword'])

    self.timeout:     int = config['train']['timeout']
    self.num_rounds:  int = config['train']['num_rounds']
    self.num_clients: int = config['train']['num_clients']
    self.round_sep:   int = config['train']['round_sep']

    self.clients = {}
    self.model_list = []


  async def handle_client(self,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter
  ) -> None:
    """
    Handle a new client connection.
    
    Args:
      reader (asyncio.StreamReader): Stream reader for the client connection
      writer (asyncio.StreamWriter): Stream writer for the client connection
    """
    address = writer.get_extra_info('peername')
    self.clients[address] = (reader, writer)
    print(f'Client {address} connected.')

    try:
      await writer.wait_closed()
    
    except asyncio.CancelledError as error:
      print(f'handle_client exception: {error}')
    
    finally:
      print(f'Client {address} disconnected.')
      del self.clients[address]
      writer.close()
      await writer.wait_closed()


  async def send_model_params(self, 
    writer: asyncio.StreamWriter,
    state_dict: dict
  ) -> None:
    """
    Send model parameters to a client.
    
    Args:
      writer (asyncio.StreamWriter): Stream writer for the client connection
      state_dict (dict): Model state dictionary to send
    """
    try:
      serialized_data = fedavg.model_encode(
        pickle.dumps(
          fedavg.model_quantization(state_dict)
        ),
        self.psword
      )
      writer.write(len(serialized_data).to_bytes(4, 'big'))
      await writer.drain()
      writer.write(serialized_data)
      await writer.drain()

    except Exception as error:
        print(f'send_model_params exception: {error}')


  async def recv_model_params(self,
    reader: asyncio.StreamReader
  ) -> dict:
    """
    Receive model parameters from a client.
    
    Args:
      reader (asyncio.StreamReader): Stream reader for the client connection
    
    Returns:
      dict: Model state dictionary received from the client
    """
    try:
      stream_length = await reader.readexactly(4)
      if not stream_length:
        raise ConnectionError
      
      stream_length = int.from_bytes(stream_length, 'big')

      serialized_data = await reader.readexactly(stream_length)
      return fedavg.model_dequantization(
        pickle.loads(
          fedavg.model_decode(
            serialized_data,
            self.psword
          )
        )
      )

    except Exception as error:
      print(f'recv_model_params exception: {error}')


  async def broadcast(self,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter
  ) -> None:
    """
    Broadcast the current model parameters to a client and receive updated parameters.
    
    Args:
      reader (asyncio.StreamReader): Stream reader for the client connection
      writer (asyncio.StreamWriter): Stream writer for the client connection
    """
    try:
      await self.send_model_params(writer, self.model.state_dict())
      self.model_list.append(await self.recv_model_params(reader))

    except Exception as error:
      print(f'broadcast exception: {error}')


  async def train(self) -> None:
    """
    Train the global model by aggregating updates from clients.
    """
    for round in range(self.num_rounds):

      await asyncio.sleep(self.round_sep)

      print(f'Round {round}:')

      tasks_list = []
      self.model_list = []

      for index, (address, (reader, writer)) in enumerate(self.clients.items()):
        reader: asyncio.StreamReader
        writer: asyncio.StreamWriter

        tasks_list.append(self.broadcast(reader, writer))

      try:
        
        async with asyncio.timeout(self.timeout):
          await asyncio.gather(*tasks_list)
        
        if len(self.model_list) != 0:
          self.model.load_state_dict(
            fedavg.aggregate_state_dicts(self.model_list)
          )


      except asyncio.TimeoutError as error:
        pass

      except Exception as error:       
        print(f'train exception: {error}')


  async def start_server(self) -> None:
    """
    Start the server and listen for incoming client connections.
    """
    server = await asyncio.start_server(self.handle_client, self.host, self.port, limit = 1048576)
    print(f'Server has been started, listening on {self.host}:{self.port}.')
    async with server:
      await server.serve_forever()


  async def start(self) -> None:
    """
    Start the server and the training process.
    """
    try:
      await asyncio.gather(
        self.train(),
        self.start_server()
      )
    
    except Exception as error:
      print(f'Exception: {error}')
