import asyncio
from federate.server import Server
from models.mlp import mlp

server = Server(mlp())
asyncio.run(server.start())