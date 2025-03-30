from server import Server
from models.mlp import mlp

server = Server(mlp(3, 5, 1), 10)


