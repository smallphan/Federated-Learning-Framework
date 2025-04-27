import asyncio
from federate.server import Server
from models.resnet import ResNet

server = Server(ResNet(10))
asyncio.run(server.start())