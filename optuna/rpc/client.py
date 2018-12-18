import grpc

from optuna.rpc.session import SessionClient


class RpcClient(object):
    def __init__(self, server_addr='localhost:50051'):
        self.channel = grpc.insecure_channel(server_addr)

    def create_session(self):
        return SessionClient(self.channel)
