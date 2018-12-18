from concurrent import futures
import grpc

import optuna

SESSION_CHECK_INTERVAL = 60


class RpcServer(object):
    def __init__(self, max_workers=10, bind_addr='localhost:50051'):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

        session_servicer = optuna.rpc.session.SessionServicer()
        session_servicer.add_to_server(self.server)

        study_servicer = optuna.rpc.study.StudyServicer()
        study_servicer.add_to_server(self.server)

        # TODO(ohta): Support secure connection.
        self.port = self.server.add_insecure_port(bind_addr)
        # self.timer = threading.Timer(SESSION_CHECK_INTERVAL, session_check, args=[servicer])
        # self.timer.start()

    # def __del__(self):
    #     self.timer.cancel()

    def start(self):
        self.server.start()

    def stop(self):
        self.server.stop(grace=False)

    def bind_port(self):
        return self.port


def session_check(servicer):
    servicer._check_session()
