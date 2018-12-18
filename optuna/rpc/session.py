from google.protobuf.duration_pb2 import Duration
from google.protobuf.empty_pb2 import Empty

from optuna.protobuf import session_pb2
from optuna.protobuf import session_pb2_grpc


class SessionServicer(session_pb2_grpc.SessionServicer):
    def __init__(self):
        self.next_session_id = 0
        self.sessions = {}

    def add_to_server(self, server):
        session_pb2_grpc.add_SessionServicer_to_server(self, server)

    def allocate_session(self, request, context):
        # TODO(ohta): handle timeout
        session_id = self.next_session_id
        self.next_session_id += 1
        session_ref = session_pb2.SessionRef(session_id=session_id)
        self.sessions[session_ref.session_id] = SessionState(request.keepalive_timeout)
        return session_ref

    def release_session(self, session_ref, context):
        # TODO(ohta): existance check
        del self.sessionss[session_ref.session_id]
        return Empty()

    def heartbeat_session(self, session_ref, context):
        # TODO(ohta): handle timeout
        return Empty()


class SessionState(object):
    def __init__(self, keepalive_timeout):
        self.keepalive_timeout = keepalive_timeout


class SessionClient(object):
    def __init__(self, channel, keepalive_timeout=60):
        self.channel = channel
        self.stub = session_pb2_grpc.SessionStub(self.channel)
        self.session_ref = None

        timeout = Duration()
        timeout.FromSeconds(keepalive_timeout)
        request = session_pb2.SessionAllocateRequest(keepalive_timeout=timeout)
        self.session_ref = self.stub.allocate_session(request)

    def __del__(self):
        if self.session_ref is not None:
            self.stub.release_session(self.session_ref)

    def heartbeat(self):
        self.stub.heartbeat_session(self.session_ref)

    def _channel(self):
        return self.channel

    def _session_ref(self):
        return self.session_ref
