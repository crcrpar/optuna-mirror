import grpc
from optuna.protobuf import study_pb2
from optuna.protobuf import study_pb2_grpc


class Client(object):
    def __init__(self):
        self.channel = grpc.insecure_channel('localhost:50051')

    def create_study(self, storage=None):
        stub = study_pb2_grpc.StudyStub(self.channel)
        options = study_pb2.CreateStudyOptions(storage=storage)
        return stub.create_study(options)

    def close_study(self, study):
        stub = study_pb2_grpc.StudyStub(self.channel)
        stub.close_study(study)

    def start_trial(self, study):
        stub = study_pb2_grpc.StudyStub(self.channel)
        return stub.start_trial(study)

    def suggest_uniform(self, trial, parameter_name, low, high):
        stub = study_pb2_grpc.StudyStub(self.channel)
        request = study_pb2.SuggestUniformRequest(
            trial=trial, parameter_name=parameter_name, low=low, high=high)
        return stub.suggest_uniform(request)

    def finish_trial(self, trial, value):
        stub = study_pb2_grpc.StudyStub(self.channel)
        request = study_pb2.FinishTrialRequest(trial=trial, value=value)
        return stub.finish_trial(request)

    def best_params(self, study):
        stub = study_pb2_grpc.StudyStub(self.channel)
        return dict(stub.best_params(study).params)
