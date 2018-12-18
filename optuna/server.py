from concurrent import futures
import grpc
import optuna
from optuna.protobuf import study_pb2
from optuna.protobuf.study_pb2 import Empty
from optuna.protobuf import study_pb2_grpc
import time

import traceback


def start():
    max_workers = 10
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    study_pb2_grpc.add_StudyServicer_to_server(StudyServicer(), server)
    server.add_insecure_port('localhost:50051')
    server.start()
    while True:
        time.sleep(24 * 60 * 60)


def start_nb():
    max_workers = 10
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    study_pb2_grpc.add_StudyServicer_to_server(StudyServicer(), server)
    server.add_insecure_port('localhost:50051')
    server.start()
    return server


class StudyServicer(study_pb2_grpc.StudyServicer):
    def __init__(self):
        self.instance_id = 123
        self.active_studies = {}  # TODO(ohta): keepalive

    def create_study(self, request, context):
        print("** create_study(active={}): {}".format(len(self.active_studies), request))

        storage = request.storage
        if storage == "":
            storage = None

        if request.direction == 0:
            direction = 'minimize'
        else:
            direction = 'maximize'

        study_name = request.study_name
        if study_name == "":
            study_name = None

        sampler = pb_to_sampler(request.sampler)
        pruner = pb_to_pruner(request.pruner)
        try:
            study = optuna.study.create_study(
                storage=storage,
                sampler=sampler,
                pruner=pruner,
                study_name=study_name,
                direction=direction
            )
            instance_id = self.instance_id
            print("# Created: {}, {}".format(instance_id, study))
            self.active_studies[instance_id] = study
            self.instance_id += 1
            return study_pb2.StudyInstance(instance_id=instance_id)
        except Exception:
            print("** error")
            traceback.print_exc()
            raise

    def close_study(self, request, context):
        print("** close_study(active={}): {}".format(len(self.active_studies), request))
        del self.active_studies[request.instance_id]
        return Empty()

    def start_trial(self, request, context):
        print("** start_trial: {}".format(request))
        study = self.active_studies[request.instance_id]
        print("- STUDY: {} (trials={})".format(study, len(study.trials)))

        trial_id = study._create_new_trial().trial_id
        print("- TRIAL: {}".format(trial_id))

        return study_pb2.TrialInstance(study=request, trial_id=trial_id)

    def finish_trial(self, request, context):
        print("** finish_trial: {}".format(request))
        study = self.active_studies[request.trial.study.instance_id]
        study._complete_trial(request.trial.trial_id, request.value)
        return Empty()

    def suggest_uniform(self, request, context):
        print("** suggest_uniform: {}".format(request))
        study = self.active_studies[request.trial.study.instance_id]
        trial = study._get_trial(request.trial.trial_id)
        value = trial.suggest_uniform(request.parameter_name, request.low, request.high)
        return study_pb2.SuggestedDouble(value=value)

    def best_params(self, study, context):
        print("** best_params: {}".format(study))
        study = self.active_studies[study.instance_id]
        return study_pb2.BestParams(params=study.best_params)


def pb_to_sampler(pb):
    if pb.HasField("tpe"):
        raise
    if pb.HasField("random"):
        raise
    return None


def pb_to_pruner(pb):
    if pb.HasField("median"):
        raise
    return None
