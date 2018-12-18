# flake8: NOQA
import json
from google.protobuf.empty_pb2 import Empty

import optuna
from optuna.protobuf import study_pb2
from optuna.protobuf import study_pb2_grpc
from optuna.protobuf import structs_pb2
from optuna.rpc.pruners import MedianPrunerParams
from optuna.rpc.pruners import message_to_pruner
from optuna.rpc.samplers import TPESamplerParams
from optuna.rpc.samplers import message_to_sampler
from optuna.rpc.structs import frozen_trial_to_message
from optuna.rpc.structs import message_to_frozen_trial
from optuna.rpc.structs import study_summary_to_message
from optuna.rpc.structs import message_to_study_summary
from optuna.structs import StudySummary


class StudyServicer(study_pb2_grpc.StudyServicer):
    def __init__(self):
        self.next_study_id = 0
        self.studies = {}

    def add_to_server(self, server):
        study_pb2_grpc.add_StudyServicer_to_server(self, server)

    def create_study(self, request, context):
        study = optuna.study.create_study(
            storage=none_if_empty(request.storage),
            sampler=message_to_sampler(request.sampler),
            pruner=message_to_pruner(request.pruner),
            study_name=none_if_empty(request.study_name),
            direction=request.direction,
            load_if_exists=request.load_if_exists,
        )
        study_id = self.next_study_id
        self.next_study_id += 1
        study_ref = study_pb2.StudyRef(session=request.session, study_id=study_id)
        self.studies[study_id] = StudyState(request.session, study_id, study)
        return study_ref

    def load_study(self, request, context):
        study = optuna.study.Study(
            storage=none_if_empty(request.storage),
            sampler=message_to_sampler(request.sampler),
            pruner=message_to_pruner(request.pruner),
            study_name=request.study_name,
            direction=request.direction,
        )
        study_id = self.next_study_id
        self.next_study_id += 1
        study_ref = study_pb2.StudyRef(session=request.session, study_id=study_id)
        self.studies[study_id] = StudyState(request.session, study_id, study)
        return study_ref

    def close_study(self, study_ref, context):
        del self.studies[study_ref.study_id]
        return Empty()

    def trials(self, study_ref, context):
        study = self.studies[study_ref.study_id]
        for trial in study.study.trials:
            yield frozen_trial_to_message(trial)

    def set_user_attr(self, request, context):
        study = self.studies[request.study.study_id]
        study.study.set_user_attr(request.key, json.loads(request.value.value))
        return Empty()

    def study_summary(self, study_ref, context):
        study = self.studies[study_ref.study_id].study
        n_trials = len(study.trials)
        if n_trials == 0:
            best_trial = None
        else:
            best_trial = study.best_trial
        summary = StudySummary(
            study_id=study.study_id,
            study_name=study.study_name,
            direction=study.direction,
            best_trial=best_trial,
            user_attrs=study.user_attrs,
            system_attrs=study.system_attrs,
            n_trials=n_trials,
            datetime_start=None
        )
        return study_summary_to_message(summary)

    def get_all_study_summaries(self, storage, context):
        pass
  # rpc get_all_study_summaries(google.protobuf.StringValue) returns (stream StudySummary) {}


class StudyState(object):
    def __init__(self, session_ref, study_id, study):
        self.session_ref = session_ref
        self.study_id = study_id  # FIXME: study_instance_id
        self.study = study


def none_if_empty(s):
    if s == "":
        return None
    return s


def create_study(session,
                 storage=None,
                 sampler=None,
                 pruner=None,
                 study_name=None,
                 direction='minimize',
                 load_if_exists=False):
    study = StudyClient(session)
    study._create_study(
        storage=storage, sampler=sampler, pruner=pruner, study_name=study_name,
        direction=direction, load_if_exists=load_if_exists)
    return study


def load_study(session,
               storage=None,
               sampler=None,
               pruner=None,
               study_name=None,
               direction='minimize'):
    study = StudyClient(session)
    study._load_study(
        storage=storage, sampler=sampler, pruner=pruner, study_name=study_name, direction=direction)
    return study


class StudyClient(object):
    def __init__(self, session_client):
        self.channel = session_client._channel()
        self.session_ref = session_client._session_ref()
        self.stub = study_pb2_grpc.StudyStub(self.channel)
        self.study_ref = None

    def __del__(self):
        # TODO(ohta): Support `with`
        if self.study_ref is not None:
            self.stub.close_study(self.study_ref)

    def _create_study(self,
                      storage=None,
                      sampler=None,
                      pruner=None,
                      study_name=None,
                      direction='minimize',
                      load_if_exists=False,
                      ):
        request = study_pb2.CreateStudyRequest(
            session=self.session_ref,
            study_name=study_name or "",
            sampler=(sampler or TPESamplerParams())._to_message(),
            pruner=(pruner or MedianPrunerParams())._to_message(),
            storage=storage or "",
            direction=direction,
            load_if_exists=load_if_exists
        )
        self.study_ref = self.stub.create_study(request)

    def _load_study(self,
                    study_name,
                    storage=None,
                    sampler=None,
                    pruner=None,
                    direction='minimize'
                    ):
        request = study_pb2.LoadStudyRequest(
            session=self.session_ref,
            study_name=study_name,
            sampler=(sampler or TPESamplerParams())._to_message(),
            pruner=(pruner or MedianPrunerParams())._to_message(),
            storage=storage or "",
            direction=direction,
        )
        self.study_ref = self.stub.create_study(request)

    def trials(self):
        assert self.study_ref is not None
        trials = []
        for trial in self.stub.trials(self.study_ref):
            trials.append(message_to_frozen_trial(trial))
        return trials

    def set_user_attr(self, key, value):
        assert self.study_ref is not None
        value = structs_pb2.JsonValue(value=json.dumps(value))
        request = study_pb2.SetUserAttrRequest(study=self.study_ref, key=key, value=value)
        self.stub.set_user_attr(request)

    def study_summary(self):
        assert self.study_ref is not None
        return message_to_study_summary(self.stub.study_summary(self.study_ref))

    def _get_all_study_summaries(self, storage):
        return self.stub.get_all_study_summaries(storage)
