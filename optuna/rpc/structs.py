# flake8: NOQA
from optuna.protobuf import structs_pb2


def frozen_trial_to_message(trial):
    message = structs_pb2.FrozenTrial(
        trial_id=trial.trial_id,
        state=from_trial_state(trial.state),
        value=trial.value,
        datetime_start=trial.datetime_start,
        datetime_complete=trial.datetime_complete,
        params=to_str_json_map(trial.params),
        user_attrs=to_str_json_map(trial.user_attrs),
        system_attrs=to_str_json_map(trial.system_attrs),
        intermediate_values=to_int_double_map(trial.intermediate_values),
        params_in_internal_repr=to_int_double_map(trial.params_in_internal_repr)
    )
    return message


def message_to_frozen_trial(message):
    trial = FrozenTrial(
        trial_id=message.trial_id,
        value=message.value,
        state=to_trial_state(message.state),
        user_attrs=from_str_json_map(message.user_attrs),
        system_attrs=from_str_json_map(message.system_attrs),
        params=from_str_json_map(message.params),
        intermediate_values=from_int_double_map(message.intermediate_values),
        params_in_internal_repr=from_int_double_map(message.params_in_internal_repr),
        datetime_start=message.datetime_start,
        datetime_complete=message.datetime_complete
    ),
    return trial


def study_summary_to_message(summary):
    message = structs_pb2.StudySummary(
        study_id=summary.study_id,
        study_name=summary.study_name,
        direction=summary.direction,
        best_trial=summary.best_trial,
        user_attrs=summary.user_attrs,
        system_attrs=summary.system_attrs,
        n_trials=summary.n_trials,
        datetime_start=summary.datetime_start)
    return messsage


def message_to_study_summary(message):
    pass
