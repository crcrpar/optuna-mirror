from optuna.protobuf import pruners_pb2
from optuna.pruners import MedianPruner


class MedianPrunerParams(object):
    def __init__(self, n_startup_trials=5, n_warmup_steps=0):
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps

    def _to_message(self):
        message = pruners_pb2.Pruner()
        message.median.CopyFrom(
            pruners_pb2.MedianPruner(n_startup_trials=self.n_startup_trials,
                                     n_warmup_steps=self.n_warmup_steps))
        return message


def message_to_pruner(message):
    if message.HasField("median"):
        params = message.median
        return MedianPruner(n_startup_trials=params.n_startup_trials,
                            n_warmup_steps=params.n_warmup_steps)
    raise ValueError()
