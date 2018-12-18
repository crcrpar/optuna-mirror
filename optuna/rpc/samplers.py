import optuna
from optuna.protobuf import samplers_pb2
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler


def message_to_sampler(message):
    if message.HasField("tpe"):
        params = message.tpe
        gamma = params.gamma
        if gamma == 0:
            gamma = optuna.samplers.tpe.sampler.default_gamma
        weights = params.weights
        if weights == []:
            weights = optuna.samplers.tpe.sampler.default_weights
        return TPESampler(
            consider_prior=params.consider_prior,
            prior_weight=none_if_zero(params.prior_weight),
            consider_magic_clip=params.consider_magic_clip,
            consider_endpoints=params.consider_endpoints,
            n_startup_trials=params.n_startup_trials,
            n_ei_candidates=params.n_ei_candidates,
            gamma=gamma,
            weights=weights,
            seed=none_if_zero(params.seed)
        )
    if message.HasField("random"):
        params = message.random
        return RandomSampler(seed=none_if_zero(params.seed))
    raise ValueError()


def none_if_zero(n):
    if n == 0:
        return None
    return n


class RandomSamplerParams(object):
    def __init__(self, seed=None):
        self.seed = seed or 0  # FIXME

    def _to_message(self):
        message = samplers_pb2.Sampler()
        message.random.CopyFrom(samplers_pb2.RandomSampler(seed=self.seed))
        return message


class TPESamplerParams(object):
    def __init__(
            self,
            consider_prior=True,  # type: bool
            prior_weight=1.0,
            consider_magic_clip=True,  # type: bool
            consider_endpoints=False,  # type: bool
            n_startup_trials=10,  # type: int
            n_ei_candidates=24,  # type: int
            gamma=None,
            weights=[],
            seed=None
    ):
        self.consider_prior = consider_prior
        self.prior_weight = prior_weight or 0.0  # FIXME
        self.consider_magic_clip = consider_magic_clip
        self.consider_endpoints = consider_endpoints
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma or 0  # FIXME
        self.weights = weights
        self.seed = seed or 0  # FIXME

    def _to_message(self):
        message = samplers_pb2.Sampler()
        message.tpe.CopyFrom(samplers_pb2.TPESampler(
            consider_prior=self.consider_prior,
            prior_weight=self.prior_weight,
            consider_magic_clip=self.consider_magic_clip,
            consider_endpoints=self.consider_endpoints,
            n_startup_trials=self.n_startup_trials,
            n_ei_candidates=self.n_ei_candidates,
            gamma=self.gamma,
            weights=self.weights,
            seed=self.seed))
        return message
