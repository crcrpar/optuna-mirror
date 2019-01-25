import numpy as np
import pytest
import typing  # NOQA

import optuna
from optuna.distributions import BaseDistribution  # NOQA
from optuna.storages import BaseStorage  # NOQA


parametrize_sampler = pytest.mark.parametrize(
    'sampler_class',
    [optuna.samplers.RandomSampler, optuna.samplers.TPESampler]
)


@parametrize_sampler
def test_uniform(sampler_class):
    # type: (typing.Callable[[], optuna.samplers.BaseSampler]) -> None

    storage = optuna.storages.get_storage(None)
    study_id = storage.create_new_study_id()

    sampler = sampler_class()
    distribution = optuna.distributions.UniformDistribution(-1., 1.)
    points = np.array([sampler.sample(storage, study_id, 'x', distribution) for _ in range(100)])
    assert np.all(points >= -1.)
    assert np.all(points < 1.)


@parametrize_sampler
def test_discrete_uniform(sampler_class):
    # type: (typing.Callable[[], optuna.samplers.BaseSampler]) -> None

    sampler = sampler_class()

    # Test to sample integer value: q = 1
    storage = optuna.storages.get_storage(None)
    study_id = storage.create_new_study_id()

    distribution = optuna.distributions.DiscreteUniformDistribution(-10., 10., 1.)
    points = np.array([sampler.sample(storage, study_id, 'x', distribution) for _ in range(100)])
    assert np.all(points >= -10)
    assert np.all(points <= 10)
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)

    # Test to sample quantized floating point value: [-10.2, 10.2], q = 0.1
    distribution = optuna.distributions.DiscreteUniformDistribution(-10.2, 10.2, 0.1)
    points = np.array([sampler.sample(storage, study_id, 'y', distribution) for _ in range(100)])
    assert np.all(points >= -10.2)
    assert np.all(points <= 10.2)
    round_points = np.round(10 * points)
    np.testing.assert_almost_equal(round_points, 10 * points)


@parametrize_sampler
def test_int(sampler_class):
    # type: (typing.Callable[[], optuna.samplers.BaseSampler]) -> None

    sampler = sampler_class()
    storage = optuna.storages.get_storage(None)
    study_id = storage.create_new_study_id()

    distribution = optuna.distributions.IntUniformDistribution(-10, 10)
    points = np.array([sampler.sample(storage, study_id, 'x', distribution) for _ in range(100)])
    assert np.all(points >= -10)
    assert np.all(points <= 10)
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)


class BeforeAfterCheckSampler(optuna.samplers.BaseSampler):

    def __init__(self):
        # type: () -> None

        self.before_call_count = 0
        self.after_call_count = 0

    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, BaseDistribution) -> float

        return 1.0

    def before(self, trial_id):
        # type: (int) -> None

        self.before_call_count += 1

    def after(self, trial_id):
        # type: (int) -> None

        self.after_call_count += 1


@pytest.mark.parametrize('objective_args', [
    (1.0, None),
    (float('nan'), None),
    (1.0, optuna.structs.TrialPruned()),
    (1.0, ValueError())
])
def test_before_after(objective_args):
    # type: (typing.Tuple[float, typing.Optional[Exception]]) -> None

    def objective(_, return_value, exception):
        # type: (optuna.trial.Trial, float, typing.Optional[Exception]) -> float

        if exception is not None:
            raise exception

        return return_value

    sampler = BeforeAfterCheckSampler()
    study = optuna.create_study(sampler=sampler)

    assert sampler.before_call_count == 0
    assert sampler.after_call_count == 0

    study._run_trial(lambda x: objective(x, *objective_args), catch=(Exception, ))

    assert sampler.before_call_count == 1
    assert sampler.after_call_count == 1
