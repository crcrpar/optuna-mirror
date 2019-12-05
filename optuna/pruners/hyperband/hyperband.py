from optuna import type_checking
from optuna import pruners

if type_checking.TYPE_CHECKING:
    from optuna import storages  # NOQA
    from optuna import Study  # NOQA
    from optuna.structs import FrozenTrial  # NOQA


class Hyperband(pruners.BasePruner):
    """This is a dummy class for consistency with the other pruners."""

    def __init__(
            self,
            min_resource,  # type: int
            reduction_factor,  # type: int
            min_early_stopping_rate_low,  # type: int
            min_early_stopping_rate_high  # type: int
    ):
        # type: (...) -> None
        self._min_resource
        self._reduction_factor = reduction_factor
        self._min_early_stopping_rate_low = min_early_stopping_rate_low
        self._min_early_stopping_rate_high = min_early_stopping_rate_high
        self._n_pruners = _min_early_stopping_rate_high - _min_early_stopping_rate_low
        self._current_resource_budget = 0
        self._current_n_pruners = 0

    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool
        """Please consult the documentation for :func:`BasePruner.prune`."""

        return False

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_n_pruners == self._n_pruners:
            raise StopIteration()

        config = self._generate_bracket_config(self._current_n_pruners)
        self._current_n_pruners == 1
        return config

    def _generate_bracket_config(self, bracket_index):
        # type: (int) -> Dict[str, int]
        """Returns dict that can be passed to SuccessiveHalvingPruner."""

        self._current_resource_budget += self._get_resource_badget(bracket_index)
        min_early_stopping_rate = self._min_early_stopping_rate_low + bracket_index

        return {
            'min_resource': self._min_resource,
            'reduction_factor': self._reduction_factor,
            'min_early_stopping_rate': min_early_stopping_rate,
        }

    def _get_resource_badget(self, pruner_index):
        # type: (int) -> int
        """Calculate budget for pruner of given index."""

        n = self._reduction_factor ** self._n_pruners
        budget = n
        for i in range(pruner_index, self._n_pruners - 1):
            budget += n / 2
        return budget

    def use_study_manager(self):
        # type: () -> bool
        """Return this pruner requires `StudyManager` or not."""

        return False

    @property
    def current_resource_budget(self):
        return self._current_resource_budget
