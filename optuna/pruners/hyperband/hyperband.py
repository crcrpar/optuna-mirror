from optuna import type_checking
from optuna import pruners
from optuna.pruners.hyperband.pruner_generator import SuccessiveHalvingPrunerGenerator

if type_checking.TYPE_CHECKING:
    from typing import Optional  # NOQA

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

    def prepare_pruner_generator(self):
        # type: () -> Optional[Callable[[int], pruners.BasePruner]]
        """Prepare a pruner generator."""

        return SuccessiveHalvingPrunerGenerator(
            self._min_resource,
            self._reduction_factor,
            self._min_early_stopping_rate_low,
            self._min_early_stopping_rate_high
        )
