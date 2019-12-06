from optuna import type_checking
from optuna import pruners

if type_checking.TYPE_CHECKING:
    from optuna.structs import FrozenTrial  # NOQA


class SuccessiveHalvingPrunerGenerator(PrunerGenerator):
    """Generator of SuccessiveHalving pruner for Hyperband."""

    def __init__(
        self,
        min_resource,  # type: int
        reduction_factor,  # type: int
        min_early_stopping_rate_low,  # type: int
        min_early_stopping_rate_high  # type: int
    ):
        # type: (...) -> None

        self.min_resource = min_resource
        self.reduction_factor = reduction_factor
        self.min_early_stopping_rate_low = min_early_stopping_rate_low
        self.min_early_stopping_rate_high = min_early_stopping_rate_high
        self.n_pruners = self.min_early_stopping_rate_high - self.min_early_stopping_rate_low

    def __len__(self):
        # type: () -> int

        return self.n_pruners

    def __call__(self, study_index):
        # type: (int) -> pruners.SuccessiveHalvingPruner
        """Create a pruner according to the given index of study.

        Args:
            study_index:
                The index of the study that uses the pruner that this method creates.

        Returns:
            A pruner whose configuration is up to `study_index`.
        """

        min_early_stopping_rate = self.min_early_stopping_rate_low + study_index

        return pruners.SuccessiveHalvingPruner(
            min_resource=self.min_resource,
            reduction_factor=self.reduction_factor,
            min_early_stopping_rate=min_early_stopping_rate
        )
