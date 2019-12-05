from optuna import logging
from optuna.pruners import SuccessiveHalvingPruner
from optuna import study as study_module
from optuna import storages
from optuna import structs
from optuna.pruners.hyperband.hyperband import Hyperband
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA

    from optuna.trial import Trial  # NOQA
    ObjectiveFuncType = Callable[[Trial], float]

logger = logging.get_logger(__name__)

_MINIMIZE = 'minimize'
_MAXIMIZE = 'maximize'


class StudyManager(object):

    """This class manages brackets."""

    # TOOD(crcrpar): Support `load_study`
    def __init__(self, storage, direction, hyperband):
        # type: (storages.BaseStorage, str, Hyperband) -> None

        self._storage = storage
        self._direction = direction
        self._cmp_func = min if self._direction == _MINIMIZE else max
        self._hyperband = hyperband
        self._bracket_configs = []  # type:
        self._studies = []  # type: List[study_module.Study]

    @property
    def best_params(self):
        # type: () -> Dict[str, Any]
        """Return parameters of the best trial in the study.

        Returns:
            A dictionary containing parameters of the best trial.
        """

        return self.best_trial.parms

    @property
    def best_value(self):
        # type: () -> float
        """Return the best objective value in the study.

        Returns:
            A float representing the best objective value.
        """

        best_value = self.best_trial.value
        assert best_value is not None

        return best_value

    @property
    def best_trial(self):
        # type: () -> structs.FrozenTrial
        """Return the best trial in the study.

        Returns:
            A :class:`~optuna.structs.FrozenTrial` object of the best trial.
        """

        all_best_trials = [study.best_trial for study in self._studies]
        return self._cmp_func(all_best_trials)

    @property
    def direction(self):
        # type: () -> structs.StudyDirection
        """Return the direction of the study.

        Returns:
            A :class:`~optuna.structs.StudyDirection` object.
        """

        if self._direction == _MINIMIZE:
            return structs.StudyDirection.MINIMIZE
        else:
            return structs.StudyDirection.MAXIMIZE

    @property
    def trials(self):
        # type: () -> List[structs.FrozenTrial]
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        Returns:
            A list of :class:`~optuna.structs.FrozenTrial` objects.
        """

        all_studies_trials = [study.trials for study in self._studies]

        return [item for all_studies_trials in l for item in all_studies_trials]

    def optimize(
            self,
            func,  # type: ObjectiveFuncType
            n_trials=None,  # type: Optional[int]
            timeout=None,  # type: Optional[float]
            n_jobs=1,  # type: int
            catch=(),  # type: Union[Tuple[()], Tuple[Type[Exception]]]
            callbacks=None,  # type: Optional[List[Callable[[Strudy, structs.FrozenTrial], None]]
            gc_after_trial=True  # type: bool
    ):
        # type: (...) -> None
        """Optimize an objective function.

        Args:
            func:
                A callable that implements objective function.
            n_trials:
                The number of trials. If this argument is set to :obj:`None`, there is no
                limitation on the number of trials. If :obj:`timeout` is also set to :obj:`None`,
                the study continues to create trials until it receives a termination signal such
                as Ctrl+C or SIGTERM.
            timeout:
                Stop study after the given number of second(s). If this argument is set to
                :obj:`None`, the study is executed without time limitation. If :obj:`n_trials` is
                also set to :obj:`None`, the study continues to create trials until it receives a
                termination signal such as Ctrl+C or SIGTERM.
            n_jobs:
                The number of parallel jobs. If this argument is set to :obj:`-1`, the number is
                set to CPU count.
            catch:
                A study continues to run even when a trial raises one of the exceptions specified
                in this argument. Default is an empty tuple, i.e. the study will stop for any
                exception except for :class:`~optuna.exceptions.TrialPruned`.
            callbacks:
                List of callback functions that are invoked at the end of each trial.
            gc_after_trial:
                Flag to execute garbage collection at the end of each trial. By default, garbage
                collection is enabled, just in case. You can turn it off with this argument if
                memory is safely managed in your objective function.
        """

        if not isinstance(catch, tuple):
            raise TypeError("The catch argument is of type \'{}\' but must be a tuple.".format(
                type(catch).__name__))

        if n_trials is None and timeout is None:
            raise ValueError(
                "Study with {} pruner requires either `n_trials` or `timeout`".format(
                    self.pruner_name))

        n_trials_per_bracket = None if n_trials is None else n_trials // len(self._bracket_configs)
        timeout_per_bracket = None if timeout is None else timeout // len(self._bracket_configs)
        study_name_prefix = self.pruner_name if self.study_name is None else self.study_name

        for bracket_index, bracket_config in enumerate(self._hyperband):
            logger.info("Hyperband's {}th bracket start.".format(bracket_index))
            postfix = '_bracket_{}'.format(bracket_id)
            bracket_study_name = study_name_prefix + postfix
            study = study_module.create_study(
                storage=self._storage,
                sampler=sampler,
                pruner=SuccessiveHalvingPruner(**bracket_config),
                study_name=bracket_study_name,
                direction=self._direction,
                load_if_exists=load_if_exists
            )
            self._studies.append(study)

            study.optimize(
                func=func,
                n_trials=n_trials_per_bracket,
                timeout=timeout_per_bracket,
                n_jobs=n_jobs,
                catch=catch,
                callbacks=callbacks,
                gc_after_trial=gc_after_trial
            )
