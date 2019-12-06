from optuna import exceptions
from optuna import logging
from optuna import pruners
from optuna import samplers
from optuna import study
from optuna.study import Study
from optuna import storages
from optuna import structs
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Callable  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA
    from typing import Union  # NOQA

    from optuna.trial import Trial  # NOQA
    ObjectiveFuncType = Callable[[Trial], float]

logger = logging.get_logger(__name__)

_MINIMIZE = 'minimize'
_MAXIMIZE = 'maximize'


class StudyManager(study.BaseStudy):

    """A manager of studies.

    This objects manages some studies that use the same pruner with different configurations.
    Initially this is implemented for Hyperband pruner that runs internally multiple
    studies with SuccessiveHalving pruner.

    Args:
        study_name:
            The name of study. This is used as a prefix of each study's name.
            If ``None``, `pruner_name` will be the prefix.
        storage:

    """

    def __init__(
            self,
            study_name,  # type: Optional[str]
            storage,  # type: Union[str, storages.BaseStorage, None]
            sampler,  # type: Optional[samplers.BaseSampler]
            load_if_exists,  # type: bool
            direction,  # type: str
            pruner_generator,  # type: pruners.PrunerGenerator
            pruner_name  # type: str
    ):
        # type: (...) -> None

        assert load_if_exists or len(direction) > 0
        self.study_name = study_name
        # N.B. (crcrpar): This is dummy for the consistency with `BaseStudy`.
        self._study_id = -1
        self._storage = storages.get_storage(storage)
        self._direction = direction
        self._sampler = sampler or samplers.TPESampler()
        if self._direction == _MINIMIZE:
            self._cmp_func = lambda t1, t2: t1.value > t2.value
        elif self._direction == _MAXIMIZE:
            self._cmp_func = lambda t1, t2: t1.value < t2.value
        self._pruner_generator = pruner_generator
        self._pruner_name = pruner_name

        self._studies = []  # type: List[Study]

        if len(self._direction) == 0:
            raise ValueError("StudyManager is not set up correctly.")

    def _prepare_all_studies(self, load_if_exists, n_trials=None, timeout=None):
        # type: (bool, Optional[int], Optional[int]) -> None

        n_studies = len(self._pruner_generator)
        n_trials_per_study = None if n_trials is None else n_trials // n_studies
        timeout_per_study = None if timeout is None else timeout // n_studies
        pruner_name = self._pruner_name if self.study_name is None else self.study_name

        for study_idx in range(len(self._pruner_generator)):
            postfix = '_study_{}'.format(study_idx)
            study_name = pruner_name + postfix

            study = self._prepare_study(
                study_name, self._pruner_generator(study_idx), load_if_exists)
            self._studies.append(study)

    @property
    def best_params(self):
        # type: () -> Dict[str, Any]
        """Return parameters of the best trial in the study.

        Returns:
            A dictionary containing parameters of the best trial.
        """

        return self.best_trial.params

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
        best_trial = all_best_trials[0]
        if len(all_best_trials) == 1:
            return best_trial

        for trial in all_best_trials[1:]:
            if self._cmp_func(best_trial, trial):
                best_trial = trial
        return best_trial

    def _get_direction(self):
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
        trials = []
        for t in all_studies_trials:
            trials.extend(t)
        return trials

    @property
    def studies(self):
        # type: () -> List[Study]
        """Return the list of finished :class:`~optuna.study.Study`\\s."""

        return self._studies

    def optimize(
            self,
            func,  # type: ObjectiveFuncType
            n_trials=None,  # type: Optional[int]
            timeout=None,  # type: Optional[float]
            n_jobs=1,  # type: int
            catch=(),  # type: Union[Tuple[()], Tuple[Type[Exception]]]
            callbacks=None,  # type: Optional[List[Callable[[Study, structs.FrozenTrial], None]]]
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

        self._prepare_all_studies(load_if_exists, n_trials, timeout)

        for study_idx, study in enumerate(self.studies):
            logger.info("{}'s {}th bracket start.".format(self._pruner_name, study_idx))

            study.optimize(
                func=func,
                n_trials=n_trials_per_study,
                timeout=timeout_per_study,
                n_jobs=n_jobs,
                catch=catch,
                callbacks=callbacks,
                gc_after_trial=gc_after_trial
            )

    def _prepare_study(self, study_name, pruner, load_if_exists):
        # type: (str, pruners.BasePruner, bool) -> Study

        try:
            study_id = self._storage.create_new_study(study_name)
        except exceptions.DuplicatedStudyError:
            if load_if_exists:
                logger.info("Using an existing study with name '{}' instead of "
                            "creating a new one".format(sub_study_name))
                study_id = storage.get_study_id_from_name(study_name)
            else:
                raise

        study_name = self._storage.get_study_name_from_id(study_id)
        study = Study(
            study_name=study_name,
            storage=self._storage,
            sampler=self._sampler,
            pruner=pruner
        )
        if load_if_exists and len(self._direction) == 0:
            direction = study.direction
            if direction == structs.StudyDirection.MINIMIZE:
                self._direction = _MINIMIZE
                self._cmp_func = lambda t1, t2: t1.value > t2.value
            elif direction == structs.StudyDirection.MAXIMIZE:
                self._direction = _MAXIMIZE
                self._cmp_func = lambda t1, t2: t1.value < t2.value

        study._storage.set_study_direction(study_id, self._get_direction())
        return study
