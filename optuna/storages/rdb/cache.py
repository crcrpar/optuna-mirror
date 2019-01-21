import copy
from datetime import datetime
import threading
from types import TracebackType  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Tuple  # NOQA
from typing import Type  # NOQA

from optuna import structs  # NOQA


class TrialsCache(object):
    def __init__(self, cache_timeout=60):
        # type: (int) -> None

        self.cache_timeout = cache_timeout
        self.studies = {}  # type: Dict[int, StudyState]
        self.lock = threading.Lock()

    def lock_study(self, study_id):
        # type: (int) -> LockedStudyCache

        return LockedStudyCache(self, study_id)


class StudyState(object):
    def __init__(self):
        # type: () -> None

        self.last_access_time = datetime.now()
        self.cached_trials = {}  # type: Dict[int, structs.FrozenTrial]


class LockedStudyCache(object):
    def __init__(self, trials_cache, study_id):
        # type: (TrialsCache, int) -> None

        self.trials_cache = trials_cache
        self.study_id = study_id

    def __enter__(self):
        # type: () -> LockedStudyCache

        if self._is_cache_disabled():
            return self

        self.trials_cache.lock.__enter__()

        if self.study_id not in self.trials_cache.studies:
            self.trials_cache.studies[self.study_id] = StudyState()
            self._set_old_study_remove_timer(self.trials_cache.cache_timeout)

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # type: (Type[BaseException], Optional[Exception], TracebackType) -> None

        if self._is_cache_disabled():
            return

        self._study().last_access_time = datetime.now()

        self.trials_cache.lock.__exit__(exception_type, exception_value, traceback)

    def empty(self):
        # type: () -> bool

        if self._is_cache_disabled():
            return True

        return len(self._study().cached_trials) == 0

    def cache_trial_if_finished(self, trial):
        # type: (structs.FrozenTrial) -> None

        if self._is_cache_disabled():
            return

        if trial.state is not structs.TrialState.RUNNING:
            # We assume that the state of a finished trial is never updated anymore,
            # so we can safely cache it.
            self._study().cached_trials[trial.trial_id] = copy.deepcopy(trial)

    def find_cached_trial(self, trial_id):
        # type: (int) -> Optional[structs.FrozenTrial]

        if self._is_cache_disabled():
            return None

        return copy.deepcopy(self._study().cached_trials.get(trial_id))

    def _study(self):
        # type: () -> StudyState

        return self.trials_cache.studies[self.study_id]

    def _is_cache_disabled(self):
        # type: () -> bool

        return self.trials_cache.cache_timeout == 0

    def _set_old_study_remove_timer(self, timeout):
        # type: (float) -> None

        timer = threading.Timer(timeout, lambda: self._remove_study_if_old())
        timer.setDaemon(True)
        timer.start()

    def _remove_study_if_old(self):
        # type: () -> None

        now = datetime.now()
        with self.trials_cache.lock:
            elapsed = (now - self._study().last_access_time).total_seconds()
            if elapsed >= self.trials_cache.cache_timeout:
                del self.trials_cache.studies[self.study_id]
            else:
                timeout = self.trials_cache.cache_timeout - elapsed
                self._set_old_study_remove_timer(timeout)
