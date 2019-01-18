import copy
from datetime import datetime
from datetime import timedelta
import heapq
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Tuple  # NOQA

from optuna import structs  # NOQA


class TrialCache(object):
    def __init__(self, cache_timeout=60):
        # type: (int) -> None

        self.cache_timeout = cache_timeout
        self.studies = {}  # type: Dict[int, StudyState]
        self.timeout_queue = []  # type: List[Tuple[datetime, int]]

    def is_known_study(self, study_id):
        # type: (int) -> bool

        return study_id in self.studies

    def remove_old_cache(self):
        # type: () -> None

        expiry_time = datetime.now() - timedelta(seconds=self.cache_timeout)
        while len(self.timeout_queue) > 0 and self.timeout_queue[0][0] < expiry_time:
            study_id = heapq.heappop(self.timeout_queue)[1]
            last_access_time = self.studies[study_id].last_access_time
            if last_access_time < expiry_time:
                del self.studies[study_id]
                continue

            heapq.heappush(self.timeout_queue, (last_access_time, study_id))

    def update_last_access_time(self, study_id):
        # type: (int) -> None

        if study_id not in self.studies:
            heapq.heappush(self.timeout_queue, (datetime.now(), study_id))
            self.studies[study_id] = StudyState()
        else:
            self.studies[study_id].last_access_time = datetime.now()

    def find_cached_trial(self, study_id, trial_id):
        # type: (int, int) -> Optional[structs.FrozenTrial]

        if study_id not in self.studies:
            return None

        if trial_id not in self.studies[study_id].cached_trials:
            return None

        return copy.deepcopy(self.studies[study_id].cached_trials[trial_id])

    def cache_trial(self, study_id, trial):
        # type: (int, structs.FrozenTrial) -> None

        if self.cache_timeout == 0:
            return

        trials = self.studies[study_id].cached_trials
        trials[trial.trial_id] = copy.deepcopy(trial)


class StudyState(object):
    def __init__(self):
        # type: () -> None

        self.last_access_time = datetime.now()
        self.cached_trials = {}  # type: Dict[int, structs.FrozenTrial]
