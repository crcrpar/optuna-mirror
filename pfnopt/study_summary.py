import enum
from typing import Any
from typing import NamedTuple
from typing import Dict


class StudyTask(enum.Enum):

    NOT_SET = 0
    MINIMIZE = 1
    MAXIMIZE = 2


StudySummary = NamedTuple(
    'StudySummary',
    [('study_id', int),
     ('study_uuid', str),
     ('user_attrs', Dict[str, Any]),
     ('n_trials', int),
     ('task', StudyTask)])
