import abc

from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from study import pruners  # NOQA


class PrunerGenerator(object, abc.ABCMeta):

    @abc.abstractmethod
    def __len__(self):
        # type: () -> int

        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, study_index):
        # type: () -> pruners.BasePruner

        raise NotImplementedError
