from typing import Any, List, Callable, Optional
import joblib

import optuna
from optuna import logging

_THREAD = 'threading_callback'
_MULTIPROCESS = 'multiprocessing_callback'


class _JoblibBackendCallbacks(object):

    def __init__(self, *callbacks: List[Callable[Any, Any]]) -> None:
        self.callbacks = [cb for cb in callbacks if callable(cb)]

    def __call__(self, out: Any) -> None:
        for cb in self.callbacks:
            cb(out)


class _JoblibBackend(object):

    def _set_progress_bar(
        self,
        progress_bar: optuna.progress_bar._ProgressBar,
    ) -> None:
        self._progress_bar = progress_bar

    def callback(self, results: List[Any]) -> None:
        for result in results:
            self._progress_bar.update(1)
            if isinstance(result, float):
                # Assumes the result is elapsed_seconds from `Study._optimize_sequential`.
                self._progress_bar.set_postfix_str(
                    '{:.02f}/{} seconds'.format(result, self._progress_bar._timeout))

    def apply_async(self, func: Callable, callback: Optional[Callable] = None) -> None:
        return super().apply_async(func, _JoblibBackendCallbacks(callback, self.callback))


class ThreadingCallbackBackend(joblib._parallel_backends.ThreadingBackend, _JoblibBackend):

    pass


class MultiprocessingCallbackBackend(
        joblib._parallel_backends.MultiprocessingBackend, _JoblibBackend):

    pass


joblib.register_parallel_backend(
    'multiprocessing_callback', MultiprocessingCallbackBackend)


def _register_parallel_backend(active_backend: joblib.parallel.parallel_backend) -> None:
    if isinstance(active_backend, joblib.parallel.ThreadingBackend):
        joblib.register_parallel_backend(_THREAD, ThreadingCallbackBackend)
    elif isinstance(active_backend, joblib.parallel.MultiprocessingBackend):
        joblib.register_parallel_backend(_MULTIPROCESS, MultiprocessingCallbackBackend)


def _get_prefer(active_backend: joblib.parallel.parallel_backend) -> str:
    if isinstance(active_backend, joblib.parallel.ThreadingBackend):
        return _THREAD
    elif isinstance(active_backend, joblib.parallel.MultiprocessingBackend):
        return _MULTIPROCESS
