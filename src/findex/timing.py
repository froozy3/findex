import functools
import logging
import time
from collections.abc import Callable

log = logging.getLogger("findex.timing")


def timed[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Log how long each call takes, in milliseconds."""

    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            log.info("%s took %.2f ms", function.__qualname__, (time.perf_counter() - started) * 1000)

    return wrapper
