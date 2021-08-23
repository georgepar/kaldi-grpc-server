import functools
import time
from typing import Callable, cast

from loguru import logger


def shorten(inp, max_len=11):
    if not isinstance(inp, bytes):
        return inp
    suffix = b" [...]"

    return inp[:max_len] + suffix if len(inp) > max_len else inp


def timefn(method=False) -> Callable:
    """Decorator to measure the time it takes for a function to complete
    Examples:
        >>> @timefn
        >>> def time_consuming_function(...): ...
    """

    try:
        logger.level("BENCHMARK", no=45, color="<magenta>", icon="🚨")  # Loguru
    except TypeError:
        pass

    def timefn_inner(func: Callable) -> Callable:
        """Inner function for decorator closure"""

        @functools.wraps(func)
        def timed(*args, **kwargs):
            """Inner function for decorator closure"""

            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            elapsed = f"{te - ts}"

            if method:

                logger.log(
                    "BENCHMARK",
                    "{cls}.{f}(*{a}, **{kw}) took: {t} sec".format(
                        f=func.__name__, cls=args[0], a=[shorten(a) for a in args[1:]], kw=kwargs, t=elapsed
                    ),
                )
            else:
                logger.log(
                    "BENCHMARK",
                    "{f}(*{a}, **{kw}) took: {t} sec".format(
                        f=func.__name__, a=[shorten(a) for a in args], kw=kwargs, t=elapsed
                    ),
                )

            return result

        return cast(Callable, timed)

    return timefn_inner


def timegen(method=False) -> Callable:
    """Decorator to measure the time it takes for a function to complete
    Examples:
        >>> @timegen
        >>> def time_consuming_function(...): ...
    """

    try:
        logger.level("BENCHMARK", no=45, color="<magenta>", icon="🚨")  # Loguru
    except TypeError:
        pass

    def timegen_inner(func: Callable) -> Callable:
        """Inner function for decorator closure"""

        @functools.wraps(func)
        def timed(*args, **kwargs):
            """Inner function for decorator closure"""

            ts = time.time()
            generator = func(*args, **kwargs)

            for result in generator:

                te = time.time()
                elapsed = f"{te - ts}"

                if method:

                    logger.log(
                        "BENCHMARK",
                        "{cls}.{f}(*{a}, **{kw}) took: {t} sec".format(
                            f=func.__name__,
                            cls=args[0],
                            a=[shorten(a) for a in args[1:]],
                            kw=kwargs,
                            t=elapsed,
                        ),
                    )
                else:
                    logger.log(
                        "BENCHMARK",
                        "{f}(*{a}, **{kw}) took: {t} sec".format(
                            f=func.__name__, a=[shorten(a) for a in args], kw=kwargs, t=elapsed
                        ),
                    )

                yield result

        return cast(Callable, timed)

    return timegen_inner
