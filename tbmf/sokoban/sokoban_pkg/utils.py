import random
from contextlib import contextmanager

import numpy as np


@contextmanager
def set_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)


@contextmanager
def NoLoggerWarnings():
    from gym import logger

    old_level = getattr(logger, "MIN_LEVEL", getattr(logger, "min_level", logger.INFO))
    logger.set_level(logger.ERROR)
    try:
        yield
    finally:
        logger.set_level(old_level)
