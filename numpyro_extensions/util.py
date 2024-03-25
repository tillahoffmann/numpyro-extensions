import functools
from jax import random
from typing import Optional


def _wrap_random(func):
    @functools.wraps(func)
    def _inner(self: "JaxRandomState", *args, **kwargs):
        return func(self.get_key(), *args, **kwargs)

    return _inner


class JaxRandomState:
    """
    Utility class for sampling random variables using the JAX interface with automatic
    random state handling.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._key = random.PRNGKey(seed)

    def get_key(self):
        """
        Get a random key and update the state.
        """
        self._key, key = random.split(self._key)
        return key

    multivariate_normal = _wrap_random(random.multivariate_normal)
    normal = _wrap_random(random.normal)
