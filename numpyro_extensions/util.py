import functools
from jax import numpy as jnp
from jax import random
import numpyro
from typing import Callable, Optional


def _wrap_random(func):
    @functools.wraps(func)
    def _inner(self: "JaxRandomState", *args, **kwargs):
        return func(self.get_key(), *args, **kwargs)

    return _inner


class JaxRandomState:
    """
    Utility class for sampling random variables using the JAX interface with automatic
    random state handling. It can also be used like a :class:`~numpyro.handlers.seed`
    handler without having to specify a seed.

    Args:
        seed: Initial random number generator seed.

    .. warning::

        This implementation is stateful and may lead to unexpected behavior if
        jit-compiled.

    Example:

        >>> import numpyro
        >>> from numpyro.distributions import Normal
        >>> from numpyro_extensions.util import JaxRandomState
        >>>
        >>> rng = JaxRandomState(7)
        >>> rng.normal()
        Array(-1.4622004, dtype=float32)
        >>> rng.normal()
        Array(2.0224454, dtype=float32)
        >>> with rng():
        ...     numpyro.sample("x", Normal())
        Array(-2.9687815, dtype=float32)
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._key = random.PRNGKey(seed)

    def get_key(self) -> jnp.ndarray:
        """
        Get a random key and update the state.
        """
        self._key, key = random.split(self._key)
        return key

    def __call__(
        self, model: Optional[Callable] = None, hide_types: Optional[list] = None
    ) -> Callable:
        return numpyro.handlers.seed(model, self.get_key(), hide_types)

    multivariate_normal = _wrap_random(random.multivariate_normal)
    normal = _wrap_random(random.normal)
