import functools
from jax import numpy as jnp
from jax import random
import numpyro
from typing import Callable, Optional, Union


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


def demean(x: jnp.ndarray, axis: Optional[Union[int, tuple]] = None) -> jnp.ndarray:
    """
    De-mean an array along specified axes.

    Args:
        x: Array to de-mean.
        axis: Axis or axes to de-mean.

    Returns:
        De-meaned array.

    Example:

        >>> from jax import numpy as jnp
        >>> from numpyro_extensions.util import demean, JaxRandomState
        >>>
        >>> rng = JaxRandomState(7)
        >>> x = demean(rng.normal((4, 5)))
        >>> jnp.allclose(x.sum(axis=-1), 0, atol=1e-6)
        Array(True, dtype=bool)
        >>> jnp.allclose(x.sum(axis=-2), 0, atol=1e-6)
        Array(True, dtype=bool)
    """
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    for i in axis:
        x = x - x.mean(axis=i, keepdims=True)
    return x
