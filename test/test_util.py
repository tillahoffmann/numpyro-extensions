from jax import numpy as jnp
from numpyro_extensions import util
import pytest


@pytest.mark.parametrize("func, args, event_shape", [("normal", ((100,),), (100,))])
def test_jax_random_state(func, args, event_shape) -> None:
    state = util.JaxRandomState(17)
    x = getattr(state, func)(*args)
    assert x.shape == event_shape
    y = getattr(state, func)(*args)
    assert (x != y).any()


def test_demean() -> None:
    state = util.JaxRandomState(17)
    x = state.normal((3, 4, 5))
    y = util.demean(x, axis=-2)
    assert jnp.allclose(y.sum(axis=-2), 0, atol=1e-6)
    assert not jnp.allclose(y.sum(axis=-1), 0, atol=1e-6)
