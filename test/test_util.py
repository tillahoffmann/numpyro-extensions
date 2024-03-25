from numpyro_extensions import util
import pytest


@pytest.mark.parametrize("func, args, event_shape", [("normal", ((100,),), (100,))])
def test_jax_random_state(func, args, event_shape) -> None:
    state = util.JaxRandomState(17)
    x = getattr(state, func)(*args)
    assert x.shape == event_shape
    y = getattr(state, func)(*args)
    assert (x != y).any()
