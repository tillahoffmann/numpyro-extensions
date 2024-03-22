from jax import numpy as jnp
from jax import random
from numpyro_extensions.distributions.transforms import ZeroSumTransform
import pytest


@pytest.mark.parametrize("transform_ndims", [1, 2, 3])
def test_zero_sum_transform(transform_ndims: int) -> None:
    x1 = random.normal(random.key(7), (3, 4, 5, 6))
    transform = ZeroSumTransform(transform_ndims)
    y = transform(x1)
    # Check desired dimensions sum to zero.
    for i in range(transform_ndims):
        assert jnp.allclose(y.sum(axis=-(i + 1)), 0, atol=1e-6)
    # Check other dimensions do not sum to zero.
    for axis in range(x1.ndim - transform_ndims):
        assert jnp.abs(y.sum(axis=axis)).min() > 1e-3
    x2 = transform.inv(y)
    assert jnp.allclose(x1, x2)
