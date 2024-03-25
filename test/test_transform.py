from jax import numpy as jnp
from jax import random
from numpyro_extensions.distributions import constraints
from numpyro_extensions.distributions.transforms import (
    biject_to,
    DecomposeSumTransform,
    ZeroSumTransform,
)
import pytest


@pytest.mark.parametrize("transform_ndims", [1, 2, 3])
def test_zero_sum_transform(transform_ndims: int) -> None:
    x1 = random.normal(random.key(7), (3, 4, 5, 6))
    transform = ZeroSumTransform(transform_ndims)
    y = transform(x1)

    # Check desired dimensions sum to zero.
    for i in range(transform_ndims):
        assert jnp.allclose(y.sum(axis=-(i + 1)), 0, atol=1e-5)

    # Check other dimensions do not sum to zero.
    for axis in range(x1.ndim - transform_ndims):
        assert jnp.abs(y.sum(axis=axis)).min() > 1e-3

    # Check inverse transform.
    x2 = transform.inv(y)
    assert jnp.allclose(x1, x2)

    # Check constraint evaluation.
    constraint = constraints.zero_sum(transform_ndims)
    passed = constraint(y)
    assert passed.shape == x1.shape[:-transform_ndims]
    assert passed.all()
    assert not constraint(y + 0.1).any()
    assert transform == biject_to(constraint)


@pytest.mark.parametrize("transform_ndims", [1, 2, 3])
def test_decompose_sum_transform(transform_ndims: int) -> None:
    x1 = random.normal(random.key(7), (3, 4, 5, 6))
    transform = DecomposeSumTransform(transform_ndims)
    y1 = transform(x1)
    assert jnp.allclose(x1, transform(y1), atol=1e-6)

    # Compare with zero sum which should give the same results except the dimension that
    # captures the mean. For the transformed variables, we slice off the first element
    # along each of the dimensions.
    for i in range(transform_ndims):
        axis = -(i + 1)
        y1 = jnp.delete(y1, 0, axis=axis)
        x1 = x1 - jnp.mean(x1, axis=axis, keepdims=True)
    y2 = ZeroSumTransform(transform_ndims).inv(x1)
    assert jnp.allclose(y1, y2, atol=1e-6)
