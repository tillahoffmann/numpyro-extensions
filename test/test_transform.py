import jax
from jax import numpy as jnp
from jax import random
import math
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


# Copied from https://github.com/pyro-ppl/numpyro/blob/master/test/test_transforms.py.
@pytest.mark.parametrize(
    "transform, shape",
    [
        (ZeroSumTransform(), (5,)),
        (ZeroSumTransform(transform_ndims=2), (5, 3)),
        (DecomposeSumTransform(), (7,)),
        (
            DecomposeSumTransform(transform_ndims=2),
            (
                9,
                7,
            ),
        ),
    ],
)
def test_bijective_transforms(transform, shape):
    if isinstance(transform, type):
        pytest.skip()
    # Get a sample from the support of the distribution.
    batch_shape = (13,)
    unconstrained = random.normal(random.key(17), batch_shape + shape)
    x1 = biject_to(transform.domain)(unconstrained)

    # Transform forward and backward, checking shapes, values, and Jacobian shape.
    y = transform(x1)
    assert y.shape == transform.forward_shape(x1.shape)

    x2 = transform.inv(y)
    assert x2.shape == transform.inverse_shape(y.shape)
    assert jnp.allclose(x1, x2, atol=1e-6)

    log_abs_det_jacobian = transform.log_abs_det_jacobian(x1, y)
    assert log_abs_det_jacobian.shape == batch_shape

    # Also check the Jacobian numerically for transforms with the same input and output
    # size, unless they are explicitly excluded. E.g., the upper triangular of the
    # CholeskyTransform is zero, giving rise to a singular Jacobian.
    size_x = int(x1.size / math.prod(batch_shape))
    size_y = int(y.size / math.prod(batch_shape))
    if size_x == size_y:
        jac = (
            jax.vmap(jax.jacfwd(transform))(x1)
            .reshape((-1,) + x1.shape[len(batch_shape) :])
            .reshape(batch_shape + (size_y, size_x))
        )
        slogdet = jnp.linalg.slogdet(jac)
        assert jnp.allclose(log_abs_det_jacobian, slogdet.logabsdet, atol=1e-5)
