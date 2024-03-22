from jax import numpy as jnp
from numpyro.distributions.transforms import biject_to, Transform
from . import constraints


def _householder_transform_forward(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    r"""
    Transform a tensor with size :code:`n - 1` along the specified axis to a tensor with
    size :code:`n` along the specified axis such that it sums to zero.

    Args:
        x: Tensor to transform.
        axis: Axis along which to transform.

    Returns:
        Transformed tensor.
    """
    n = x.shape[axis] + 1
    sqrt_n = jnp.sqrt(n)
    agg = x.sum(axis=axis, keepdims=True)
    return jnp.concatenate([x - agg / (sqrt_n + n), -agg / sqrt_n], axis=axis)


def _householder_transform_inverse(y: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Inverse of :func:`_householder_transform_forward`.
    """
    n = y.shape[axis]
    sqrt_n = jnp.sqrt(n)
    agg = -jnp.take(y, jnp.asarray([-1]), axis=axis) * sqrt_n
    return jnp.delete(y, -1, axis=axis) + agg / (sqrt_n + n)


class ZeroSumTransform(Transform):
    """
    Transform from unconstrained space to a tensor which sums to zero along the trailing
    :code:`transform_ndims` axes.

    Args:
        transform_ndims: Number of trailing dimensions to transform.

    Example:

        >>> from jax import numpy as jnp
        >>> from jax import random
        >>> from numpyro_extensions.distributions.transforms import ZeroSumTransform
        >>>
        >>> n = 5
        >>> x = random.normal(random.key(8), (n - 1,))
        >>> transform = ZeroSumTransform()
        >>> y = transform(x)
        >>> y.shape
        (5,)
        >>> jnp.allclose(y.sum(), 0, atol=1e-6)
        Array(True, dtype=bool)
        >>> jnp.allclose(x, transform.inv(y))
        Array(True, dtype=bool)
    """

    def __init__(self, transform_ndims: int = 1) -> None:
        self.transform_ndims = transform_ndims

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.transform_ndims):
            x = _householder_transform_forward(x, -(i + 1))
        return x

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.transform_ndims):
            y = _householder_transform_inverse(y, -(i + 1))
        return y

    def log_abs_det_jacobian(
        self, x: jnp.ndarray, y: jnp.ndarray, intermediates: None = None
    ) -> jnp.ndarray:
        return super().log_abs_det_jacobian(x, y, intermediates)

    def tree_flatten(self):
        return (), ((), {"transform_ndims": self.transform_ndims})

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ZeroSumTransform)
            and self.transform_ndims == other.transform_ndims
        )


@biject_to.register(constraints._ZeroSum)
def _biject_to_zero_sum_tensor(constraint: constraints._ZeroSum):
    return ZeroSumTransform(constraint.event_dim)
