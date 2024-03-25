from jax import numpy as jnp
from numpyro.distributions.constraints import (
    Constraint,
    dependent_property,
    independent,
    real,
)
from numpyro.distributions.transforms import biject_to, Transform
from . import constraints


class IndependentDimensionTransform(Transform):
    """
    Transform applied independently to the specified number of dimensions.
    """

    def __init__(self, transform_ndims: int = 1) -> None:
        self.transform_ndims = transform_ndims

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.transform_ndims):
            x = self._forward_by_axis(x, -(i + 1))
        return x

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.transform_ndims):
            y = self._inverse_by_axis(y, -(i + 1))
        return y

    def forward_shape(self, shape: tuple) -> tuple:
        batch_shape = shape[: -self.transform_ndims]
        event_shape = tuple(
            self._forward_size_by_axis(shape, -self.transform_ndims + i)
            for i in range(self.transform_ndims)
        )
        return batch_shape + event_shape

    def inverse_shape(self, shape: tuple) -> tuple:
        batch_shape = shape[: -self.transform_ndims]
        event_shape = tuple(
            self._inverse_size_by_axis(shape, -self.transform_ndims + i)
            for i in range(self.transform_ndims)
        )
        return batch_shape + event_shape

    def _forward_by_axis(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def _inverse_by_axis(self, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def _forward_size_by_axis(self, shape: tuple, axis: int) -> int:
        raise NotImplementedError

    def _inverse_size_by_axis(self, shape: tuple, axis: int) -> int:
        raise NotImplementedError

    def tree_flatten(self):
        return (), ((), {"transform_ndims": self.transform_ndims})

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.transform_ndims == other.transform_ndims
        )


class ZeroSumTransform(IndependentDimensionTransform):
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

    @dependent_property(is_discrete=False)
    def codomain(self) -> Constraint:
        return constraints.zero_sum(self.transform_ndims)

    @dependent_property(is_discrete=False)
    def domain(self) -> Constraint:
        return independent(real, self.transform_ndims)

    def _forward_by_axis(self, x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        r"""
        Transform a tensor with size :code:`n - 1` along the specified axis to a tensor
        with size :code:`n` along the specified axis such that it sums to zero.

        Args:
            x: Tensor to transform.
            axis: Axis along which to transform.

        Returns:
            Transformed tensor.
        """
        n = x.shape[axis] + 1
        sqrt_n = jnp.sqrt(n)
        agg = x.sum(axis=axis, keepdims=True)
        return jnp.concatenate([-agg / sqrt_n, x - agg / (sqrt_n + n)], axis=axis)

    def _inverse_by_axis(self, y: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        """
        Inverse of :meth:`ZeroSumTransform._forward_by_axis`.
        """
        n = y.shape[axis]
        sqrt_n = jnp.sqrt(n)
        agg = -jnp.take(y, jnp.asarray([0]), axis=axis) * sqrt_n
        return jnp.delete(y, 0, axis=axis) + agg / (sqrt_n + n)

    def log_abs_det_jacobian(
        self, x: jnp.ndarray, y: jnp.ndarray, intermediates: None = None
    ) -> jnp.ndarray:
        return jnp.zeros_like(x, shape=x.shape[: -self.transform_ndims])

    def _forward_size_by_axis(self, shape: tuple, axis: int) -> int:
        return shape[axis] + 1

    def _inverse_size_by_axis(self, shape: tuple, axis: int) -> int:
        return shape[axis] - 1


class DecomposeSumTransform(IndependentDimensionTransform):
    r"""
    Transform a tensor with event dimensions to decouple the sum and residuals
    orthogonal to the sum.

    Args:
        transform_ndims: Number of dimensions to transform.

    Example:

        >>> from jax import numpy as jnp
        >>> from jax import random
        >>> from numpyro_extensions.distributions.transforms import \
        ...     DecomposeSumTransform
        >>>
        >>> n = 4
        >>> x = random.normal(random.key(3), (n,))
        >>> transform = DecomposeSumTransform()
        >>> y = transform(x)
        >>> jnp.allclose(x, transform.inv(y))
        Array(True, dtype=bool)

        Because reflection is
        `involutory <https://en.wikipedia.org/wiki/Involutory_matrix>`__, applying the
        transform forward or backward yields the same result.

        >>> jnp.allclose(transform(x), transform.inv(x))
        Array(True, dtype=bool)

        The sum of :math:`x` is captured by the first element of :math:`y` in the
        transformed space.

        >>> jnp.allclose(y[0], -x.sum() / jnp.sqrt(n))
        Array(True, dtype=bool)
    """

    def _forward_by_axis(self, x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        n = x.shape[axis]
        sqrt_n = jnp.sqrt(n)
        v = jnp.ones(n).at[0].add(sqrt_n).reshape((n, *(1 for _ in range(-(axis + 1)))))
        # This is like taking the dot product along a specified axis. We keep the
        # dimension to recover the v v^T in the Householder transformation.
        return x - v * (v * x).sum(axis=axis, keepdims=True) / (n + sqrt_n)

    def _inverse_by_axis(self, y: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        return self._forward_by_axis(y, axis)

    def log_abs_det_jacobian(
        self, x: jnp.ndarray, y: jnp.ndarray, intermediates: None = None
    ) -> jnp.ndarray:
        return jnp.zeros_like(x, shape=x.shape[: -self.transform_ndims])

    def _forward_size_by_axis(self, shape: tuple, axis: int) -> int:
        return shape[axis]

    def _inverse_size_by_axis(self, shape: tuple, axis: int) -> int:
        return shape[axis]


@biject_to.register(constraints._ZeroSum)
def _biject_to_zero_sum_tensor(constraint: constraints._ZeroSum):
    return ZeroSumTransform(constraint.event_dim)
