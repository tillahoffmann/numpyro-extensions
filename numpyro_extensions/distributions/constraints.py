from jax import numpy as jnp
from numpyro.distributions.constraints import Constraint


class _ZeroSum(Constraint):
    def __init__(self, event_dim: int):
        self.event_dim = event_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        axes = tuple(-(i + 1) for i in range(self.event_dim))
        passed = True
        for axis in axes:
            # Numerical tolerance based on _CorrCholesky implementation.
            tol: float = jnp.finfo(x.dtype).eps * x.shape[axis] * 10
            agg = x.sum(axis=axis, keepdims=True)
            passed = passed & (jnp.abs(agg) < tol).all(axis=axes)
        return passed

    def feasible_like(self, prototype: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(prototype)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ZeroSum) and self.event_dim == other.event_dim

    def tree_flatten(self):
        return (), ((), {"event_dim": self.event_dim})


zero_sum = _ZeroSum
