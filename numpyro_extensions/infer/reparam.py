from jax import numpy as jnp
import numpyro
from numpyro.infer.reparam import ExplicitReparam
from typing import Callable, Optional
from ..distributions.transforms import DecomposeSumTransform


class DecomposeSumReparam(ExplicitReparam):
    """
    Reparameterize a latent variable to decouple the sum and residuals orthogonal to the
    sum. See :class:`~numpyro_extensions.distributions.transforms.DecomposeSumTransform`
    for details.

    .. seealso::

        :func:`sample_zero_sum` uses :class:`DecomposeSumReparam` to sample general
        random variables with sum-to-zero constraints.

    Args:
        transform_ndims: Number of dimensions to transform.
    """

    def __init__(self, transform_ndims: int = 1) -> None:
        transform = DecomposeSumTransform(transform_ndims)
        super().__init__(transform)


def sample_zero_sum(
    name: str,
    fn: Callable,
    obs: Optional[jnp.ndarray] = None,
    rng_key: Optional[jnp.ndarray] = None,
    sample_shape: tuple = (),
    infer: Optional[dict] = None,
    obs_mask: Optional[jnp.ndarray] = None,
    *,
    transform_ndims: int = 1,
) -> jnp.ndarray:
    """
    Sample a random variable subject to a sum-to-zero constraint. See
    :func:`numpyro.primitives.sample` for details on arguments.

    Args:
        name: Name of the sample site.
        fn: Stochastic function that returns a sample.
        obs: Observed value.
        rng_key: Optional random key for :code:`fn`.
        sample_shape: Shape of samples to draw.
        infer: Optional dictionary containing additional information for infernece
            algorithms.
        obs_mask: Optional boolean mask broadcastable with :code:`fn.batch_shape`
            indicating which values to condition on.
        transform_ndims: Number of dimensions to sum to zero.

    Examples:

        >>> from jax import numpy as jnp
        >>> import numpyro
        >>> from numpyro.distributions import Gamma
        >>> from numpyro_extensions.infer.reparam import sample_zero_sum
        >>>
        >>> with numpyro.handlers.seed(rng_seed=13):
        ...     x = sample_zero_sum("x", Gamma(4 * jnp.ones(3)).to_event(1))
        >>> x
        Array([-1.0068426, -2.0927474,  3.0995898], dtype=float32)
        >>> jnp.allclose(x.sum(), 0, atol=1e-6)
        Array(True, dtype=bool)

    Notes:

        The sum-to-zero sample is obtained by reparameterizing the random variable using
        a :class:`.DecomposeSumReparam` such that the mean of the random variable is
        orthogonal to residuals. The resulting sample is mean-subtracted which fully
        decouples the mean from the rest of the model. The parameter capturing the mean
        remains part of the model but is only affected by the prior.
    """
    raw_name = f"{name}_raw"
    config = {raw_name: DecomposeSumReparam(transform_ndims)}
    with numpyro.handlers.reparam(config=config):
        x = numpyro.sample(raw_name, fn, obs, rng_key, sample_shape, infer, obs_mask)
    for i in range(transform_ndims):
        x = x - x.mean(axis=-(i + 1))
    return numpyro.deterministic(name, x)
