Transforms
==========

.. autoclass:: numpyro_extensions.distributions.transforms.ZeroSumTransform

.. note::

    The transform can be achieved using a `Householder transformation <https://en.wikipedia.org/wiki/Householder_transformation>`__ such that the vector of ones is reflected onto one of the axes, typically :math:`\mathbf{e}=\left(1,0,\ldots\right)`. Because the Householder transformation is orthogonal, this implies that the vector of ones (which corresponds to the total) is orthogonal to all the other axes after the transformation. The Householder transform is

    .. math::
        \mathbf{H} = \mathbf{I} - 2 \mathbf{v} \mathbf{v}^\intercal,

    where :math:`v` is the vector normal to the plane of reflection. The desired reflection `can be constructed <https://de.wikipedia.org/wiki/Householdertransformation#Konstruktion_einer_spezifischen_Spiegelung>`__ by letting

    .. math::
        \mathbf{v} = \frac{\mathbf{a} + \sqrt n \mathbf{e}}{\left\vert\mathbf{a} + \sqrt n \mathbf{e}\right\vert},

    where :math:`\mathbf{a}=\left(1,\ldots,1\right)` is the vector of ones. The norm in the denominator is :math:`2\left(n+\sqrt n\right)`.

    >>> from jax import numpy as jnp
    >>> from jax import random
    >>>
    >>> n = 5
    >>> y = random.normal(random.key(8), (n,))
    >>> ones = jnp.ones(n)
    >>> v = ones.at[0].add(jnp.sqrt(n))
    >>> jnp.allclose(v @ v, 2 * (n + jnp.sqrt(n)))
    Array(True, dtype=bool)
    >>> v = v / jnp.linalg.norm(v)
    >>> H = jnp.eye(n) - 2 * v[:, None] * v
    >>> x = H @ y
    >>> jnp.allclose(x[0], - y.sum() / jnp.sqrt(n))
    Array(True, dtype=bool)

    Setting the first element of :math:`\mathbf{x}` to zero removes the mean.

    >>> x = x.at[0].set(0)
    >>> x_prime = H @ x
    >>> jnp.allclose(x_prime.sum(), 0, atol=1e-6)
    Array(True, dtype=bool)
