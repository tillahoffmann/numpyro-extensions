Transforms
==========

.. _householder-transformations:

Householder Transformations for Centering
-----------------------------------------

Additive predictors in hierarchical models can lead to degeneracies because only the sum of predictors is identified by the data. This problem can be alleviated by using a :class:`~numpyro_extensions.distributions.transforms.ZeroSumTransform` which ensures parameters sum to zero, eliminating the degeneracy. However, zero-sum transforms change the underlying model, and inferences need to be interpreted carefully. Alternatively, a :class:`~numpyro_extensions.distributions.transforms.DecomposeSumTransform` can be used to reparameterize the vector of random variables such that only a single component affects the overall sum (equivalently mean).

Both transformations are implemented using `Householder transformations <https://en.wikipedia.org/wiki/Householder_transformation>`__. The transform is constructed such that the vector of ones :math:`\mathbf{a}=\left(1,\ldots,1\right)` is reflected onto one of the axes, typically :math:`\mathbf{e}=\left(1,0,\ldots\right)`. Because the Householder transformation is orthogonal, this implies that the vector of ones (which corresponds to the total) is orthogonal to all the other axes after the transformation. The Householder transform is

.. math::
    \mathbf{H} = \mathbf{I} - 2 \mathbf{v} \mathbf{v}^\intercal,

where :math:`v` is the vector normal to the plane of reflection. The desired reflection `can be constructed <https://de.wikipedia.org/wiki/Householdertransformation#Konstruktion_einer_spezifischen_Spiegelung>`__ by letting

.. math::
    \mathbf{v} = \frac{\mathbf{a} + \sqrt n \mathbf{e}}{\left\vert\mathbf{a} + \sqrt n \mathbf{e}\right\vert}.

The norm in the denominator is :math:`2\left(n+\sqrt n\right)`.

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

.. autoclass:: numpyro_extensions.distributions.transforms.ZeroSumTransform
.. autoclass:: numpyro_extensions.distributions.transforms.DecomposeSumTransform
