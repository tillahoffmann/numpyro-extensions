---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Mean Decomposition

Additive predictors in hierarchical models can lead to degeneracies because only the sum of predictors is identified by the data. This problem can be alleviated by using a {class}`~numpyro_extensions.distributions.transforms.ZeroSumTransform` which ensures parameters sum to zero, eliminating the degeneracy. However, zero-sum transforms change the underlying model, and inferences need to be interpreted carefully. Alternatively, a {class}`~numpyro_extensions.distributions.transforms.DecomposeSumTransform` can be used to reparameterize the vector of random variables such that only a single component affects the overall sum (equivalently mean).

```{seealso}
{ref}`householder-transformations` discusses the background for the implementation of both transforms.
```

Here, we consider the effect of different parameterization strategies on the posterior using simple examples of (multivariate) normal distributions. We have a random vector $\mathbf{y}\sim\mathsf{Normal}\left(0,\mathbf{C}\right)$ of length $n$ we seek to infer, where $\mathbf{C}$ is a covariance matrix. We first consider the setting of independent random variables.

```{code-cell} ipython3
import jax
from jax import numpy as jnp
from localscope import localscope
from matplotlib import pyplot as plt
import numpyro
from numpyro import distributions
from numpyro import handlers
from tqdm.notebook import tqdm
from numpyro_extensions.distributions.transforms import (
    DecomposeSumTransform,
    ZeroSumTransform,
)
from numpyro_extensions.util import JaxRandomState


@localscope.mfc
def fit_model(
    model,
    *args,
    guide_cls=numpyro.infer.autoguide.AutoMultivariateNormal,
    lr=5e-4,
    seed=87,
    period=10,
    num_steps=20_000,
    num_samples=100,
):
    """
    Helper function to fit a model with sensible defaults.
    """
    guide = guide_cls(model)
    optim = numpyro.optim.Adam(step_size=lr)
    svi = numpyro.infer.SVI(model, guide, optim, numpyro.infer.Trace_ELBO())
    update = jax.jit(svi.update, *args)
    state = svi.init(jax.random.key(seed))
    traces = {}
    losses = []
    for step in tqdm(range(num_steps)):
        state, loss = update(state, *args)
        if step % period == 0:
            losses.append(loss)
            params = svi.get_params(state)
            median = guide.median(params)
            for key, value in median.items():
                traces.setdefault(key, []).append(value)

    params = svi.get_params(state)
    median = guide.median(params)
    samples = guide.sample_posterior(
        jax.random.key(seed),
        params,
        sample_shape=[num_samples],
    )
    return {
        "losses": jnp.asarray(losses),
        "traces": {key: jnp.asarray(value) for key, value in traces.items()},
        "median": median,
        "samples": samples,
        "params": params,
    }


def plot_fit(fit):
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(fit["losses"])
    axes[1].plot(fit["traces"]["y"])


def model():
    n = 4
    dist = distributions.Normal().expand([n]).to_event(1)
    numpyro.sample("y", dist)


fit = fit_model(model)
plot_fit(fit)
```

If there is an additive degeneracy in the model, convergence can be very slow because the likelihood induces a sharp valley in the posterior density which is only attenuated by the comparatively weak prior. This effect can be readily seen in the trace of median values.

```{code-cell} ipython3
def model(sigma=0.2):
    n = 4
    dist = distributions.Normal().expand([n]).to_event(1)
    y = numpyro.sample("y", dist)
    numpyro.sample("a", distributions.Normal(y.sum(), sigma), obs=0.)


fit = fit_model(model)
plot_fit(fit)
```

## Sum-Residual Decomposition

We can reparameterize the model by decomposing the random variable $y$ into its mean and residuals that are orthogonal to the mean, i.e., changing the residuals does not affect the mean. This reparameterization vastly accelerates fitting because it rotates the posterior such that the parameter capturing the overall mean is approximately aligned with the direction of the valley. In practice, the transformed variables do not directly correspond to the mean and residuals; see {ref}`householder-transformations` for details.

```{code-cell} ipython3
reparam = handlers.reparam(
    model,
    config={
        "y": numpyro.infer.reparam.ExplicitReparam(DecomposeSumTransform()),
    },
)
fit = fit_model(reparam)
plot_fit(fit)
```

## Zero-Sum Parameterization

A more aggressive approach is to demand that parameters sum to zero exactly which changes the model. Consider the mean contraction $\mathbf{m}=\frac{1}{n}\mathbf{1}$ such that $\mathbf{m}^\intercal \mathbf{y} = \bar y$. Then $\mathbf{r}=\mathbf{D}\mathbf{y}$ sums to zero, where $\mathbf{D} = \mathbf{I} - \mathbf{1}\mathbf{m}^\intercal$ is the de-meaning operator. To satisfy the sum-to-zero constraint, we consider a vector $\mathbf{z}$ of length $n - 1$ in an unconstrained space together with a linear transformation $\mathbf{A}$ such that $\mathbf{r}=\mathbf{A}\mathbf{z}$.

Considering multivariate normal $\mathbf{y}$, the covariance of $\mathbf{r}$ is
$$\begin{aligned}
\mathbb{E}\left[\mathbf{r}\mathbf{r}^\intercal\right]&=\mathbb{E}\left[\mathbf{D}\mathbf{y}\mathbf{y}^\intercal\mathbf{D}\right]\\
&=\mathbf{D}\mathbf{C}\mathbf{D}^\intercal.
\end{aligned}$$
While $\mathbf{A}$ is not strictly invertible because it has shape $\left(n, n - 1\right)$, dropping one of the rows allows for $\mathbf{z}$ to be obtained from $\mathbf{r}$ because the covariance matrix of $\mathbf{R}$ is singular. With a slight abuse of notation, we denote this transformation by $\mathbf{A}^{-1}$ such that $\mathbf{z}=\mathbf{A}^{-1}\mathbf{r}$. We assume that $\mathbf{A}$ is [orthonormal](https://en.wikipedia.org/wiki/Orthogonal_matrix) such that $\mathbf{A}^{-1}=\mathbf{A}^\intercal$. The desired covariance matrix in the transformed space is thus $\mathbf{z}$
$$
\mathbb{E}\left[\mathbf{r}\mathbf{r}^\intercal\right]=\mathbf{A}^{-1}\mathbf{D}\mathbf{C}\mathbf{D}^\intercal{\mathbf{A}^{-1}}^\intercal.
$$

Here, we verify these relationships numerically.

```{code-cell} ipython3
# Create transform and verify basic properties about its Jacobian.
n = 4
transform = ZeroSumTransform()
rng = JaxRandomState(8)
z = rng.normal([n - 1])
A = jax.jacfwd(transform)(z)
assert A.shape == (n, n - 1), "A has the wrong shape"
assert jnp.allclose(A.sum(axis=0), 0, atol=1e-6), "rows of A don't sum to zero"
y = transform(z)
assert jnp.allclose(y, A @ z)
assert jnp.allclose(z, transform.inv(y)), "input not reconstructed by inverse transform"
A_inv = jnp.linalg.pinv(A)
assert jnp.allclose(A_inv, A.T), "A is not orthonormal"
assert jnp.allclose(z, A_inv @ y)

# Sample from the full space, demean, transform, and check the covariance.
demean = jnp.eye(n) - 1 / n
C_y = jnp.cov(rng.normal((n, 20)))
ys = rng.multivariate_normal(jnp.zeros(n), C_y, [1_000_000])
rs = ys @ demean.T
zs = transform.inv(rs)
empirical_C_z = jnp.cov(zs.T)
theoretical_C_z = A_inv @ (demean @ C_y @ demean.T) @ A_inv.T
empirical_C_z.round(2), theoretical_C_z.round(2)
```

If $\mathbf{C} = \sigma^2 \mathbf{I} + \rho \mathbf{1}\mathbf{1}^\intercal$ and $\mathbf{A}$ is an orthogonal transformation, the covariance in the transformed space is diagonal. First, de-meaning removes the $\rho$ term because it does not vary across rows or columns. Second, de-meaning is [idempotent](https://en.wikipedia.org/wiki/Idempotence) such that $\mathbf{D}\mathbf{D}=\mathbf{D}$, and the covariance is
$$
\sigma^2\mathbf{A}^\intercal\mathbf{D}\mathbf{A},
$$
where we have used the orthogonality of $\mathbf{A}$ to rewrite $\mathbf{A}^{-1}=\mathbf{A}^\intercal$. By construction, $\mathbf{A}$ is already de-meaned, and $\mathbf{D}\mathbf{A}=\mathbf{A}$. Using the orthogonality property again, the covariance of $\mathbf{z}$ is $\sigma^2\mathbf{I}$.

```{code-cell} ipython3
C_y = 4 * jnp.eye(n) + 1.3
theoretical_C_z = A_inv @ (demean @ C_y @ demean.T) @ A_inv.T
assert jnp.allclose(theoretical_C_z, 4 * jnp.eye(n - 1), atol=1e-6)
```

Let us apply these insights to the above model. Convergence improves futher, but only marginally because the {class}`~numpyro_extensions.distributions.transforms.DecomposeSumTransform` already simplifies the posterior geometry significantly.

```{code-cell} ipython3
def model(sigma=0.2):
    n = 4
    dist = distributions.TransformedDistribution(
        distributions.Normal().expand([n - 1]).to_event(1),
        ZeroSumTransform(),
    )
    y = numpyro.sample("y", dist)
    numpyro.sample("a", distributions.Normal(y.sum(), sigma), obs=0.)


fit = fit_model(model)
plot_fit(fit)
```
