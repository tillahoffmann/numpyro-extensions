import jax
from jax import numpy as jnp
import numpy as np
import numpyro
from numpyro import distributions
from numpyro_extensions.distributions.transforms import DecomposeSumTransform
from numpyro_extensions.infer.reparam import sample_zero_sum
from numpyro_extensions.util import JaxRandomState


def assert_linear_close(x: jnp.ndarray, y: jnp.ndarray, atol: float = 0.01) -> None:
    poly = np.polynomial.Polynomial.fit(x.ravel(), y.ravel(), 1).convert()
    assert jnp.isclose(poly.coef[1], 1, atol=atol), poly


def test_zero_sum_statistics() -> None:
    cov = jnp.asarray(
        [
            [1.4, 0.3, -0.4],
            [0.3, 3.5, 0.2],
            [-0.4, 0.2, 1.3],
        ]
    )
    n, _ = cov.shape

    def model():
        dist = distributions.MultivariateNormal(jnp.zeros(n), cov)
        sample_zero_sum("y", dist)

    # Obtain an exact variational approximation using a multivariate normal guide.
    guide = numpyro.infer.autoguide.AutoMultivariateNormal(
        model,
        init_loc_fn=numpyro.infer.init_to_value(values={"y_raw_base": jnp.zeros(n)}),
    )
    optim = numpyro.optim.Adam(step_size=1e-5)
    svi = numpyro.infer.SVI(model, guide, optim, numpyro.infer.Trace_ELBO())
    rng = JaxRandomState(9)
    result = svi.run(rng.get_key(), 1_000_000, progress_bar=False)

    # Sample the posterior and verify the expected covariance.
    demean = jnp.eye(n) - 1 / n
    expected_cov = demean @ cov @ demean
    samples = guide.sample_posterior(
        rng.get_key(), result.params, sample_shape=(10_000_000,)
    )
    empirical_cov = jnp.cov(samples["y"].T)
    assert_linear_close(expected_cov, empirical_cov)

    # Evaluate the covariance in the unconstrained space and compare with expectation.
    # We conly compare the components that do not correspond to the grand mean which can
    # have arbitrary distribution.
    scale_tril = result.params["auto_scale_tril"]
    empirical_unconstrained_cov = scale_tril @ scale_tril.T
    A = jax.jacfwd(DecomposeSumTransform())(jnp.empty(n))
    expected_unconstrained_cov = (A @ cov @ A)[1:, 1:]
    assert_linear_close(expected_unconstrained_cov, empirical_unconstrained_cov[1:, 1:])
