from jax import numpy as jnp
import numpyro
from numpyro_extensions.infer import autoguide
from numpyro_extensions import util
import pytest


@pytest.mark.parametrize(
    "guide_cls",
    [
        autoguide.AutoBatchedLowRankMultivariateNormal,
        autoguide.AutoBatchedMultivariateNormal,
    ],
)
def test_auto_batched_low_rank_multivariate_normal_guide(guide_cls) -> None:
    def model():
        numpyro.sample(
            "x",
            numpyro.distributions.MultivariateNormal(covariance_matrix=jnp.eye(5))
            .expand([10])
            .to_event(1),
        )

    rng = util.JaxRandomState(9)
    guide = guide_cls(model)
    with pytest.raises(ValueError, match="only has shape"), rng():
        guide()

    guide = guide_cls(model, allow_event_ndims=1)
    with rng():
        assert guide()["x"].shape == (10, 5)
