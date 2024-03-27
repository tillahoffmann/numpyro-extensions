import math
from numpyro.infer import autoguide


class AutoBatchedMixin(autoguide.AutoBatchedMixin):
    def __init__(self, *args, **kwargs):
        self.allow_event_ndims = kwargs.pop("allow_event_ndims")
        super().__init__(*args, **kwargs)

    def _setup_prototype(self, *args, **kwargs):
        # We jump over the parent to the grandparent because we want to replace the
        # parent's _setup_prototype but inherit everything else.
        super(autoguide.AutoBatchedMixin, self)._setup_prototype(*args, **kwargs)

        # Extract the batch shape.
        batch_shape = None
        for site in self.prototype_trace.values():
            if site["type"] == "sample" and not site["is_observed"]:
                shape = site["value"].shape
                expected_ndim = (
                    self.batch_ndim + site["fn"].event_dim - self.allow_event_ndims
                )
                if site["value"].ndim < expected_ndim:
                    raise ValueError(
                        f"Expected {expected_ndim} dimensions, but site "
                        f"`{site['name']}` only has shape {shape}."
                    )
                shape = shape[: self.batch_ndim]
                if batch_shape is None:
                    batch_shape = shape
                elif shape != batch_shape:
                    raise ValueError("Encountered inconsistent batch shapes.")
        self._batch_shape = batch_shape

        # Save the event shape of the non-batched part. This will always be a vector.
        batch_size = math.prod(self._batch_shape)
        if self.latent_dim % batch_size:
            raise RuntimeError(
                f"Incompatible batch shape {batch_shape} (size {batch_size}) and "
                f"latent dims {self.latent_dim}."
            )
        self._event_shape = (self.latent_dim // batch_size,)


class AutoBatchedLowRankMultivariateNormal(
    AutoBatchedMixin, autoguide.AutoBatchedLowRankMultivariateNormal
):
    """
    Guide that uses a batched :class:`~numpyro.distributions.LowRankMultivariateNormal`
    distribution to construct a guide over the entire latent space. This implementation
    is identical to
    :class:`~numpyro.infer.autoguide.AutoBatchedLowRankMultivariateNormal` except it
    allows :code:`allow_event_ndims` event dimensions to be batched.
    """

    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=autoguide.init_to_uniform,
        init_scale=0.1,
        rank=None,
        batch_ndim=1,
        allow_event_ndims=0,
    ):
        super().__init__(
            model,
            prefix=prefix,
            init_loc_fn=init_loc_fn,
            init_scale=init_scale,
            rank=rank,
            batch_ndim=batch_ndim,
            allow_event_ndims=allow_event_ndims,
        )


class AutoBatchedMultivariateNormal(
    AutoBatchedMixin, autoguide.AutoBatchedMultivariateNormal
):
    """
    Guide that uses a batched :class:`~numpyro.distributions.MultivariateNormal`
    distribution to construct a guide over the entire latent space. This implementation
    is identical to
    :class:`~numpyro.infer.autoguide.AutoBatchedMultivariateNormal` except it allows
    :code:`allow_event_ndims` event dimensions to be batched.
    """

    def __init__(
        self,
        model,
        *,
        prefix="auto",
        init_loc_fn=autoguide.init_to_uniform,
        init_scale=0.1,
        batch_ndim=1,
        allow_event_ndims=0,
    ):
        super().__init__(
            model,
            prefix=prefix,
            init_loc_fn=init_loc_fn,
            init_scale=init_scale,
            batch_ndim=batch_ndim,
            allow_event_ndims=allow_event_ndims,
        )
