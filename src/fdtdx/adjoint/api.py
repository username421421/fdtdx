"""The production entry points for reciprocity gradients.

Two calls, both built from a single placed scene:

    phasor_fn = reciprocity_phasor_fn(arrays, objects, config, key,
                                      objective_detector="mon",
                                      design_detector="design_region")
    loss      = lambda ie: my_fom(phasor_fn(ie))
    value, g  = jax.value_and_grad(loss)(arrays.inv_permittivities)

    param_fn  = reciprocity_param_fn(arrays, objects, config, key, ...)
    loss      = lambda p: my_fom(param_fn(p, beta=beta))
    value, g  = jax.value_and_grad(loss)(params)     # g is a ParameterContainer

The adjoint scene is derived internally, so there is no second scene to keep in
sync and no second ``place_objects`` call (see :mod:`fdtdx.adjoint.scene` for why
the second call would be a correctness bug).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import jax

from fdtdx.adjoint.reciprocity import gaussian_window
from fdtdx.adjoint.scene import derive_adjoint_objects, find_object
from fdtdx.adjoint.vjp import make_reciprocity_phasor_fn
from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.initialization import apply_params


def reciprocity_phasor_fn(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    *,
    objective_detectors: str | Sequence[str],
    design_detector: str,
    window: jax.Array | None = None,
    cond_limit: float = 1e8,
) -> Callable[[jax.Array], jax.Array | tuple[jax.Array, ...]]:
    """Differentiable monitor phasors from a single placed scene.

    Args:
        arrays: placed arrays.
        objects: placed objects, carrying the real source, the objective monitor
            and a detector covering the design region.
        config: resolved config from the same ``place_objects`` call. Its
            ``gradient_config`` should be ``None``; both solves are plain forward
            runs.
        key: PRNG key used for both solves.
        objective_detectors: monitor name, or a sequence of them. Several
            monitors cost one adjoint solve, not one each, which is what makes
            a near-to-far box affordable.
        design_detector: name of the ``PhasorDetector`` covering the design
            region. The returned gradient is nonzero only there.
        window: adjoint excitation envelope; defaults to
            :func:`~fdtdx.adjoint.reciprocity.gaussian_window` over the full run.
        cond_limit: conditioning ceiling for the adjoint amplitude solve.

    Returns:
        ``phasor_fn(inv_permittivities) -> phasors``, differentiable, costing two
        forward solves per gradient.
    """
    if window is None:
        window = gaussian_window(int(config.time_steps_total))
    adjoint_objects, adjoint_sources = derive_adjoint_objects(
        objects=objects,
        config=config,
        objective_detectors=objective_detectors,
        window=window,
        key=key,
    )
    return make_reciprocity_phasor_fn(
        forward_arrays=arrays,
        forward_objects=objects,
        adjoint_arrays=arrays,
        adjoint_objects=adjoint_objects,
        config=config,
        key=key,
        objective_detectors=objective_detectors,
        design_detector=design_detector,
        adjoint_sources=adjoint_sources,
        window=window,
        cond_limit=cond_limit,
    )


def reciprocity_param_fn(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    *,
    objective_detectors: str | Sequence[str],
    design_detector: str,
    window: jax.Array | None = None,
    cond_limit: float = 1e8,
) -> Callable[..., jax.Array | tuple[jax.Array, ...]]:
    """Differentiable monitor phasors as a function of design **parameters**.

    This is the entry point an optimizer wants: differentiate straight through to
    a :class:`ParameterContainer`, so filters, projections and any other
    ``param_transforms`` are handled by ordinary JAX and only the FDTD time loop
    is replaced.

    Args:
        arrays: placed arrays.
        objects: placed objects.
        config: resolved config.
        key: PRNG key.
        objective_detectors: monitor name, or a sequence of them.
        design_detector: detector covering the design region.
        window: adjoint excitation envelope.
        cond_limit: conditioning ceiling for the amplitude solve.

    Returns:
        ``param_fn(params, **transform_kwargs) -> phasors``. Extra keyword
        arguments are forwarded to ``apply_params``, so a continuation schedule
        such as ``beta=`` stays live per optimizer step.

    Notes:
        ``apply_params`` also returns a refreshed ``ObjectContainer``, which is
        discarded here. That is safe because the refresh runs under
        ``stop_gradient`` on the permittivities, so the objects carry no
        parameter gradient; the whole parameter dependence flows through
        ``arrays.inv_permittivities``.
    """
    phasor_fn = reciprocity_phasor_fn(
        arrays,
        objects,
        config,
        key,
        objective_detectors=objective_detectors,
        design_detector=design_detector,
        window=window,
        cond_limit=cond_limit,
    )

    def param_fn(params: Any, **transform_kwargs: Any):
        updated, _, _ = apply_params(arrays, objects, params, key, **transform_kwargs)
        return phasor_fn(updated.inv_permittivities)

    return param_fn


def design_region_slice(objects: ObjectContainer, design_detector: str):
    """Grid slice of the design detector, for indexing a returned gradient."""
    return find_object(objects, design_detector).grid_slice
