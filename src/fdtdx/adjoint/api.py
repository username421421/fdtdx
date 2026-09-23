"""The production entry points for reciprocity gradients.

Two calls, both built from a single placed scene, and neither needs any extra
objects in it -- no adjoint source and no design-region detector:

    param_fn  = reciprocity_param_fn(arrays, objects, config, key,
                                     objective_detectors="mon")
    loss      = lambda p: my_fom(param_fn(p, beta=beta))
    value, g  = jax.value_and_grad(loss)(params)     # g is a ParameterContainer

    phasor_fn = reciprocity_phasor_fn(arrays, objects, config, key,
                                      objective_detectors="mon")
    loss      = lambda ie: my_fom(phasor_fn(ie))
    value, g  = jax.value_and_grad(loss)(arrays.inv_permittivities)

The design region defaults to every :class:`~fdtdx.objects.device.device.Device`
in the scene. The adjoint scene and the detector recording the design-region
fields are both derived internally, so there is no second scene to keep in sync
and no second ``place_objects`` call (see :mod:`fdtdx.adjoint.scene` for why the
second call would be a correctness bug).
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
    design_detector: str | Sequence[str] | None = None,
    window: jax.Array | None = None,
    cond_limit: float = 1e8,
) -> Callable[[jax.Array], jax.Array | tuple[jax.Array, ...]]:
    """Differentiable monitor phasors from a single placed scene.

    Args:
        arrays: placed arrays.
        objects: placed objects, carrying the real source and the objective
            monitors.
        config: resolved config from the same ``place_objects`` call. Its
            ``gradient_config`` should be ``None``; both solves are plain forward
            runs.
        key: PRNG key used for both solves.
        objective_detectors: monitor name, or a sequence of them. Several
            monitors cost one adjoint solve, not one each, which is what makes
            a near-to-far box affordable. Either ``scaling_mode`` works.
        design_detector: where the gradient is taken. Leave it out to use every
            ``Device`` in the scene. Otherwise the name, or names, of placed
            objects whose cells form the design region: a ``Device``, a detector
            in any configuration (only its cells are used), or a static material
            block. The returned gradient is nonzero only there.
        window: adjoint excitation envelope; defaults to
            :func:`~fdtdx.adjoint.reciprocity.gaussian_window` over the full run.
        cond_limit: conditioning ceiling for the adjoint amplitude solve.

    Returns:
        ``phasor_fn(inv_permittivities) -> phasors``, differentiable, costing two
        forward solves per gradient.
    """
    if window is None:
        window = gaussian_window(int(config.time_steps_total), dtype=config.dtype)
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
    design_detector: str | Sequence[str] | None = None,
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
        design_detector: leave it out: the default, every ``Device`` in the
            scene, is exactly the set of cells ``apply_params`` writes, so the
            gradient is the full parameter gradient. See
            :func:`reciprocity_phasor_fn` for the other forms.
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

        Naming a region that does not cover every ``Device`` drops the
        gradient of the parameters outside it, silently by construction, since
        the reciprocity gradient is zero outside the design regions.

    Raises:
        NotImplementedError: if a ``Device`` has a dispersive material. See
            :func:`_reject_dispersive_devices`.
    """
    _reject_dispersive_devices(objects)
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


def _reject_dispersive_devices(objects: ObjectContainer) -> None:
    """Refuse a Device whose materials include a dispersive one.

    ``apply_params`` writes the design-dependent ADE coefficients
    (``dispersive_c1..c3``) into the Device cells as well as
    ``inv_permittivities``. The reciprocity function takes only
    ``inv_permittivities``, so the forward solve would run with the coefficients
    the scene was placed with, and the gradient through the coefficients has no
    kernel term at all. Measured on a 24^3 scene with one Lorentz material: the
    forward FoM was off by 15% and the gradient by rel 0.82 at cosine 0.67, with
    nothing raised.

    Raises:
        NotImplementedError: naming the Devices and materials.
    """
    offenders = [
        f"{device.name!r}/{mat_name!r}"
        for device in objects.devices
        for mat_name, material in device.materials.items()
        if material.is_dispersive
    ]
    if offenders:
        raise NotImplementedError(
            f"Device material(s) {', '.join(offenders)} are dispersive. The reciprocity gradient "
            "only carries the design dependence of inv_permittivities, not of the dispersion "
            "coefficients apply_params also writes, so both the forward value and the gradient "
            "would be wrong. Use GradientConfig(method='checkpointed') with run_fdtd for this "
            "scene."
        )


def design_region_slice(objects: ObjectContainer, name: str):
    """Grid slice of a named design region (a ``Device`` or any placed object).

    Use it to index a gradient returned by :func:`reciprocity_phasor_fn`, which
    is nonzero only on the design regions.
    """
    return find_object(objects, name).grid_slice
