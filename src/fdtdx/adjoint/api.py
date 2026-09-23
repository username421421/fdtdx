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

``reciprocity_param_fn`` takes the objects straight from ``place_objects``.
``place_objects`` leaves every object whose projection overlaps a Device's
unapplied (a plane source spanning the design's footprint, a mode port sharing
its periodic span), and ``apply_params`` applies them on every call. Here they
are applied once, at setup, exactly as ``apply_params`` would
(:func:`apply_objects_once`), and the applied scene is ``param_fn.objects``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax

from fdtdx.adjoint.reciprocity import DEFAULT_COND_LIMIT, DEFAULT_TAIL_TOLERANCE, gaussian_window
from fdtdx.adjoint.scene import derive_adjoint_objects, design_regions, find_object
from fdtdx.adjoint.vjp import (
    ReciprocityPhasorFn,
    _slices_overlap,
    describe_uncovered,
    make_reciprocity_phasor_fn,
    uncovered_device_cells,
)
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.default_key import default_key
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.initialization import apply_params
from fdtdx.objects.object import SimulationObject


def reciprocity_phasor_fn(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    *,
    objective_detectors: str | Sequence[str],
    design_detector: str | Sequence[str] | None = None,
    window: jax.Array | None = None,
    cond_limit: float = DEFAULT_COND_LIMIT,
    tail_tolerance: float | None = DEFAULT_TAIL_TOLERANCE,
) -> ReciprocityPhasorFn:
    """Differentiable monitor phasors from a single placed scene.

    Args:
        arrays: placed arrays.
        objects: placed objects, carrying the real source and the objective
            monitors, in the state ``run_fdtd`` would run them: applied. A TFSF
            source ``place_objects`` left unapplied (it shares a Device's
            footprint) raises ``ValueError``; :func:`reciprocity_param_fn`, or
            :func:`apply_objects_once`, applies it.
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
            block. The returned gradient is nonzero only there, so a named region
            that leaves cells of a ``Device`` uncovered warns.
        window: adjoint excitation envelope; defaults to
            :func:`~fdtdx.adjoint.reciprocity.gaussian_window` over the full run.
        cond_limit: conditioning ceiling for the adjoint amplitude solve
            (:data:`~fdtdx.adjoint.reciprocity.DEFAULT_COND_LIMIT`).
        tail_tolerance: warn when the phasors look unconverged
            (:func:`~fdtdx.adjoint.vjp.dft_tail`); ``None`` disables it.

    Returns:
        ``phasor_fn(inv_permittivities) -> phasors``, differentiable, costing two
        forward solves per gradient. ``phasor_fn.diagnostics`` holds the
        amplitude-solve condition number and the convergence estimates of the
        latest call.
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
        tail_tolerance=tail_tolerance,
    )


class ReciprocityParamFn:
    """``param_fn(params, **transform_kwargs) -> phasors``, from :func:`reciprocity_param_fn`.

    Attributes:
        objects: the applied scene both solves run on (see
            :func:`apply_objects_once`). Read detector post-processing state from
            it, e.g. the reference mode a ``ModeOverlapDetector`` needs for
            ``compute_overlap``: in the container ``place_objects`` returned that
            mode is unset whenever the port shares a Device's footprint.
        phasor_fn: the underlying ``phasor_fn(inv_permittivities)``.
        diagnostics: ``phasor_fn.diagnostics``: amplitude-solve ``cond`` and the
            convergence estimates of the latest call.
    """

    def __init__(
        self,
        phasor_fn: ReciprocityPhasorFn,
        arrays: ArrayContainer,
        objects: ObjectContainer,
        key: jax.Array,
    ):
        self.phasor_fn = phasor_fn
        self.objects = objects
        self._arrays = arrays
        self._key = key
        # apply_params only needs the Devices to write inv_permittivities; handing it
        # the whole scene would re-apply every overlapping object (a mode solve per
        # port) on every call, for a result discarded here.
        self._design_objects = ObjectContainer(object_list=[objects.volume, *objects.devices], volume_idx=0)

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self.phasor_fn.diagnostics

    def __call__(self, params: Any, **transform_kwargs: Any):
        updated, _, _ = apply_params(self._arrays, self._design_objects, params, self._key, **transform_kwargs)
        return self.phasor_fn(updated.inv_permittivities)


def reciprocity_param_fn(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    *,
    objective_detectors: str | Sequence[str],
    design_detector: str | Sequence[str] | None = None,
    window: jax.Array | None = None,
    cond_limit: float = DEFAULT_COND_LIMIT,
    tail_tolerance: float | None = DEFAULT_TAIL_TOLERANCE,
) -> ReciprocityParamFn:
    """Differentiable monitor phasors as a function of design **parameters**.

    This is the entry point an optimizer wants: differentiate straight through to
    a :class:`ParameterContainer`, so filters, projections and any other
    ``param_transforms`` are handled by ordinary JAX and only the FDTD time loop
    is replaced.

    Args:
        arrays: placed arrays.
        objects: placed objects, as ``place_objects`` returned them (applied
            ones work too).
        config: resolved config.
        key: PRNG key. It seeds :func:`apply_objects_once` the way ``apply_params``
            would be seeded with it, and both solves.
        objective_detectors: monitor name, or a sequence of them.
        design_detector: leave it out: the default, every ``Device`` in the
            scene, is exactly the set of cells ``apply_params`` writes, so the
            gradient is the full parameter gradient. A named region must cover
            every Device cell (it may be larger); one that does not is refused,
            because the parameters mapped onto the uncovered cells would get a
            zero gradient instead of theirs. See :func:`reciprocity_phasor_fn`
            for the forms it takes.
        window: adjoint excitation envelope.
        cond_limit: conditioning ceiling for the amplitude solve.
        tail_tolerance: convergence warning threshold; see
            :func:`reciprocity_phasor_fn`.

    Returns:
        ``param_fn(params, **transform_kwargs) -> phasors``. Extra keyword
        arguments are forwarded to ``apply_params``, so a continuation schedule
        such as ``beta=`` stays live per optimizer step. ``param_fn.objects`` is
        the applied scene.

    Notes:
        The reference pipeline, ``apply_params`` then ``run_fdtd``, re-applies
        the objects on every call, under ``stop_gradient``, so they carry no
        parameter gradient. For an object outside every Device the result is the
        same on every call, which is why applying it once is exact; an object
        whose own cells overlap a Device is refused
        (:func:`apply_objects_once`). ``apply_params`` also zeroes the dispersion
        coefficients in the Device cells on every call, which is likewise
        parameter-independent and done once here
        (:func:`device_dispersion_as_applied`), so a dispersive block under a
        Device is simulated as ``run_fdtd`` would.

    Raises:
        NotImplementedError: if a ``Device`` has a dispersive material (see
            :func:`_reject_dispersive_devices`), or an applied object overlaps
            a Device (see :func:`apply_objects_once`).
        ValueError: if ``design_detector`` leaves cells of a Device uncovered.
    """
    _reject_dispersive_devices(objects)
    if design_detector is not None:
        regions = design_regions(objects, design_detector)
        uncovered = uncovered_device_cells(objects, [r.grid_slice_tuple for r in regions])
        if uncovered:
            raise ValueError(
                describe_uncovered(uncovered, design_detector)
                + " The reciprocity gradient is zero outside the design region, so every parameter "
                "apply_params maps onto those cells would get a zero gradient instead of its own "
                "(measured before this check: rel 8.0e-01 for a region one cell short of the Device, "
                "4.6e-01 for one shifted by a cell, 7.0e-01 with a second Device left out). Leave "
                "design_detector=None, which is every Device, or name regions covering them all."
            )
    arrays = device_dispersion_as_applied(arrays, objects)
    applied = apply_objects_once(arrays, objects, key)
    phasor_fn = reciprocity_phasor_fn(
        arrays,
        applied,
        config,
        key,
        objective_detectors=objective_detectors,
        design_detector=design_detector,
        window=window,
        cond_limit=cond_limit,
        tail_tolerance=tail_tolerance,
    )
    return ReciprocityParamFn(phasor_fn, arrays, applied, key)


def _has_own_apply(obj: SimulationObject) -> bool:
    return type(obj).apply is not SimulationObject.apply


def apply_objects_once(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    key: jax.Array | None,
) -> ObjectContainer:
    """Apply the objects ``apply_params`` refreshes, once, as it would.

    ``place_objects`` applies an object only when no Device's projection
    overlaps its own on any axis (``SimulationObject.check_overlap``), so a plane
    source above a design, or a mode port on the periodic span a design also
    fills, comes back unapplied and crashes the solve. ``apply_params`` applies
    exactly those objects on every call. This runs that same loop -- same
    objects, same order, the same ``key`` split once per applied object, the
    same ``stop_gradient`` material arrays -- without the design parameters.

    That is exact for every object whose ``apply`` reads only cells outside the
    Devices, which FDTDX's sources and mode detectors do (each reads its own
    slice). An object with its own ``apply`` whose cells overlap a Device would
    see the design, so its applied state would change with the parameters; it
    is refused rather than frozen at one design.

    Args:
        arrays: the placed arrays. ``initial_inv_permittivities`` is used when
            set, as ``apply_params`` does.
        objects: placed objects, applied or not.
        key: the key ``apply_params`` would be called with.

    Returns:
        A new container with those objects applied; ``objects`` is unchanged.

    Raises:
        NotImplementedError: naming an applied object that overlaps a Device.
    """
    key = default_key(key)
    devices = objects.devices
    inside = [
        f"{getattr(obj, 'name', None)!r} ({type(obj).__name__}) in {dev.name!r}"
        for obj in objects.object_list
        if _has_own_apply(obj)
        for dev in devices
        if _slices_overlap(obj.grid_slice_tuple, dev.grid_slice_tuple)
    ]
    if inside:
        raise NotImplementedError(
            f"Object(s) {', '.join(inside)} overlap a Device and compute their state from the "
            "material there (a mode solve or an impedance), so that state changes with the design "
            "parameters. reciprocity_param_fn applies such objects once, at setup, which is exact "
            "only outside the Devices. Move them out of the design region, or use run_fdtd with "
            "GradientConfig(method='checkpointed')."
        )

    inv_eps = arrays.initial_inv_permittivities
    if inv_eps is None:
        inv_eps = arrays.inv_permittivities

    def frozen(value):
        return None if value is None else jax.lax.stop_gradient(value)

    new_list = []
    for obj in objects.object_list:
        if any(dev.check_overlap(obj) for dev in devices):
            key, subkey = jax.random.split(key)
            obj = obj.apply(
                key=subkey,
                inv_permittivities=jax.lax.stop_gradient(inv_eps),
                inv_permeabilities=jax.lax.stop_gradient(arrays.inv_permeabilities),
                dispersive_c1=frozen(arrays.dispersive_c1),
                dispersive_c2=frozen(arrays.dispersive_c2),
                dispersive_c3=frozen(arrays.dispersive_c3),
                electric_conductivity=frozen(arrays.electric_conductivity),
            )
        new_list.append(obj)
    return ObjectContainer(object_list=new_list, volume_idx=objects.volume_idx)


def device_dispersion_as_applied(arrays: ArrayContainer, objects: ObjectContainer) -> ArrayContainer:
    """Write the Devices' dispersion coefficients the way ``apply_params`` does, once.

    ``apply_params`` overwrites ``dispersive_c1..c3`` in every Device's cells
    whenever the scene has any dispersive material, with the coefficients of the
    Device's own materials. Those are refused when dispersive
    (:func:`_reject_dispersive_devices`), so what it writes is zero, whatever the
    parameters. ``place_objects`` instead leaves the coefficients of a dispersive
    object *under* a Device in those cells, and the reciprocity solves, which take
    only ``inv_permittivities`` per call, would keep simulating them: measured on a
    Lorentz block under an air/silicon Device, forward FoM off by 60% and gradient
    rel 8.3e-01 at cosine 0.66 against ``run_fdtd``, with nothing raised.

    Args:
        arrays: the placed arrays.
        objects: the placed objects.

    Returns:
        ``arrays`` with the Device cells' ADE coefficients zeroed (unchanged when
        the scene has no dispersive material).
    """
    if arrays.dispersive_c1 is None:
        return arrays
    for name in ("dispersive_c1", "dispersive_c2", "dispersive_c3"):
        value = getattr(arrays, name)
        if value is None:
            continue
        for device in objects.devices:
            value = value.at[:, :, *device.grid_slice].set(0.0)
        arrays = arrays.aset(name, value)
    return arrays


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
