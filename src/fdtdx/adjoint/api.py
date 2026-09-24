"""The two entry points, both built from one placed scene with nothing added to it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax

from fdtdx.adjoint import validation
from fdtdx.adjoint.design import apply_objects_once, design_regions, device_dispersion_as_applied
from fdtdx.adjoint.kernel import DEFAULT_COND_LIMIT, DEFAULT_TAIL_TOLERANCE, gaussian_window
from fdtdx.adjoint.vjp import ReciprocityPhasorFn, make_phasor_fn
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
    cond_limit: float = DEFAULT_COND_LIMIT,
    tail_tolerance: float | None = DEFAULT_TAIL_TOLERANCE,
) -> ReciprocityPhasorFn:
    """Differentiable objective phasors as a function of ``inv_permittivities`` (and ``electric_conductivity``).

    Everything else is the placed scene's, except what ``apply_params`` rewrites whatever the
    design: the Device cells' dispersion coefficients are zeroed
    (:func:`~fdtdx.adjoint.design.device_dispersion_as_applied`).

    Args:
        arrays: placed arrays.
        objects: placed objects in the state ``run_fdtd`` runs them, i.e. applied (a TFSF
            source ``place_objects`` left unapplied raises; see :func:`reciprocity_param_fn`).
        config: the resolved config; both solves are plain forward runs.
        key: PRNG key for both solves.
        objective_detectors: the monitor, or monitors, the figure of merit reads: phasor
            detectors at their stock settings (either scaling mode, exact interpolation,
            ``dft_subsample``), a box-mode field projection, a flux box. Several cost one
            adjoint solve together. They must share frequencies.
        design_detector: where the gradient is taken; ``None`` is every Device. Otherwise the
            name(s) of placed objects whose cells form the region (a Device, a detector of any
            configuration, a static block). The gradient is zero outside it.
        window: envelope of the adjoint excitation; :func:`~fdtdx.adjoint.kernel.gaussian_window`
            over the run by default.
        cond_limit: ceiling on the amplitude solve's condition number.
        tail_tolerance: warn (:class:`~fdtdx.adjoint.kernel.ConvergenceWarning`) when a DFT tail
            estimate exceeds this; ``None`` disables the warning.

    Returns:
        ``phasor_fn(inv_permittivities, electric_conductivity=None)``: the monitor's phasor
        array, or a tuple over several monitors; a monitor storing several arrays (a box) gives
        its whole state dict, to feed its own readout (``project``, ``compute_net_flux``).
        ``electric_conductivity`` (``None``: the scene's, held constant; refused when
        ``apply_params`` writes another one into a Device) is differentiated too.
        ``phasor_fn.diagnostics`` holds the solve's ``cond`` and the latest convergence estimates.

    Raises:
        NotImplementedError, ValueError: for a configuration the gradient would get wrong
            (:mod:`fdtdx.adjoint.validation`), a dispersive Device material among them.

    Warns:
        UserWarning: when a named design region leaves Device cells uncovered.
    """
    names = validation.as_names(objective_detectors)
    detectors = validation.objective_detectors(objects, names)
    validation.check_scene(objects, arrays, config)
    validation.check_device_materials(objects)
    arrays = device_dispersion_as_applied(arrays, objects)
    regions = design_regions(objects, design_detector)
    validation.check_outside_pml(objects, regions)
    validation.check_sources_outside(objects, regions)
    if design_detector is not None:
        validation.check_coverage(objects, regions, design_detector, refuse=False)
    if window is None:
        window = gaussian_window(int(config.time_steps_total), dtype=config.dtype)
    return make_phasor_fn(
        arrays,
        objects,
        config,
        key,
        names=names,
        detectors=detectors,
        single=isinstance(objective_detectors, str),
        design=design_detector,
        window=window,
        cond_limit=cond_limit,
        tail_tolerance=tail_tolerance,
    )


class ReciprocityParamFn:
    """``param_fn(params, **transform_kwargs) -> phasors``, from :func:`reciprocity_param_fn`.

    Attributes:
        objects: the applied scene both solves run on. Read detector state from it, e.g. the
            reference mode ``ModeOverlapDetector.compute_overlap`` needs.
        phasor_fn: the underlying ``phasor_fn(inv_permittivities, electric_conductivity)``.
        diagnostics: ``phasor_fn.diagnostics``.
    """

    def __init__(self, phasor_fn: ReciprocityPhasorFn, arrays: ArrayContainer, objects: ObjectContainer, key):
        self.phasor_fn = phasor_fn
        self.objects = objects
        self._arrays = arrays
        self._key = key
        # only the Devices: the whole scene would re-apply every overlapping object (a mode
        # solve per port) on every call, for a result discarded here
        self._design_objects = ObjectContainer(object_list=[objects.volume, *objects.devices], volume_idx=0)

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self.phasor_fn.diagnostics

    def __call__(self, params: Any, **transform_kwargs: Any):
        updated, _, _ = apply_params(self._arrays, self._design_objects, params, self._key, **transform_kwargs)
        return self.phasor_fn(updated.inv_permittivities, updated.electric_conductivity)


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
    """Differentiable objective phasors as a function of the Device parameters.

    ``param_fn(params, **transform_kwargs)`` runs ``apply_params`` (so ``param_transforms``
    and schedules such as ``beta=`` are ordinary JAX) and :func:`reciprocity_phasor_fn`; the
    gradient is a ``ParameterContainer``, equal to ``jax.grad`` of ``apply_params`` then
    ``run_fdtd`` with ``GradientConfig("checkpointed")``. It carries the design dependence of
    both the permittivity and the electric conductivity ``apply_params`` writes (lossy Device
    materials).

    ``objects`` may come straight from ``place_objects``: the objects it left unapplied
    (sources and ports sharing a Device's projection) are applied once here, exactly as
    ``apply_params`` would (:func:`~fdtdx.adjoint.design.apply_objects_once`), and the
    Device cells' dispersion coefficients are zeroed as ``apply_params`` does
    (:func:`~fdtdx.adjoint.design.device_dispersion_as_applied`). ``param_fn.objects`` is
    the applied scene.

    Args: as :func:`reciprocity_phasor_fn`; ``key`` also seeds the one-time application.
        Leave ``design_detector`` at ``None``: every Device is exactly the set of cells
        ``apply_params`` writes. A named region must cover every Device cell.

    Raises:
        NotImplementedError: for a dispersive Device material, or an applied object inside a Device.
        ValueError: if ``design_detector`` leaves Device cells uncovered.
    """
    if design_detector is not None:
        validation.check_coverage(objects, design_regions(objects, design_detector), design_detector, refuse=True)
    # as apply_params leaves it, for the objects applied once below
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
