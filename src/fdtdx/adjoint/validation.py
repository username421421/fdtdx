"""Every configuration the reciprocity gradient refuses, checked at setup.

Each refusal is a case that was measured silently wrong against
``run_fdtd(GradientConfig("checkpointed"))`` (numbers in ``notes/adjoint/05-guards.md``),
so it raises rather than approximating. The usual alternative for a refused scene is
``run_fdtd`` with ``GradientConfig(method="checkpointed")``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import jax
import numpy as np

from fdtdx.adjoint.objective import is_box_projection
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.utils import is_jax_tracer
from fdtdx.core.null import Null
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.detectors.detector import Detector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.object import INVALID_SLICE_TUPLE_3D, SimulationObject
from fdtdx.objects.sources.adjoint import AdjointCurrentSource
from fdtdx.objects.sources.tfsf import TFSFPlaneSource

_CHECKPOINTED = "run_fdtd with GradientConfig(method='checkpointed')"


def as_names(names: str | Sequence[str]) -> tuple[str, ...]:
    """A name or a sequence of names, as a tuple."""
    return (names,) if isinstance(names, str) else tuple(names)


def slices_overlap(a, b) -> bool:
    """Do two grid slice tuples intersect on all three axes (``check_overlap`` asks for any one)?"""
    return all(lo_a < hi_b and lo_b < hi_a for (lo_a, hi_a), (lo_b, hi_b) in zip(a, b))


def objective_detectors(objects: ObjectContainer, names: Sequence[str]) -> list[PhasorDetector]:
    """The named objective monitors, each checked to be transposable; they must share frequencies."""
    if not names:
        raise ValueError("at least one objective detector is required")
    detectors = []
    for name in names:
        det = objects[name]
        if not isinstance(det, Detector):
            raise ValueError(f"{name!r} is a {type(det).__name__}, not a Detector")
        if not isinstance(det, PhasorDetector):
            raise NotImplementedError(
                f"Detector {name!r} is a {type(det).__name__}, which does not accumulate complex phasors, so "
                "there is no linear transpose. Record a PhasorDetector and compute the quantity in JAX on top."
            )
        _check_objective_settings(det, name)
        detectors.append(det)
    omegas = tuple(float(w) for w in detectors[0]._angular_frequencies)
    for name, det in zip(names, detectors):
        # a tolerance: a float32 run stores frequencies in float32
        det_omegas = tuple(float(w) for w in det._angular_frequencies)
        if len(det_omegas) != len(omegas) or not np.allclose(det_omegas, omegas, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"detector {name!r} has frequencies {det_omegas}, expected {omegas}: all objective "
                "monitors share one amplitude solve, so they must share frequencies"
            )
    return detectors


def _check_objective_settings(det: PhasorDetector, name: str) -> None:
    cfg = det._config
    # the switch's own on-list, before any DFT stride
    on = det.switch.calculate_on_list(
        num_total_time_steps=cfg.time_steps_total, time_step_duration=cfg.time_step_duration
    )
    stride = int(det._dft_stride)
    f_max = max(abs(float(wc.get_frequency())) for wc in det.wave_characters)
    samples = 1.0 / (stride * float(cfg.time_step_duration) * f_max)
    refusals = (
        (
            is_box_projection(det) and len(det.components) != 6,
            f"records {len(det.components)} components; box-mode field projection concatenates E and H and "
            "needs all six (Ex, Ey, Ez, Hx, Hy, Hz).",
        ),
        (
            det.apodization is not None,
            "has an apodization window, which the transpose does not apply. Remove it.",
        ),
        (
            not all(on),
            f"records only {sum(on)} of {len(on)} time steps (its switch). A phasor over part of the run is "
            "not a frequency-domain quantity. Record every time step (the default OnOffSwitch()).",
        ),
        (
            stride > 1 and samples < 4.0,
            f"has dft_subsample stride {stride}, {samples:.2f} samples per period of its highest frequency. "
            "Below 4 the strided phasor aliases and only its unaliased part is transposed. Use "
            "dft_subsample='auto' or a smaller stride.",
        ),
        (det.reduce_volume, "must have reduce_volume=False."),
        (det.inverse, "has inverse=True: it records during a backward run only, so a forward run stores nothing."),
    )
    for refused, why in refusals:
        if refused:
            raise NotImplementedError(f"The objective detector {name!r} {why}")


def _stretched_axes(config: SimulationConfig) -> list[str]:
    """The axes whose cell widths vary, at ``RectilinearGrid``'s own uniformity tolerance."""
    grid = config.resolved_grid
    # static, so also known when a config.aset under jit has traced the edges
    return [] if grid is None else ["xyz"[axis] for axis, uniform in enumerate(grid._uniform_axes) if not uniform]


def check_scene(objects: ObjectContainer, arrays: ArrayContainer, config: SimulationConfig) -> None:
    """Refuse varying cell widths, Bloch wave vectors, full 3x3 material tensors and unapplied TFSF sources."""
    stretched = _stretched_axes(config)
    if stretched:
        raise NotImplementedError(
            f"The grid's cell widths vary along {', '.join(stretched)}. FDTDX's curls then scale E by the primal "
            "and H by the dual (averaged) widths, a pairing the adjoint currents and the gradient kernel do not "
            f"weight. Use one width per axis (UniformGrid or QuasiUniformGrid) or {_CHECKPOINTED}."
        )
    bloch = [
        f"{b.name!r} (bloch_vector={tuple(b.bloch_vector)})"
        for b in objects.boundary_objects
        if isinstance(b, BlochBoundary) and any(float(k) != 0.0 for k in b.bloch_vector)
    ]
    if bloch:
        raise NotImplementedError(
            f"Bloch boundaries with a nonzero wave vector ({', '.join(bloch)}) are not supported: the adjoint "
            f"solve would need the opposite wave vector. Use periodic boundaries (bloch_vector=0) or {_CHECKPOINTED}."
        )
    for label in ("inv_permittivities", "inv_permeabilities", "electric_conductivity", "magnetic_conductivity"):
        value = getattr(arrays, label)
        if isinstance(value, jax.Array) and value.ndim > 0 and value.shape[0] == 9:
            raise NotImplementedError(
                f"The scene stores {label} as a full 3x3 tensor (shape {tuple(value.shape)}). The adjoint current, "
                "its lossy correction and the gradient kernel assume an isotropic or diagonal material (1 or 3 "
                f"components). Use {_CHECKPOINTED}."
            )
    unapplied = [
        f"{src.name!r} ({type(src).__name__})"
        for src in objects.sources
        if isinstance(src, TFSFPlaneSource)
        and any(
            getattr(src, f, None) is None or isinstance(getattr(src, f), Null)
            for f in ("_E", "_H", "_time_offset_E", "_time_offset_H")
        )
    ]
    if unapplied:
        raise ValueError(
            f"Source(s) {', '.join(unapplied)} were never applied: place_objects skips every object whose "
            "projection overlaps a Device's. Use reciprocity_param_fn, which applies them once at setup, or pass "
            "the objects apply_params returned."
        )


def check_outside_pml(objects: ObjectContainer, regions: Sequence[SimulationObject]) -> None:
    """Refuse a design region reaching into a PML, where the kernel's reciprocal pairing does not hold."""
    inside = [
        f"{region.name!r} (the {pml.descriptive_name} PML)"
        for region in regions
        for pml in objects.pml_objects
        if slices_overlap(region.grid_slice_tuple, pml.grid_slice_tuple)
    ]
    if inside:
        raise NotImplementedError(
            f"Design region(s) {', '.join(inside)} reach into a PML, where the gradient would be wrong. Keep "
            f"design regions out of the PML, or use {_CHECKPOINTED}."
        )


def check_sources_outside(objects: ObjectContainer, regions: Sequence[SimulationObject]) -> None:
    """Refuse a stock source inside a design region.

    Stock sources freeze the ``inv_eps`` in their injection (``PointDipoleSource`` caches it,
    the TFSF plane sources ``stop_gradient`` it), which the kernel's ``(E_new - E_old) / inv_eps``
    factoring does not model. An ``AdjointCurrentSource`` reads it live and is exact anywhere.
    """
    offenders = [
        f"{src.name!r} ({type(src).__name__})"
        for region in regions
        for src in objects.sources
        if not isinstance(src, AdjointCurrentSource)
        and getattr(src, "_grid_slice_tuple", INVALID_SLICE_TUPLE_3D) != INVALID_SLICE_TUPLE_3D
        and slices_overlap(src.grid_slice_tuple, region.grid_slice_tuple)
    ]
    if offenders:
        raise NotImplementedError(
            f"Source(s) {', '.join(dict.fromkeys(offenders))} overlap the design region. Stock sources freeze the "
            "inverse permittivity in their injection, so run_fdtd's gradient omits their term and this one would "
            "not. Move the source out of the design region, or drive the scene with an AdjointCurrentSource."
        )


def uncovered_device_cells(
    objects: ObjectContainer, region_slices: Sequence[Sequence[tuple[int, int]]]
) -> list[tuple[str, int, int]]:
    """``[(device_name, uncovered_cells, device_cells), ...]`` for each Device with cells outside every region."""
    out = []
    for dev in objects.devices:
        lows = [int(lo) for lo, _ in dev.grid_slice_tuple]
        covered = np.zeros([int(hi) - lo for lo, (_, hi) in zip(lows, dev.grid_slice_tuple)], dtype=bool)
        for region in region_slices:
            # the region in Device-local cells; numpy clips it, and a region missing the Device is empty
            box = tuple(slice(max(int(r_lo) - lo, 0), max(int(r_hi) - lo, 0)) for lo, (r_lo, r_hi) in zip(lows, region))
            covered[box] = True
        missing = int(covered.size - covered.sum())
        if missing:
            out.append((str(dev.name), missing, int(covered.size)))
    return out


def check_coverage(objects: ObjectContainer, regions: Sequence[SimulationObject], design, *, refuse: bool) -> None:
    """Refuse (parameter level) or warn (phasor level) when named design regions leave Device cells uncovered.

    The gradient is zero outside the design regions, so every parameter mapped onto an
    uncovered cell would get a zero gradient instead of its own.
    """
    uncovered = uncovered_device_cells(objects, [r.grid_slice_tuple for r in regions])
    if not uncovered:
        return
    parts = ", ".join(f"{name!r} ({missing} of {total} cells)" for name, missing, total in uncovered)
    message = (
        f"The design region {design!r} does not cover every Device: {parts} lie outside it, and the gradient "
        "there is exactly zero instead of the true one. Leave design_detector=None, which is every Device, or "
        "name regions covering them all."
    )
    if refuse:
        raise ValueError(message)
    warnings.warn(message, UserWarning, stacklevel=3)


def check_device_materials(objects: ObjectContainer) -> None:
    """Refuse a dispersive Device material, whose ADE coefficients carry a design dependence the kernel lacks."""
    offenders = [
        f"{device.name!r}/{mat_name!r}"
        for device in objects.devices
        for mat_name, material in device.materials.items()
        if material.is_dispersive
    ]
    if offenders:
        raise NotImplementedError(
            f"Device material(s) {', '.join(offenders)} are dispersive. The reciprocity gradient carries the design "
            "dependence of inv_permittivities only, not of the dispersion coefficients apply_params also writes, "
            f"so the value and the gradient would both be wrong. Use {_CHECKPOINTED}."
        )


def conductive_devices(objects: ObjectContainer, arrays: ArrayContainer) -> list[str]:
    """The Devices whose cells ``apply_params`` gives a conductivity other than the placed one.

    A lossy Device material, or loss placed under a Device (``apply_params`` replaces it).
    """
    sigma = arrays.electric_conductivity
    if sigma is None:
        return []
    return [
        str(dev.name)
        for dev in objects.devices
        if any(np.any(np.asarray(m.electric_conductivity) != 0) for m in dev.materials.values())
        or (not is_jax_tracer(sigma) and bool(np.any(np.asarray(sigma[:, *dev.grid_slice]) != 0)))
    ]


def check_applied_outside_devices(objects: ObjectContainer) -> None:
    """Refuse an object with its own ``apply()`` inside a Device: its applied state would depend on the design."""
    inside = [
        f"{getattr(obj, 'name', None)!r} ({type(obj).__name__}) in {dev.name!r}"
        for obj in objects.object_list
        if type(obj).apply is not SimulationObject.apply
        for dev in objects.devices
        if slices_overlap(obj.grid_slice_tuple, dev.grid_slice_tuple)
    ]
    if inside:
        raise NotImplementedError(
            f"Object(s) {', '.join(inside)} overlap a Device and compute their state from the material there (a "
            "mode solve or an impedance), which changes with the design. reciprocity_param_fn applies them once, "
            f"which is exact only outside the Devices. Move them out of the design region, or use {_CHECKPOINTED}."
        )


def check_conditioning(cond: float, cond_limit: float) -> None:
    """Refuse an adjoint amplitude solve worse conditioned than ``cond_limit``."""
    if not np.isfinite(cond) or cond > cond_limit:
        raise ValueError(
            f"Adjoint amplitude solve is ill-conditioned (cond={cond:.3e} > {cond_limit:.1e}): the window is too "
            "short to separate the objective frequencies, and the gradient would be garbage. Lengthen the "
            "simulation, widen the window's sigma_frac, or space the frequencies further apart."
        )
