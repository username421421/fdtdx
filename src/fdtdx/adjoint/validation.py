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
import jax.numpy as jnp
import numpy as np

from fdtdx.adjoint.objective import is_box_projection, pml_weight
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.utils import is_jax_tracer
from fdtdx.core.null import Null
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.detectors.detector import Detector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.object import INVALID_SLICE_TUPLE_3D, SimulationObject, slices_overlap
from fdtdx.objects.sources.adjoint import AdjointCurrentSource
from fdtdx.objects.sources.source import Source
from fdtdx.objects.sources.tfsf import TFSFPlaneSource
from fdtdx.objects.sources.tfsf_region import TFSFPlaneSourceRegion

_CHECKPOINTED = "run_fdtd with GradientConfig(method='checkpointed')"


def as_names(names: str | Sequence[str]) -> tuple[str, ...]:
    """A name or a sequence of names, as a tuple."""
    return (names,) if isinstance(names, str) else tuple(names)


def objective_detectors(objects: ObjectContainer, names: Sequence[str]) -> list[PhasorDetector]:
    """The named objective monitors, each checked to be transposable (at any frequencies)."""
    if not names:
        raise ValueError("at least one objective detector is required")
    repeated = sorted({n for n in names if list(names).count(n) > 1})
    if repeated:
        # both would drive one adjoint current, under one name, and one cotangent would be dropped
        raise ValueError(
            f"objective_detectors names {', '.join(map(repr, repeated))} more than once: list each monitor once, "
            "and read it as often as you like in the figure of merit."
        )
    detectors = []
    for name in names:
        det = objects[name]
        if not isinstance(det, Detector):
            raise ValueError(f"{name!r} is a {type(det).__name__}, not a Detector")
        if not isinstance(det, PhasorDetector):
            raise NotImplementedError(
                f"Detector {name!r} is a {type(det).__name__}, which does not accumulate complex phasors, so "
                "there is no linear transpose. Record a PhasorDetector and compute the quantity in JAX on top, "
                f"or use {_CHECKPOINTED}."
            )
        _check_objective_settings(det, name)
        detectors.append(det)
    return detectors


def _check_objective_settings(det: PhasorDetector, name: str) -> None:
    cfg = det._config
    # the switch's own on-list, before any DFT stride (a stride is checked against the sources'
    # spectra, in check_source_waveforms)
    on = det.switch.calculate_on_list(
        num_total_time_steps=cfg.time_steps_total, time_step_duration=cfg.time_step_duration
    )
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
        (det.reduce_volume, "must have reduce_volume=False."),
        (det.inverse, "has inverse=True: it records during a backward run only, so a forward run stores nothing."),
    )
    for refused, why in refusals:
        if refused:
            raise NotImplementedError(f"The objective detector {name!r} {why} Or use {_CHECKPOINTED}.")


def _stretched_axes(config: SimulationConfig) -> list[str]:
    """The axes whose cell widths vary, at ``RectilinearGrid``'s own uniformity tolerance."""
    grid = config.resolved_grid
    # both static, so also known when a config.aset under jit has traced the edges; a uniform grid
    # (the curls then apply no metric) is exact even with width jitter below its tolerance
    if grid is None or grid._is_uniform:
        return []
    return ["xyz"[axis] for axis, uniform in enumerate(grid._uniform_axes) if not uniform]


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
        and any(getattr(src, f, None) is None or isinstance(getattr(src, f), Null) for f in _applied_fields(src))
    ]
    if unapplied:
        raise ValueError(
            f"Source(s) {', '.join(unapplied)} were never applied: place_objects skips every object whose "
            "projection overlaps a Device's. Use reciprocity_param_fn, which applies them once at setup, or pass "
            "the objects apply_params returned."
        )


def _applied_fields(src: TFSFPlaneSource) -> tuple[str, ...]:
    """The private fields ``apply`` sets on a TFSF source: per face for a box, else on its plane."""
    if isinstance(src, TFSFPlaneSourceRegion):
        return ("_face_incident_E", "_face_incident_H", "_face_time_offset_E", "_face_time_offset_H")
    return ("_E", "_H", "_time_offset_E", "_time_offset_H")


def _injection_cells(src: SimulationObject) -> list:
    """The cells a source injects into: a TFSF box's faces, not the scatterer inside it; else its own cells."""
    if (
        isinstance(src, TFSFPlaneSourceRegion)
        and src._face_E_slice_tuples is not None
        and src._face_H_slice_tuples is not None
    ):
        return [*src._face_E_slice_tuples, *src._face_H_slice_tuples]
    return [src.grid_slice_tuple]


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
        and any(slices_overlap(tuple(cells), region.grid_slice_tuple) for cells in _injection_cells(src))
    ]
    if offenders:
        raise NotImplementedError(
            f"Source(s) {', '.join(dict.fromkeys(offenders))} overlap the design region. Stock sources freeze the "
            "inverse permittivity in their injection, so run_fdtd's gradient omits their term and this one would "
            f"not. Move the source out of the design region, or use {_CHECKPOINTED}."
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


def check_objectives_outside_pml(objects: ObjectContainer, labelled_blocks: Sequence[tuple[str, tuple]]) -> None:
    """Refuse an objective whose adjoint currents reach the lossy part of a PML.

    ``labelled_blocks`` is ``(label, block)`` per adjoint-current block. The reciprocal pairing
    fails wherever the local CPML strength is nonzero; only the interface cell, graded to zero,
    is exact. A monitor spanning the cross-section into the PML was 38% wrong (ceviche corner),
    one two cells deep 0.5%.
    """
    reaching = sorted({label for label, block in labelled_blocks if pml_weight(objects, block) is not None})
    if reaching:
        raise NotImplementedError(
            f"Objective detector channel(s) {', '.join(repr(r) for r in reaching)} read fields inside a PML (the adjoint "
            "current then lies in its lossy cells, where the reciprocal pairing does not hold and the gradient is "
            "wrong). Crop the monitors to the non-PML interior, or use "
            f"{_CHECKPOINTED}."
        )


#: A source whose waveform over the last tenth of the run exceeds this fraction of its peak is
#: still injecting: its phasors never converge (FDTDX's default SingleFrequencyProfile was 1-4%
#: wrong, silently).
WAVEFORM_END_TOLERANCE = 1e-6

#: The largest spectrum of a source at a frequency a strided objective monitor folds onto an
#: objective frequency, relative to the spectrum at that frequency itself (a folded band gave
#: cosine -0.63; relative to the strongest frequency's instead, a weak row folded at 6e-2 of its
#: own content passed, and its gradient was 11% off).
ALIAS_TOLERANCE = 1e-6


def source_waveform(source: Source, config: SimulationConfig) -> np.ndarray:
    """What ``source`` injects at every time step: its profile at the switch-adjusted time, zero while off.

    The small per-component time offsets of plane and TFSF sources are left out; they do not
    change whether the waveform decays or its spectrum at the frequencies checked.
    """
    steps = int(config.time_steps_total)
    dt = float(config.time_step_duration)
    on = np.asarray(source.switch.calculate_on_list(num_total_time_steps=steps, time_step_duration=dt), dtype=bool)
    on_index = np.asarray(
        source.switch.calculate_time_step_to_on_arr_idx(num_total_time_steps=steps, time_step_duration=dt),
        dtype=np.float64,
    )
    with jax.ensure_compile_time_eval():
        amplitude = source.temporal_profile.get_amplitude(
            time=jnp.asarray(on_index * dt),
            period=source.wave_character.get_period(),
            phase_shift=source.wave_character.phase_shift,
        )
    if is_jax_tracer(amplitude):
        raise NotImplementedError(
            f"The waveform of source {source.name!r} is traced (its temporal profile is a jit argument), so whether it "
            "decays, and what it injects at the frequencies the reciprocity gradient needs, cannot be checked. Close "
            f"over the objects instead of passing them through jit, or use {_CHECKPOINTED}."
        )
    return np.where(on, np.real(np.asarray(amplitude, dtype=np.complex128)), 0.0)


def check_source_waveforms(
    objects: ObjectContainer,
    config: SimulationConfig,
    detectors: Sequence[PhasorDetector],
    names: Sequence[str],
    *,
    convergence: bool = True,
) -> list[np.ndarray]:
    """Refuse a source whose phasors cannot converge; return how much a strided monitor folds in, per row.

    Per stock source: still injecting at the end of the run, unless ``convergence`` is False
    (``tail_tolerance=None``), is refused. For an objective monitor with ``dft_subsample`` stride
    ``k > 1``, the sources' spectrum at the frequencies ``2 pi m / (k dt) +- w`` that the strided
    DFT at ``w`` cannot tell from ``w``, relative to their spectrum at ``w``, is returned per
    detector and row (0 unstrided), for the backward rule to refuse where the figure of merit reads
    such a row (an unread aliased row was refused, though its gradient was exact to 1.9e-9). DC
    content is not refused here: whether it leaves a static remainder depends on the geometry (a
    plane source across the cell leaves none: rel 6e-7 at DC 3e-3), and a remainder is measured by
    the convergence estimate.
    """
    steps = int(config.time_steps_total)
    dt = float(config.time_step_duration)
    t = np.arange(steps) * dt
    folded_rows = [np.zeros(len(det._angular_frequencies)) for det in detectors]
    strided = [
        (i, int(det._dft_stride), np.asarray([float(w) for w in det._angular_frequencies]))
        for i, det in enumerate(detectors)
        if int(det._dft_stride) > 1
    ]
    for src in objects.sources:
        if isinstance(src, AdjointCurrentSource):
            continue
        s = source_waveform(src, config)
        peak = float(np.max(np.abs(s)))
        if peak == 0.0:
            continue
        label = f"{src.name!r} ({type(src.temporal_profile).__name__})"
        end = float(np.max(np.abs(s[int(0.9 * steps) :]))) / peak
        if convergence and end > WAVEFORM_END_TOLERANCE:
            raise NotImplementedError(
                f"Source {label} still injects {end:.1e} of its peak over the last tenth of the run, so the phasors "
                "never converge, and the reciprocity gradient differs from run_fdtd's by percent-level errors without "
                "any other sign. Use a decaying pulse (GaussianPulseProfile) the run outlasts, or switch the source "
                f"off, or use {_CHECKPOINTED}."
            )
        for i, stride, omegas in strided:
            fold = 2.0 * np.pi / (stride * dt)
            nyquist = np.pi / dt
            for row, w in enumerate(omegas):
                images = np.asarray(
                    [m * fold + sign * w for m in range(1, int(nyquist / fold) + 2) for sign in (1.0, -1.0)]
                )
                images = np.abs(images[(np.abs(images) > 0.0) & (np.abs(images) <= nyquist)])
                if images.size == 0:
                    continue
                own = max(float(np.abs(np.exp(1j * w * t) @ s)), np.finfo(float).tiny)
                folded = float(np.max(np.abs(np.exp(1j * np.outer(images, t)) @ s))) / own
                folded_rows[i][row] = max(folded_rows[i][row], folded)
    return folded_rows


def alias_limit(config: SimulationConfig) -> float:
    """``ALIAS_TOLERANCE``, or float32's own floor for a float32 run: its waveform, evaluated in float32,
    folds 1.5e-6 of a row's spectrum from round-off alone (the gradient was exact to 1.8e-6)."""
    eps = float(np.finfo(np.dtype(config.dtype)).eps) if np.issubdtype(np.dtype(config.dtype), np.floating) else 0.0
    return max(ALIAS_TOLERANCE, 100.0 * eps)


def alias_message(names: Sequence[str], detectors: Sequence[PhasorDetector], limit: float) -> str:
    """The refusal of a read row a strided objective monitor folds foreign spectrum onto."""
    strided = [f"{n!r} (every {int(d._dft_stride)}th step)" for n, d in zip(names, detectors) if d._dft_stride > 1]
    return (
        f"The reciprocity gradient is refused: the figure of merit reads a row of {', '.join(strided)} that the stride "
        "(dft_subsample) folds frequencies 2*pi*m/(k*dt) +- w onto, where the sources inject more than "
        f"{limit:.0e} of the row's own spectrum. The stored phasor then mixes those bands in, which the transpose "
        "cannot model: the gradient can point the wrong way. Record every step (dft_subsample=1), or read only "
        f"rows the stride keeps clean, or use {_CHECKPOINTED}."
    )


def check_distinct_frequencies(omegas: Sequence[float], names: Sequence[str], rows: Sequence[np.ndarray]) -> None:
    """Refuse objective frequencies apart by less than 1e-6 that were not merged as the same one.

    Built from different floats for one wavelength (600e-9 and float32(600e-9): 3.5e-8 apart), they
    are two frequencies no run can separate, and the amplitude solve refused them as ill-conditioned
    with the advice to run longer, which cannot help.
    """
    w = np.asarray(omegas, dtype=np.float64)
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            if abs(w[i] - w[j]) <= 1e-6 * abs(w[i]):
                owners = [n for n, r in zip(names, rows) if i in r or j in r]
                raise ValueError(
                    f"The objective detectors {owners} record angular frequencies {w[i]!r} and {w[j]!r}, "
                    f"{abs(w[i] - w[j]) / abs(w[i]):.1e} apart: meant as one frequency, they differ in how it was "
                    "written. Build the monitors from the same WaveCharacter (or the same float), so they record "
                    "exactly one frequency."
                )


def check_conditioning(cond: float, cond_limit: float) -> None:
    """Refuse an adjoint amplitude solve worse conditioned than ``cond_limit``."""
    if not np.isfinite(cond) or cond > cond_limit:
        raise ValueError(
            f"Adjoint amplitude solve is ill-conditioned (cond={cond:.3e} > {cond_limit:.1e}): the run is too "
            "short to separate the objective frequencies, and the gradient would be garbage. Lengthen the "
            f"simulation, space the frequencies further apart, or use {_CHECKPOINTED}."
        )
