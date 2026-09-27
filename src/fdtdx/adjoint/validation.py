"""The configurations the reciprocity gradient refuses.

Each was measured silently wrong against ``GradientConfig(method="checkpointed")`` (numbers in
``notes/adjoint/05-guards.md``), so it raises instead of approximating. The checks run when the
reciprocity gradient is traced and refuse that gradient only: the forward run, and every other
method, are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.adjoint.objective import is_box_projection, pml_weight
from fdtdx.adjoint.source import AdjointCurrentSource
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.utils import is_jax_tracer
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.object import INVALID_SLICE_TUPLE_3D
from fdtdx.objects.sources.source import Source
from fdtdx.objects.sources.tfsf_region import TFSFPlaneSourceRegion

_CHECKPOINTED = "GradientConfig(method='checkpointed')"

#: The refusal of a grid whose cell widths vary along ``{axes}``.
STRETCHED_GRID = (
    "The grid's cell widths vary along {axes}. FDTDX's curls then scale E by the primal and H by the dual "
    "(averaged) widths, a pairing the adjoint currents and the gradient kernel do not weight. Use one width per "
    f"axis (UniformGrid or QuasiUniformGrid) or {_CHECKPOINTED}."
)


def _overlap(a, b) -> bool:
    """Whether two grid slice tuples intersect on all three axes."""
    return all(lo_a < hi_b and lo_b < hi_a for (lo_a, hi_a), (lo_b, hi_b) in zip(a, b))


def objective_detectors(objects: ObjectContainer, names: Sequence[str]) -> list[PhasorDetector]:
    """The named objective monitors, each checked to be transposable."""
    detectors = []
    for name in names:
        det = objects[name]
        if not isinstance(det, PhasorDetector):
            raise NotImplementedError(
                f"Detector {name!r} is a {type(det).__name__}, which does not accumulate complex phasors, so "
                "there is no linear transpose. Compute the figure of merit from a PhasorDetector, or use "
                f"{_CHECKPOINTED}."
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


def check_grid(config: SimulationConfig) -> jax.Array | None:
    """Refuse a grid whose cell widths vary along an axis, at ``RectilinearGrid``'s own tolerance.

    A grid rebuilt inside a trace (``config.aset`` under ``jax.jit``) has traced widths: their
    check is returned, for the backward rule to refuse on.
    """
    grid = config.resolved_grid
    if grid is None or grid._is_uniform:
        return None
    stretched = []
    for edges, widths in zip((grid.x_edges, grid.y_edges, grid.z_edges), grid._cell_widths):
        eps = float(jnp.finfo(edges.dtype).eps) if jnp.issubdtype(edges.dtype, jnp.floating) else 0.0
        roundoff = 8.0 * eps * jnp.max(jnp.abs(edges))
        stretched.append(jnp.max(jnp.abs(widths - widths[0])) > 1e-4 * jnp.abs(widths[0]) + roundoff)
    flags = jnp.stack(stretched)
    if is_jax_tracer(flags):
        return flags
    axes = [axis for axis, flag in zip("xyz", np.asarray(flags)) if flag]
    if axes:
        raise NotImplementedError(STRETCHED_GRID.format(axes=", ".join(axes)))
    return None


def check_scene(objects: ObjectContainer, arrays: ArrayContainer, config: SimulationConfig) -> jax.Array | None:
    """Refuse a scene the gradient kernel does not model; returns :func:`check_grid`'s traced check, if any.

    A scene without Devices, Bloch wave vectors, full 3x3 material tensors, and the refusals of
    :func:`check_device_materials`, :func:`check_outside_pml`, :func:`check_sources_outside` and
    :func:`check_grid`.
    """
    if not objects.devices:
        raise ValueError(
            "GradientConfig(method='reciprocity') takes the gradient inside the Devices, and the scene has none."
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
    check_device_materials(objects)
    check_outside_pml(objects)
    check_sources_outside(objects)
    return check_grid(config)


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
            "dependence of inv_permittivities only, not of the dispersion coefficients apply_params also writes. "
            f"Use {_CHECKPOINTED}."
        )


def check_outside_pml(objects: ObjectContainer) -> None:
    """Refuse a Device reaching into a PML, where the kernel's reciprocal pairing does not hold."""
    inside = [
        f"{device.name!r} (the {pml.descriptive_name} PML)"
        for device in objects.devices
        for pml in objects.pml_objects
        if _overlap(device.grid_slice_tuple, pml.grid_slice_tuple)
    ]
    if inside:
        raise NotImplementedError(
            f"Device(s) {', '.join(inside)} reach into a PML, where the gradient would be wrong. Keep the Devices "
            f"out of the PML, or use {_CHECKPOINTED}."
        )


def _injection_cells(src: Source) -> list:
    """The cells a source injects into: a TFSF box's faces, not the scatterer inside it; else its own cells."""
    if (
        isinstance(src, TFSFPlaneSourceRegion)
        and src._face_E_slice_tuples is not None
        and src._face_H_slice_tuples is not None
    ):
        return [*src._face_E_slice_tuples, *src._face_H_slice_tuples]
    return [src.grid_slice_tuple]


def check_sources_outside(objects: ObjectContainer) -> None:
    """Refuse a stock source injecting inside a Device.

    Stock sources freeze the ``inv_eps`` of their injection (``PointDipoleSource`` caches it,
    the TFSF plane sources ``stop_gradient`` it), so ``run_fdtd``'s gradient omits that term,
    and the kernel's ``(E_new - E_old) / inv_eps`` factoring would not.
    """
    offenders = [
        f"{src.name!r} ({type(src).__name__})"
        for device in objects.devices
        for src in objects.sources
        if not isinstance(src, AdjointCurrentSource)
        and getattr(src, "_grid_slice_tuple", INVALID_SLICE_TUPLE_3D) != INVALID_SLICE_TUPLE_3D
        and any(_overlap(tuple(cells), device.grid_slice_tuple) for cells in _injection_cells(src))
    ]
    if offenders:
        raise NotImplementedError(
            f"Source(s) {', '.join(dict.fromkeys(offenders))} inject inside a Device. Stock sources freeze the "
            "inverse permittivity of their injection, so the gradient would differ from run_fdtd's own. Move the "
            f"source out of the Devices, or use {_CHECKPOINTED}."
        )


def check_objectives_outside_pml(objects: ObjectContainer, labelled_blocks: Sequence[tuple[str, tuple]]) -> None:
    """Refuse an objective whose adjoint currents reach the lossy part of a PML.

    ``labelled_blocks`` is ``(label, block)`` per adjoint-current block. The reciprocal pairing
    fails wherever the local CPML strength is nonzero; only the interface cell, graded to zero,
    is exact.
    """
    reaching = sorted({label for label, block in labelled_blocks if pml_weight(objects, block) is not None})
    if reaching:
        raise NotImplementedError(
            f"Objective detector channel(s) {', '.join(repr(r) for r in reaching)} read fields inside a PML, where "
            "the reciprocal pairing of their adjoint currents does not hold. Crop the monitors to the non-PML "
            f"interior, or use {_CHECKPOINTED}."
        )


#: A source whose waveform over the last tenth of the run exceeds this fraction of its peak is
#: still injecting: its phasors never converge.
WAVEFORM_END_TOLERANCE = 1e-6

#: The largest spectrum of a source at a frequency a strided objective monitor folds onto an
#: objective frequency, relative to the spectrum at that frequency itself.
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
            "decays, and what it injects at the objective frequencies, cannot be checked. Close over the objects "
            f"instead of passing them through jit, or use {_CHECKPOINTED}."
        )
    return np.where(on, np.real(np.asarray(amplitude, dtype=np.complex128)), 0.0)


def check_source_waveforms(
    objects: ObjectContainer,
    config: SimulationConfig,
    detectors: Sequence[PhasorDetector],
    *,
    convergence: bool = True,
) -> list[np.ndarray]:
    """Refuse a source whose phasors cannot converge; return how much a strided monitor folds in, per row.

    A stock source still injecting at the end of the run is refused, unless ``convergence`` is
    False (``tail_tolerance=None``). For an objective monitor with ``dft_subsample`` stride
    ``k > 1``, the sources' spectrum at the frequencies ``2 pi m / (k dt) +- w`` that the strided
    DFT at ``w`` cannot tell from ``w``, relative to their spectrum at ``w``, is returned per
    detector and row (0 unstrided), for the backward rule to refuse where the figure of merit
    reads such a row. DC content is left to the convergence estimate: whether it leaves a static
    remainder depends on the geometry.
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
                "never converge, and the reciprocity gradient differs from run_fdtd's by percent-level errors. Use a "
                f"decaying pulse (GaussianPulseProfile) the run outlasts, switch the source off, or use {_CHECKPOINTED}."
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
    """``ALIAS_TOLERANCE``, or float32's own floor for a float32 run, whose waveform folds about 1e-6
    of a row's spectrum from round-off alone."""
    eps = float(np.finfo(np.dtype(config.dtype)).eps) if np.issubdtype(np.dtype(config.dtype), np.floating) else 0.0
    return max(ALIAS_TOLERANCE, 100.0 * eps)


def alias_message(names: Sequence[str], detectors: Sequence[PhasorDetector], limit: float) -> str:
    """The refusal of a read row a strided objective monitor folds foreign spectrum onto."""
    strided = [f"{n!r} (every {int(d._dft_stride)}th step)" for n, d in zip(names, detectors) if d._dft_stride > 1]
    return (
        f"The reciprocity gradient is refused: the figure of merit reads a row of {', '.join(strided)} that the stride "
        "(dft_subsample) folds frequencies 2*pi*m/(k*dt) +- w onto, where the sources inject more than "
        f"{limit:.0e} of the row's own spectrum. The stored phasor then mixes those bands in, which the transpose "
        "cannot model. Record every step (dft_subsample=1), read only rows the stride keeps clean, or use "
        f"{_CHECKPOINTED}."
    )


def check_distinct_frequencies(omegas: Sequence[float], names: Sequence[str], rows: Sequence[np.ndarray]) -> None:
    """Refuse objective frequencies less than 1e-6 apart that were not merged as one: the same frequency
    written differently (600e-9 and float32(600e-9)), which no run can separate."""
    w = np.asarray(omegas, dtype=np.float64)
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            if abs(w[i] - w[j]) <= 1e-6 * abs(w[i]):
                owners = [n for n, r in zip(names, rows) if i in r or j in r]
                raise ValueError(
                    f"The objective detectors {owners} record angular frequencies {w[i]!r} and {w[j]!r}, "
                    f"{abs(w[i] - w[j]) / abs(w[i]):.1e} apart: one frequency written two ways. Build the monitors "
                    "from the same WaveCharacter (or the same float)."
                )


def check_conditioning(cond: float, cond_limit: float) -> None:
    """Refuse an adjoint amplitude solve worse conditioned than ``cond_limit``."""
    if not np.isfinite(cond) or cond > cond_limit:
        raise ValueError(
            f"Adjoint amplitude solve is ill-conditioned (cond={cond:.3e} > {cond_limit:.1e}): the run is too "
            "short to separate the objective frequencies, and the gradient would be garbage. Lengthen the "
            f"simulation, space the frequencies further apart, or use {_CHECKPOINTED}."
        )
