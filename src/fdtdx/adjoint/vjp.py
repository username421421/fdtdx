"""``jax.custom_vjp`` wrapper: reciprocity gradients for any DFT-monitor FoM.

Usage is the point of this module. You build two placed scenes -- a forward one
with your real source, and an adjoint one carrying an
:class:`~fdtdx.objects.sources.adjoint.AdjointCurrentSource` at the objective
monitor -- hand them to :func:`make_reciprocity_phasor_fn`, and get back a plain
JAX function

    phasors = phasor_fn(inv_permittivities)

that is differentiable by ``jax.grad`` and costs two forward solves per gradient
instead of differentiating through the time loop.

Because the differentiable boundary sits at the monitor's *raw phasors*, the
figure of merit is arbitrary: anything differentiable you write on top of those
phasors gets its cotangent from JAX, exactly as with Meep's ``FourierFields``.
Mode overlaps, near-to-far projections and diffraction orders are ordinary JAX
post-processing of the same phasors, so they compose without needing their own
adjoint-source rules.

    loss = lambda ie: my_fom(phasor_fn(ie))
    value, grad = jax.value_and_grad(loss)(inv_permittivities)

The design region needs no detector. By default it is every
:class:`~fdtdx.objects.device.device.Device` in the scene; the detector that
records the fields there is built internally, in the one configuration the
kernel is calibrated for (:data:`~fdtdx.adjoint.scene.DESIGN_DETECTOR_SETTINGS`).
Objective monitors are accepted at PhasorDetector's stock settings: both scaling
modes; ``exact_interpolation=True``, whose co-location stencil is transposed
exactly (:mod:`fdtdx.adjoint.recording`), including the padded whole-domain path
FDTDX takes for a detector touching the domain edge or a symmetry plane, or
spanning a periodic axis;
and ``dft_subsample`` strides, treated as the every-step DFT they estimate (see
Accuracy).

Restrictions, all enforced with an exception rather than silently approximated:

* On the objective monitors: ``reduce_volume=False``, no apodization, a switch
  that records every time step, and a ``dft_subsample`` stride leaving at least
  four samples per period of the highest objective frequency.
* No Bloch boundary with a nonzero ``bloch_vector`` (the adjoint scene would need
  the opposite wave vector), and no fully anisotropic nine-component
  ``inv_permittivities`` or ``inv_permeabilities`` (the adjoint current and the
  gradient kernel both assume a diagonal tensor).
* The design region must not contain a source. FDTDX sources scale their
  injection by the local ``inv_eps``, a term this gradient does not model.
* The returned gradient is zero outside the design regions. That is what
  inverse design wants, but it is not the full-grid gradient.

Accuracy. Reciprocity is a frequency-domain identity, so it equals the exact
discrete adjoint only once both DFTs have converged. Measured against
``checkpointed`` autodiff on a 20^3 float64 scene at three wavelengths, the
relative L2 error was 1.6e-4 at 100 fs and 6.1e-6 at 400 fs, i.e. it converges
with decay time rather than sitting at a fixed floor. Give the fields time to
leave the domain before trusting a tight tolerance.

A strided monitor (``dft_subsample`` > 1) records ``stride * sum over every
stride-th step``, which is exactly the sum of the every-step DFT at ``w`` and at
its ``stride - 1`` aliases ``w + k * 2 pi / (stride * dt)``. Only the ``w`` term
is transposed: the aliases sit near the grid's Nyquist frequency, where a
band-limited source puts no field, which is the premise of ``dft_subsample``
itself. The gradient error this leaves tracks the forward phasor's own aliasing.
Measured against the exact gradient of the strided recording: rel 5.4e-07 to
5.7e-07 at 21 down to 2.6 samples per period (forward aliasing at most 7.6e-09),
but 4.5e-01 at 2.1 samples per period, where the forward phasor is itself 70%
aliased. Strides below four samples per period of the highest objective
frequency, where FDTDX already warns the phasor may alias, are refused.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.adjoint.reciprocity import (
    DEFAULT_COND_LIMIT,
    DEFAULT_TAIL_TOLERANCE,
    _design_matrix,
    assemble_material_gradient,
    gaussian_window,
)
from fdtdx.adjoint.recording import FAMILY_AXIS, ChannelRecording, channel_recordings
from fdtdx.adjoint.scene import (
    DESIGN_DETECTOR_SETTINGS,
    canonical_components,
    detector_channels,
    internal_scene,
    is_box_projection,
)
from fdtdx.config import SimulationConfig
from fdtdx.constants import eta0
from fdtdx.core.null import Null
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import checkpointed_fdtd
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.object import INVALID_SLICE_TUPLE_3D
from fdtdx.objects.sources.adjoint import AdjointCurrentSource
from fdtdx.objects.sources.tfsf import TFSFPlaneSource


def _slices_overlap(a, b) -> bool:
    """Do two ``((lo, hi), (lo, hi), (lo, hi))`` grid slice tuples intersect?

    All three axes must overlap. ``SimulationObject.check_overlap`` is not used
    here: it reports True when the projections overlap on any single axis, which
    would call a source plane sitting above a design block an overlap.
    """
    return all(lo_a < hi_b and lo_b < hi_a for (lo_a, hi_a), (lo_b, hi_b) in zip(a, b))


def _reject_frozen_sources_in_design(objects, design_slice_tuple, skip=()):
    """Refuse a source that overlaps the design region but freezes its inv_eps.

    FDTDX injects an impressed current as ``E += -courant * inv_eps * J``, so a
    source inside the design region contributes a term to the design gradient.
    Every stock source suppresses that term:
    :class:`~fdtdx.objects.sources.dipole.PointDipoleSource` caches the factor in
    a private field during ``apply()``, and the TFSF plane sources wrap their
    injection in ``jax.lax.stop_gradient``.

    Our kernel's ``(E_new - E_old) / inv_eps`` factoring assumes the factor is
    live, so with a stock source overlapping the design region the kernel and
    FDTDX's own autodiff disagree badly: measured relative error 1.16 with a
    dipole at the centre of the design region.

    Clearing the caches makes the kernel self-consistent, but it then computes
    the *physically complete* gradient, which deliberately differs from what
    ``run_fdtd`` returns. Since matching the official pipeline is the contract
    here, this refuses instead of silently choosing one of the two conventions.

    Raises:
        NotImplementedError: naming the offending sources and both remedies.
    """
    offenders = []
    for obj in objects.object_list:
        name = getattr(obj, "name", None)
        if name in skip or not hasattr(obj, "update_E"):
            continue
        # INVALID_SLICE_TUPLE_3D, not None, is the unplaced sentinel; testing for
        # None silently disabled this guard.
        if getattr(obj, "_grid_slice_tuple", INVALID_SLICE_TUPLE_3D) == INVALID_SLICE_TUPLE_3D:
            continue
        if isinstance(obj, AdjointCurrentSource):
            continue  # reads inv_permittivities live, so the kernel is exact
        if _slices_overlap(obj.grid_slice_tuple, design_slice_tuple):
            offenders.append(f"{name!r} ({type(obj).__name__})")
    if offenders:
        raise NotImplementedError(
            f"Source(s) {', '.join(offenders)} overlap the design region. Stock FDTDX sources "
            "freeze the inverse permittivity in their injection (PointDipoleSource caches it, "
            "the TFSF plane sources stop_gradient it), so their contribution to the design "
            "gradient is suppressed and this kernel would disagree with run_fdtd by order one. "
            "Either move the source out of the design region, or drive the scene with an "
            "AdjointCurrentSource, which reads inv_permittivities live and is exact here."
        )


def _find(container, name: str, kind: str):
    for obj in container:
        if getattr(obj, "name", None) == name:
            return obj
    raise ValueError(f"No {kind} named {name!r}; found {[getattr(o, 'name', None) for o in container]}")


def _validate_objective_detector(det, role: str) -> None:
    """Accept any phasor detector whose state layout we can place adjoint currents for.

    The transpose only needs two things: complex phasors accumulated linearly
    from the fields, and a known mapping from each state key to the cells it
    reads. :func:`~fdtdx.adjoint.scene.detector_channels` decides the second and
    raises if it cannot. Anything a detector computes *on top* of its phasors is
    pure JAX above the VJP boundary and differentiates itself, which is why
    ``PhasorPoyntingFluxDetector.compute_poynting_flux``,
    ``ClosedSurfacePhasorPoyntingFluxDetector.compute_net_flux`` and
    ``FieldProjectionAngleDetector.project`` all work without special cases.
    """
    if not isinstance(det, PhasorDetector):
        raise NotImplementedError(
            f"The {role} detector is a {type(det).__name__}, which does not accumulate complex "
            "phasors, so there is no linear transpose to take. Record phasors with a "
            "PhasorDetector and compute the quantity you want in JAX on top."
        )
    if is_box_projection(det) and len(det.components) != 6:
        raise NotImplementedError(
            f"The {role} detector records {len(det.components)} components; box-mode field "
            "projection concatenates E and H and so needs all six (Ex, Ey, Ez, Hx, Hy, Hz)."
        )
    # scaling_mode is not checked: either mode is accepted, and its scale is divided
    # out of the adjoint target per detector in make_reciprocity_phasor_fn.
    if det.apodization is not None:
        raise NotImplementedError(
            f"The {role} detector has an apodization window. The transpose does not apply it to "
            "the adjoint current, so the gradient would be scaled per time step. Remove it."
        )
    # Judged on the switch's own on-list, before any DFT thinning, so a stride
    # (checked separately) is not mistaken for a restricted switch.
    cfg = det._config
    on_list = det.switch.calculate_on_list(
        num_total_time_steps=cfg.time_steps_total, time_step_duration=cfg.time_step_duration
    )
    if not all(on_list):
        raise NotImplementedError(
            f"The {role} detector records only {sum(on_list)} of {len(on_list)} time steps (its "
            "switch). A phasor summed over part of the run is not a frequency-domain quantity, so "
            "reciprocity does not give its gradient: measured rel 1.8e-01 at cosine 0.984 with "
            "recording starting at 20 fs of 150 fs, and rel 8.2e+02 at cosine 0.008 starting at "
            "40 fs, neither raising. Record every time step (the default OnOffSwitch())."
        )
    # A stride's samples are weighted to estimate the every-step DFT and its scale is
    # divided out with the scaling mode's; only its aliases are neglected (module
    # docstring), which is safe while the recording itself does not alias. FDTDX
    # warns below four samples per period of the highest frequency; here that is
    # where the neglected term stops being negligible, so it is refused there.
    # exact_interpolation is not checked: its stencil is transposed in fdtdx.adjoint.recording.
    stride = int(det._dft_stride)
    f_max = max(abs(float(wc.get_frequency())) for wc in det.wave_characters)
    samples_per_period = 1.0 / (stride * float(cfg.time_step_duration) * f_max)
    if stride > 1 and samples_per_period < 4.0:
        raise NotImplementedError(
            f"The {role} detector's dft_subsample resolves to stride {stride}, {samples_per_period:.2f} "
            "samples per period of its highest frequency. Below 4 the strided phasor aliases (FDTDX warns "
            "at placement), and the gradient transposes only its unaliased part: measured rel 4.5e-01 at "
            "2.1 samples per period with a 0.1 f0 source (forward phasor 70% aliased), and 7.5e-01 at 2.6 "
            "with a 0.5 f0 source. Use dft_subsample='auto' or a smaller stride."
        )
    if det.reduce_volume:
        raise NotImplementedError(f"The {role} detector must have reduce_volume=False.")


def _reject_bloch_and_full_tensors(objects: ObjectContainer, arrays: ArrayContainer, role: str) -> None:
    """Refuse the two scene configurations that are silently wrong at setup.

    * A Bloch boundary with a nonzero wave vector. FDTDX's Bloch update is not its
      own transpose: the adjoint solve would need the opposite wave vector, and the
      derived adjoint scene carries the forward one. Measured before this guard:
      gradient rel 1.24 at cosine 0.35 (200 fs) and rel 1.54 at 0.37 (500 fs), with
      the forward value exact. A zero wave vector is plain periodicity and is fine.
    * A nine-component (fully anisotropic) ``inv_permittivities`` or
      ``inv_permeabilities``. The adjoint current injects ``inv[axis]``, which is the
      diagonal only for the one- and three-component layouts (for nine it picks
      xx, xy, xz), and the gradient kernel contracts against a diagonal tensor.

    Raises:
        NotImplementedError: naming the boundary or the array.
    """
    bloch = [
        f"{b.name!r} (bloch_vector={tuple(b.bloch_vector)})"
        for b in objects.boundary_objects
        if isinstance(b, BlochBoundary) and any(float(k) != 0.0 for k in b.bloch_vector)
    ]
    if bloch:
        raise NotImplementedError(
            f"The {role} scene has Bloch boundaries with a nonzero wave vector: {', '.join(bloch)}. "
            "The reciprocity adjoint of a Bloch-periodic scene needs the opposite wave vector, which "
            "is not implemented, so the gradient would be wrong (measured rel 1.2-1.5) with nothing "
            "raised. Use periodic boundaries (bloch_vector=0), or run_fdtd with "
            "GradientConfig(method='checkpointed')."
        )
    for label, value in (
        ("inv_permittivities", arrays.inv_permittivities),
        ("inv_permeabilities", arrays.inv_permeabilities),
        # a full conductivity tensor switches update_E/H to the coupled anisotropic
        # update, whose lossy factor is a 3x3 matrix, not the per-component divisor
        # the adjoint current is corrected by (_LossyInjection)
        ("electric_conductivity", arrays.electric_conductivity),
        ("magnetic_conductivity", arrays.magnetic_conductivity),
    ):
        if isinstance(value, jax.Array) and value.ndim > 0 and value.shape[0] == 9:
            raise NotImplementedError(
                f"The {role} scene stores {label} as a full 3x3 tensor (shape {tuple(value.shape)}). "
                "The adjoint current and the gradient kernel assume an isotropic or diagonal "
                "material (1 or 3 components), so a fully anisotropic one would be injected and "
                "differentiated wrongly. Use run_fdtd with GradientConfig(method='checkpointed')."
            )


def _reject_unapplied_sources(objects: ObjectContainer) -> None:
    """Refuse a TFSF plane source that was never applied, before it crashes the solve.

    ``place_objects`` leaves every object whose projection overlaps a Device's
    unapplied, and a TFSF source's incident fields and time offsets stay unset
    until ``apply``. The forward solve then fails deep inside the time loop with
    ``TypeError: 'Null' object is not subscriptable``.

    Raises:
        ValueError: naming the sources and the two remedies.
    """
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
            f"Source(s) {', '.join(unapplied)} were never applied: place_objects skips every object "
            "whose projection overlaps a Device's, and apply_params applies them. Use "
            "reciprocity_param_fn, which applies them once at setup, or pass the objects returned by "
            "apply_params."
        )


def _check_internal_design_detector(det, config: SimulationConfig) -> None:
    """Assert the internal design detector is in the configuration the kernel needs.

    The design detectors are built by :func:`~fdtdx.adjoint.scene.make_design_detector`,
    never by the user, so a failure here is a bug in this package rather than a
    user error. It is checked anyway because every one of these settings fails
    silently: see :data:`~fdtdx.adjoint.scene.DESIGN_DETECTOR_SETTINGS`.
    """
    # The switch is judged by its effect, the on-step count below, not by equality.
    wrong = {k: getattr(det, k) for k, v in DESIGN_DETECTOR_SETTINGS.items() if k != "switch" and getattr(det, k) != v}
    if type(det) is not PhasorDetector:
        wrong["type"] = type(det).__name__
    if det._dft_stride != 1:
        wrong["_dft_stride"] = det._dft_stride
    if det._num_time_steps_on != int(config.time_steps_total):
        wrong["_num_time_steps_on"] = det._num_time_steps_on
    if wrong:
        raise RuntimeError(f"internal design detector {det.name!r} is misconfigured: {wrong}")


def _component_row(arr: jax.Array, axis: int) -> jax.Array:
    """Row ``axis`` of a ``(1 | 3, ...)`` material array: the one row when isotropic."""
    return arr[axis] if arr.shape[0] > 1 else arr[0]


@dataclass(frozen=True)
class _LossyInjection:
    """FDTDX's lossy-update divisor on the cells of one adjoint current.

    ``update_E`` divides the whole curl update by ``1 + a``,
    ``a = courant * sigma_E * eta0 * inv_eps / 2`` (Schneider 3.12), and only
    *then* adds the sources, so a current injected in a lossy cell enters the
    discrete operator ``1 + a`` times stronger than the reciprocal partner of the
    field the monitor reads there. The cotangent is divided back by ``1 + a``
    per cell and component; ``update_H`` does the same with
    ``b = courant * sigma_H * inv_mu / (2 * eta0)``. In a lossy *design* cell the
    same factor appears on both sides of the reciprocity pairing and cancels,
    which is why only the injection cells need it. Measured before this
    correction, a lossy block around a one-cell monitor gave a pure scale error
    (rel 2.39e-01 at cosine 1.0000000000, best-fit scale exactly ``1 + a``).

    Attributes:
        index: the block's cells.
        eps_rows: per stored component, the ``inv_permittivities`` row it uses.
        electric: ``courant * sigma_E * eta0 / 2`` per component and cell (0 on H).
        magnetic: ``b`` per component and cell (0 on E); ``inv_mu`` is static.
    """

    index: tuple[slice, ...]
    eps_rows: tuple[int, ...]
    electric: np.ndarray
    magnetic: np.ndarray

    def divisor(self, inv_eps: jax.Array) -> jax.Array:
        """``1 + a`` (E) or ``1 + b`` (H), shape ``(nc, *block)``, at the live ``inv_eps``."""
        local = inv_eps[:, *self.index]
        rows = jnp.stack([local[r] for r in self.eps_rows], axis=0)
        electric = jnp.asarray(self.electric, dtype=inv_eps.dtype)
        magnetic = jnp.asarray(self.magnetic, dtype=inv_eps.dtype)
        return 1.0 + electric * rows + magnetic


def _lossy_injection(
    arrays: ArrayContainer,
    courant: float,
    block: Sequence[tuple[int, int]],
    components: Sequence[str],
) -> _LossyInjection | None:
    """The lossy-update divisor on ``block``, or ``None`` where every cell is lossless."""
    sigma_e = arrays.electric_conductivity
    sigma_h = arrays.magnetic_conductivity
    if sigma_e is None and sigma_h is None:
        return None
    index = tuple(slice(int(lo), int(hi)) for lo, hi in block)
    shape = tuple(int(hi) - int(lo) for lo, hi in block)
    electric = np.zeros((len(components), *shape))
    magnetic = np.zeros((len(components), *shape))
    n_eps = int(arrays.inv_permittivities.shape[0])
    inv_mu = arrays.inv_permeabilities
    eps_rows = []
    for k, comp in enumerate(components):
        family, axis = FAMILY_AXIS[comp]
        eps_rows.append(axis if n_eps > 1 else 0)
        if family == 0 and sigma_e is not None:
            sigma = np.asarray(jax.device_get(_component_row(sigma_e, axis)[index]), dtype=np.float64)
            electric[k] = courant * float(eta0) * sigma / 2.0
        elif family == 1 and sigma_h is not None:
            sigma = np.asarray(jax.device_get(_component_row(sigma_h, axis)[index]), dtype=np.float64)
            if isinstance(inv_mu, jax.Array) and inv_mu.ndim > 0:
                mu = np.asarray(jax.device_get(_component_row(inv_mu, axis)[index]), dtype=np.float64)
            else:
                mu = float(inv_mu)
            magnetic[k] = courant * sigma * mu / (2.0 * float(eta0))
    if not electric.any() and not magnetic.any():
        return None
    return _LossyInjection(index=index, eps_rows=tuple(eps_rows), electric=electric, magnetic=magnetic)


class ConvergenceWarning(UserWarning):
    """The phasors a reciprocity gradient is built from have not converged.

    Reciprocity is a frequency-domain identity: it equals the exact discrete
    adjoint only once the forward and adjoint DFTs have converged, i.e. once the
    fields have left the domain. See :func:`dft_tail`.
    """


def dft_tail(
    fields: jax.Array,
    phasors: jax.Array,
    angular_frequencies: Sequence[float],
    dt: float,
    rel_floor: float = 1e-3,
) -> jax.Array:
    """Estimated DFT truncation error of ``phasors``, relative to their size.

    ``fields`` are the fields left on the phasors' cells at the last step,
    ``(nc, *cells)``, and ``phasors`` the DFT they were accumulated into,
    ``(nf, nc, *cells)``, at raw (every-step, unit-weight) scale. For each
    frequency the estimate is

        eta(w) = ||fields|| / (|1 - exp(i w dt)| * ||phasors(w)||),

    the size of the geometric tail ``sum_{n >= T} exp(i w n dt) E_n`` the DFT is
    missing if the remaining field neither decayed nor oscillated, relative to
    the recorded phasor. It costs two norms of arrays already in memory, no
    extra state. It is an estimate, not a bound: a field still ringing at ``w``
    leaves a longer tail than this, a decaying one a shorter one.

    Frequencies whose phasor is below ``rel_floor`` times the largest are left
    out: an adjoint excitation with no cotangent at some frequency has no phasor
    there to converge.

    Returns:
        The largest ``eta(w)`` over the kept frequencies (0 where no field is left).
    """
    w = jnp.asarray(np.asarray(angular_frequencies, dtype=np.float64))
    per_freq = jnp.sqrt(jnp.sum(jnp.abs(phasors.reshape(phasors.shape[0], -1)) ** 2, axis=1))
    left = jnp.sqrt(jnp.sum(jnp.abs(fields) ** 2))
    gap = jnp.abs(1.0 - jnp.exp(1j * w * dt)).astype(per_freq.dtype)
    kept = per_freq >= rel_floor * jnp.max(per_freq)
    tiny = jnp.finfo(per_freq.dtype).tiny
    eta = jnp.where(kept, left / (gap * jnp.maximum(per_freq, tiny)), 0.0)
    return jnp.max(eta)


def uncovered_device_cells(
    objects: ObjectContainer,
    region_slices: Sequence[Sequence[tuple[int, int]]],
) -> list[tuple[str, int, int]]:
    """Devices with cells outside every design region.

    The reciprocity gradient is zero outside the design regions by construction,
    so a Device cell no region covers gets a zero gradient instead of its true
    one, and every parameter ``apply_params`` maps onto it loses that part.

    Returns:
        ``[(device_name, uncovered_cells, device_cells), ...]`` for each Device
        with at least one uncovered cell.
    """
    out = []
    for dev in objects.devices:
        lo = [int(a) for a, _ in dev.grid_slice_tuple]
        hi = [int(b) for _, b in dev.grid_slice_tuple]
        covered = np.zeros(tuple(h - lo_ for lo_, h in zip(lo, hi)), dtype=bool)
        for region in region_slices:
            box = []
            for axis, (r_lo, r_hi) in enumerate(region):
                a, b = max(int(r_lo), lo[axis]), min(int(r_hi), hi[axis])
                if a >= b:
                    break
                box.append(slice(a - lo[axis], b - lo[axis]))
            else:
                covered[tuple(box)] = True
        missing = int(covered.size - covered.sum())
        if missing:
            out.append((str(dev.name), missing, int(covered.size)))
    return out


def describe_uncovered(uncovered: Sequence[tuple[str, int, int]], design: Any) -> str:
    """One sentence naming the Devices :func:`uncovered_device_cells` found."""
    parts = ", ".join(f"{name!r} ({missing} of {total} cells)" for name, missing, total in uncovered)
    return f"The design region {design!r} does not cover every Device: {parts} lie outside it."


def make_reciprocity_phasor_fn(
    forward_arrays: ArrayContainer,
    forward_objects: ObjectContainer,
    adjoint_arrays: ArrayContainer,
    adjoint_objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    objective_detectors: str | Sequence[str],
    design_detector: str | Sequence[str] | None,
    adjoint_sources: Sequence[Sequence[str]],
    window: jax.Array | None = None,
    cond_limit: float = DEFAULT_COND_LIMIT,
    tail_tolerance: float | None = DEFAULT_TAIL_TOLERANCE,
) -> ReciprocityPhasorFn:
    """Build a differentiable phasor function backed by a reciprocity gradient.

    Several objective monitors cost **one** adjoint solve, not one each: their
    adjoint currents are injected together and the design field they produce is
    their superposition, which is exactly the sum of their gradient
    contributions. The same mechanism carries a near-to-far box, which is one
    detector storing one phasor array per face and therefore needing one adjoint
    current per face.

    Neither scene needs a design-region detector. Both solves run on private
    copies of the containers (:func:`~fdtdx.adjoint.scene.internal_scene`) that
    drop every detector whose state is not needed and add one internally
    configured design detector per design region.

    Args:
        forward_arrays: placed arrays for the forward scene.
        forward_objects: placed objects, carrying the real source and the
            objective monitors.
        adjoint_arrays: placed arrays for the adjoint scene.
        adjoint_objects: placed objects, carrying the adjoint sources and no real
            source. Build it with
            :func:`~fdtdx.adjoint.scene.derive_adjoint_objects`.
        config: shared resolved config; ``gradient_config`` should be ``None``.
        key: PRNG key for both solves.
        objective_detectors: monitor name, or a sequence of them. All must share
            frequencies, since they share one amplitude-solve matrix.
        design_detector: where the gradient is taken. ``None`` means every
            ``Device`` in the scene; otherwise the name, or names, of placed
            objects whose cells form the design region (a ``Device``, a detector
            of any configuration, or a static material block). The object must
            exist in both containers.
        adjoint_sources: for each objective detector, the source names for its
            channels, ordered as :func:`~fdtdx.adjoint.scene.detector_channels`
            returns them and, within a channel, by the blocks of
            :func:`~fdtdx.adjoint.recording.channel_recordings`. Each source must
            cover exactly its block; for a raw-field monitor that is the
            monitor's own cells.
        window: adjoint excitation envelope; defaults to
            :func:`gaussian_window` over the full run.
        cond_limit: conditioning ceiling for the amplitude solve, checked here so
            a bad window fails at setup rather than inside the VJP. The default,
            :data:`~fdtdx.adjoint.reciprocity.DEFAULT_COND_LIMIT`, keeps the
            float32 solve error near 1e-4 (see there for the measurements).
        tail_tolerance: warn (:class:`ConvergenceWarning`, once per function)
            when the DFT truncation estimate :func:`dft_tail` of the objective
            phasors, or of the forward or adjoint design phasors, exceeds this.
            ``None`` disables the check. The estimates of the latest call are in
            ``phasor_fn.diagnostics`` either way.

    Returns:
        ``phasor_fn(inv_permittivities)``. For one plain monitor it returns that
        monitor's phasor array. Otherwise it returns a tuple in the given order,
        whose entries are a phasor array for a plain monitor and the full state
        dict for a box projection detector. Feed a box dict straight to the
        detector's own projection, which is pure JAX, and the gradient follows.
        ``phasor_fn.diagnostics`` is a dict with the amplitude-solve ``cond``
        and, filled in by every call (a host callback, so also under ``jit``),
        ``objective_tail`` and ``forward_design_tail`` (forward solve) and
        ``adjoint_design_tail`` (backward solve): name -> :func:`dft_tail`.

    Raises:
        NotImplementedError: for an unsupported objective configuration, or a
            frozen source overlapping a design region.
        ValueError: on missing objects, mismatched names or frequencies, or an
            ill-conditioned amplitude solve.

    Warns:
        UserWarning: when an explicitly named design region leaves cells of a
            ``Device`` uncovered; the gradient there is zero, not the true one.
        ConvergenceWarning: see ``tail_tolerance``.
    """
    single = isinstance(objective_detectors, str)
    det_names = (objective_detectors,) if single else tuple(objective_detectors)
    src_groups = tuple(tuple(g) for g in adjoint_sources)
    if len(det_names) != len(src_groups):
        raise ValueError(f"{len(det_names)} objective detector(s) but {len(src_groups)} adjoint source group(s)")
    if not det_names:
        raise ValueError("at least one objective detector is required")

    obj_dets = [_find(forward_objects.detectors, n, "detector") for n in det_names]
    for name, det in zip(det_names, obj_dets):
        _validate_objective_detector(det, f"objective {name!r}")
    _reject_bloch_and_full_tensors(forward_objects, forward_arrays, "forward")
    _reject_bloch_and_full_tensors(adjoint_objects, adjoint_arrays, "adjoint")
    _reject_unapplied_sources(forward_objects)

    # Private copies of both scenes: objective monitors kept in the forward one
    # only, since the adjoint run is read at the design region alone, and one
    # internal design detector per design region added to each.
    wave_characters = tuple(obj_dets[0].wave_characters)
    fwd_objects, fwd_arrays, des_names = internal_scene(
        forward_objects,
        forward_arrays,
        config,
        key,
        keep_detectors=det_names,
        design=design_detector,
        wave_characters=wave_characters,
    )
    adj_objects, adj_arrays, des_names_a = internal_scene(
        adjoint_objects,
        adjoint_arrays,
        config,
        key,
        keep_detectors=(),
        design=design_detector,
        wave_characters=wave_characters,
    )
    if des_names != des_names_a:
        raise ValueError(f"design regions differ between scenes: forward {des_names} vs adjoint {des_names_a}")
    des_dets = [_find(fwd_objects.detectors, n, "detector") for n in des_names]
    des_dets_a = [_find(adj_objects.detectors, n, "detector") for n in des_names]
    for d_f, d_a in zip(des_dets, des_dets_a):
        _check_internal_design_detector(d_f, config)
        _check_internal_design_detector(d_a, config)
        if d_f.grid_slice_tuple != d_a.grid_slice_tuple:
            raise ValueError(
                f"design region {d_f.name!r} differs between scenes: forward {d_f.grid_slice_tuple} vs "
                f"adjoint {d_a.grid_slice_tuple}"
            )
    if design_detector is not None:
        uncovered = uncovered_device_cells(forward_objects, [d.grid_slice_tuple for d in des_dets])
        if uncovered:
            warnings.warn(
                describe_uncovered(uncovered, design_detector)
                + " The returned gradient is exactly zero on those cells, not the true gradient, so a "
                "parameter gradient through apply_params is silently truncated (measured: rel 8.0e-01 "
                "for a region one cell short of the Device, 7.0e-01 with a second Device left out). "
                "Leave design_detector=None to use every Device.",
                UserWarning,
                stacklevel=3,
            )

    omegas = tuple(float(w) for w in obj_dets[0]._angular_frequencies)
    dt = float(config.time_step_duration)
    courant = float(config.courant_number)
    T = int(config.time_steps_total)

    # Flatten every objective detector into channels (one per stored phasor array),
    # each with the transpose of what it records and one adjoint current per block
    # of that transpose's support, and FDTDX's lossy divisor on each block.
    # (detector, recording, source indices, lossy divisor per block)
    channels: list[tuple[int, ChannelRecording, tuple[int, ...], tuple[_LossyInjection | None, ...]]] = []
    returns_dict: list[bool] = []
    for d_i, (name, det, group) in enumerate(zip(det_names, obj_dets, src_groups)):
        chans = detector_channels(det, name)
        # The adjoint current must follow the order the phasors are STORED in,
        # which is canonical whatever order `components` was declared in.
        stored = canonical_components(det)
        recordings = channel_recordings(det, chans, stored, objects=forward_objects, config=config)
        needed = sum(len(rec.blocks) for rec in recordings)
        if needed != len(group):
            raise ValueError(f"detector {name!r} needs {needed} adjoint source(s) but {len(group)} given")
        det_omegas = tuple(float(w) for w in det._angular_frequencies)
        # Tolerance, not equality: a float32 run stores frequencies in float32, so
        # a round-tripped value no longer equals the Python float it came from.
        if len(det_omegas) != len(omegas) or not np.allclose(
            np.asarray(det_omegas), np.asarray(omegas), rtol=1e-6, atol=0.0
        ):
            raise ValueError(
                f"detector {name!r} has frequencies {det_omegas}, expected {omegas}; all objective "
                "monitors must share frequencies because they share one amplitude solve"
            )
        names_iter = iter(group)
        for rec in recordings:
            indices = []
            losses = []
            for block in rec.blocks:
                src_name = next(names_iter)
                src = _find(adj_objects.sources, src_name, "source")
                if not isinstance(src, AdjointCurrentSource):
                    raise ValueError(f"{src_name!r} is a {type(src).__name__}, not an AdjointCurrentSource")
                if tuple(src.components) != stored:
                    raise ValueError(
                        f"adjoint source {src_name!r} components {tuple(src.components)} must match the "
                        f"order detector {name!r} stores its phasors in, {stored}"
                    )
                # A misplaced current is silent, so the placement is checked, not assumed.
                if tuple(tuple(int(v) for v in ax) for ax in src.grid_slice_tuple) != block:
                    raise ValueError(
                        f"adjoint source {src_name!r} covers {src.grid_slice_tuple}, but channel "
                        f"{rec.state_key!r} of detector {name!r} reads the fields on {block}"
                        + (" (exact_interpolation widens it by the co-location stencil)" if rec.exact else "")
                    )
                indices.append(
                    next(i for i, o in enumerate(adj_objects.object_list) if getattr(o, "name", None) == src_name)
                )
                # the arrays the adjoint solve runs on, whose conductivity the
                # current is injected into
                losses.append(_lossy_injection(adj_arrays, courant, block, stored))
            channels.append((d_i, rec, tuple(indices), tuple(losses)))
        # more than one state key means the caller receives the whole dict and
        # applies the detector's own readout to it
        returns_dict.append(len(chans) > 1)

    # A stock source overlapping the design region would make this kernel disagree
    # with run_fdtd; refuse rather than silently pick a convention.
    all_src_names = tuple(n for g in src_groups for n in g)
    for des_det in des_dets:
        _reject_frozen_sources_in_design(forward_objects, des_det.grid_slice_tuple)
        _reject_frozen_sources_in_design(adj_objects, des_det.grid_slice_tuple, skip=all_src_names)

    win = gaussian_window(T, dtype=config.dtype) if window is None else window
    # Precompute the amplitude-solve matrix on concrete values so the solve itself
    # is a plain jnp.linalg.solve and works under trace inside the VJP.
    A_np = _design_matrix(np.asarray(omegas, dtype=np.float64), dt, np.asarray(jax.device_get(win)))
    cond = float(np.linalg.cond(A_np))
    if not np.isfinite(cond) or cond > cond_limit:
        raise ValueError(
            f"Adjoint amplitude solve is ill-conditioned (cond={cond:.3e} > {cond_limit:.1e}). "
            "The window is too short to separate the objective frequencies: the adjoint current "
            "then cancels itself out and the gradient is garbage (measured on the colour splitter: "
            "rel 20 at cond 7.2e7, 0.78 at 3.5e4). Lengthen the simulation, widen the window's "
            "sigma_frac, or space the objective frequencies further apart."
        )
    A = jnp.asarray(A_np)
    nf = len(omegas)
    diagnostics: dict[str, Any] = {
        "cond": cond,
        "tail_tolerance": tail_tolerance,
        "objective_tail": {},
        "forward_design_tail": {},
        "adjoint_design_tail": {},
    }

    # PhasorDetector multiplies every recorded sample by _static_scale(): 1 in
    # "pulse" mode, 2/sum(window) in "continuous", which is its default. The kernel
    # is calibrated on RAW phasors, so each scale is divided back out:
    #
    # * objective monitor, P = s_m * P_raw. JAX hands the VJP dL/dP, but the adjoint
    #   current must reproduce dL/dP_raw = s_m * dL/dP. That is a per-DETECTOR
    #   factor on the amplitude target: monitors in different modes each need their
    #   own, so it cannot be one global factor on the gradient.
    # * design region, F = s_d * F_raw in the forward run and Lambda = s_d * Lambda_raw
    #   in the adjoint run, so the assembled gradient carries s_d twice.
    #
    # Measured on a 24^3 float64 scene, uncorrected best-fit scale against run_fdtd:
    # 6.19e+05, 1.27e-03 and 7.87e+02 for the three non-pulse combinations. The
    # internal design detectors are pulse, so s_d == 1 today; it is still divided
    # out, from the placed detectors themselves, so the two stay consistent.
    #
    # A stride s records every s-th step, and s * (that sum) estimates the every-step
    # sum, so a strided monitor's P is (_static_scale() / s) * P_raw: pulse mode's
    # scale IS the stride, and continuous mode's 2/sum(window) counts kept steps only.
    s_m_per_det = [float(d._static_scale()) / int(d._dft_stride) for d in obj_dets]
    s_d_sq = [float(f._static_scale()) * float(a._static_scale()) for f, a in zip(des_dets, des_dets_a)]

    # Magnetic components carry an extra factor, for two independent reasons.
    #
    # Sign: Lorentz reciprocity is not symmetric between the electric and magnetic
    # pairings, int(E_a . J_b - H_a . M_b) = int(E_b . J_a - H_b . M_a), so a
    # cotangent on a magnetic component maps to an adjoint magnetic current of the
    # opposite sign. Without the flip the gradient points backwards: cosine -0.998
    # against checkpointed autodiff on an Hx objective.
    #
    # Half-step: with exact_interpolation=False the detector stores the
    # post-update H, which lives at n+1/2, but weights it with the integer-step
    # kernel exp(+i w n dt), while the adjoint magnetic current is injected at
    # time_step + 0.5 (fdtd/update.py:813). Compensating costs exp(-i w dt / 2).
    #
    # Measured on an Hx objective: plain sign flip 1.04e-01, with exp(+i w dt/2)
    # 2.26e-01, with exp(-i w dt/2) 6.3e-07 at cosine 1.000000000.
    #
    # Time average: with exact_interpolation=True the detector records
    # (H_prev + H) / 2 = (H^{n-1/2} + H^{n+1/2}) / 2 at step n, whose DFT is
    # (1 + exp(+i w dt)) / 2 times the post-update one. Composed with the half-step
    # factor this is -cos(w dt / 2), but it is kept as the product of both named
    # terms: dropping the average costs rel 8.5e-02 at cosine 0.996, and its
    # conjugate rel 1.8e-01, both silent.
    w_np = np.asarray(omegas, dtype=np.float64)
    half_step = -np.exp(-1j * w_np * dt / 2.0)
    time_average = (1.0 + np.exp(1j * w_np * dt)) / 2.0
    signs: list[jax.Array] = []
    for d_i, rec, _indices, _losses in channels:
        det = obj_dets[d_i]
        magnetic_factor = half_step * time_average if rec.exact else half_step
        is_magnetic = np.asarray([c.startswith("H") for c in canonical_components(det)])
        s = np.where(is_magnetic[None, :], magnetic_factor[:, None], 1.0 + 0.0j) * s_m_per_det[d_i]
        ndim_spatial = len(det.grid_shape)
        signs.append(jnp.asarray(s).reshape(s.shape + (1,) * ndim_spatial))

    des_slices = [d.grid_slice for d in des_dets]

    # Convergence diagnostic: dft_tail of every objective channel (raw scale) and of
    # the design phasors in both solves, from the fields each solve leaves at its
    # last step. Reported through a host callback so it also works under jit.
    channel_labels = [
        det_names[d_i] if rec.state_key == "phasor" else f"{det_names[d_i]}[{rec.state_key}]"
        for d_i, rec, _indices, _losses in channels
    ]
    warned: set[str] = set()

    def _report(stage: str, labels: tuple[str, ...], values) -> None:
        tails = {label: float(v) for label, v in zip(labels, values)}
        diagnostics[stage] = tails
        if tail_tolerance is None or stage in warned:
            return
        worst = max(tails, key=lambda k: tails[k])
        if tails[worst] > tail_tolerance:
            warned.add(stage)
            what, meaning = {
                "objective_tail": (
                    "objective phasors",
                    "The returned phasors, and a figure of merit built on them, carry a truncation "
                    "error of about that size.",
                ),
                "forward_design_tail": (
                    "forward design-region phasors",
                    "Reciprocity equals the exact gradient only once both solves have converged; on "
                    "the colour splitter the gradient error was 2-5x the larger design-region estimate.",
                ),
                "adjoint_design_tail": (
                    "adjoint design-region phasors",
                    "Reciprocity equals the exact gradient only once both solves have converged; on "
                    "the colour splitter the gradient error was 2-5x the larger design-region estimate "
                    "(1.46 at 22 fs, where it was rel 20).",
                ),
            }[stage]
            warnings.warn(
                f"The {what} have not converged: DFT tail estimate {tails[worst]:.2e} for {worst!r} "
                f"exceeds tail_tolerance={tail_tolerance:.1e} (all: "
                + ", ".join(f"{k}={v:.2e}" for k, v in tails.items())
                + f"). {meaning} Lengthen the simulation so the fields leave the domain (see dft_tail); "
                "pass tail_tolerance=None to silence this.",
                ConvergenceWarning,
                stacklevel=2,
            )

    def _stored_fields(fields, components: tuple[str, ...], cells) -> jax.Array:
        E, H = fields.E, fields.H
        index = tuple(slice(int(lo), int(hi)) for lo, hi in cells)
        return jnp.stack([(E if FAMILY_AXIS[c][0] == 0 else H)[FAMILY_AXIS[c][1]][index] for c in components])

    def _forward_tails(out):
        objective = []
        for d_i, rec, _indices, _losses in channels:
            det = obj_dets[d_i]
            phasors = out.detector_states[det_names[d_i]][rec.state_key][0] / s_m_per_det[d_i]
            left = _stored_fields(out.fields, canonical_components(det), rec.channel_slice)
            objective.append(dft_tail(left, phasors, omegas, dt))
        design = [
            dft_tail(out.fields.E[:, *sl], out.detector_states[n]["phasor"][0], omegas, dt)
            for n, sl in zip(des_names, des_slices)
        ]
        jax.debug.callback(partial(_report, "objective_tail", tuple(channel_labels)), tuple(objective))
        jax.debug.callback(partial(_report, "forward_design_tail", tuple(des_names)), tuple(design))

    def _solve_amplitudes(target: jax.Array, sign: jax.Array) -> jax.Array:
        """Trace-safe solve_adjoint_amplitudes with the matrix precomputed."""
        target = target * sign
        tail = target.shape[1:]
        flat = target.reshape(nf, -1)
        rhs = jnp.concatenate([jnp.real(flat), jnp.imag(flat)], axis=0)
        sol = jnp.linalg.solve(A, rhs)
        return (sol[:nf] + 1j * sol[nf:]).reshape(nf, *tail)

    def _forward(inv_eps: jax.Array):
        arrays = fwd_arrays.aset("inv_permittivities", inv_eps)
        _, out = checkpointed_fdtd(arrays, fwd_objects, config, key, show_progress=False)
        _forward_tails(out)
        outs = []
        for name, wants_dict in zip(det_names, returns_dict):
            state = out.detector_states[name]
            outs.append(dict(state) if wants_dict else state["phasor"])
        return tuple(outs), tuple(out.detector_states[n]["phasor"] for n in des_names)

    @jax.custom_vjp
    def phasor_fn(inv_eps: jax.Array):
        return _forward(inv_eps)[0]

    def phasor_fwd(inv_eps: jax.Array):
        outs, F = _forward(inv_eps)
        return outs, (inv_eps, F)

    def phasor_bwd(res, ct):
        inv_eps, F = res
        # Build the adjoint container HERE, not in a closure: closing over a
        # traced source leaf is what raises UnexpectedTracerError.
        new_list = list(adj_objects.object_list)
        for (d_i, rec, indices, losses), sign in zip(channels, signs):
            ct_det = ct[d_i]
            ct_one = ct_det[rec.state_key] if isinstance(ct_det, dict) else ct_det
            # Cotangent on the recorded values -> cotangent on the raw Yee fields
            # the adjoint current drives: the identity for a raw-field monitor, the
            # transposed co-location stencil (and padding) for an exact one.
            for idx, target, loss in zip(indices, rec.transpose(ct_one[0]), losses):
                if loss is not None:
                    # FDTDX adds sources after the lossy division (_LossyInjection)
                    target = target / loss.divisor(inv_eps)[None]
                new_list[idx] = new_list[idx].aset("amplitudes", _solve_amplitudes(target, sign))
        objects_a = adj_objects.aset("object_list", new_list)
        arrays_a = adj_arrays.aset("inv_permittivities", inv_eps)
        # One solve for every channel: the adjoint currents superpose, and so do
        # their contributions to the design gradient.
        _, out_a = checkpointed_fdtd(arrays_a, objects_a, config, key, show_progress=False)
        adjoint_tails = [
            dft_tail(out_a.fields.E[:, *sl], out_a.detector_states[n]["phasor"][0], omegas, dt)
            for n, sl in zip(des_names, des_slices)
        ]
        jax.debug.callback(partial(_report, "adjoint_design_tail", tuple(des_names)), tuple(adjoint_tails))

        grad = jnp.zeros_like(inv_eps)
        for name, F_i, sl, scale_sq in zip(des_names, F, des_slices, s_d_sq):
            g_design = assemble_material_gradient(
                adjoint_phasors=out_a.detector_states[name]["phasor"],
                forward_phasors=F_i,
                inv_permittivities=inv_eps[:, *sl],
                angular_frequencies=omegas,
                dt=dt,
                courant_number=courant,
            )
            # divided by a Python float so a float32 run keeps full precision
            g_design = g_design / scale_sq
            # set, not add: where two regions overlap both compute the same
            # value from the same fields, and adding would double it
            grad = grad.at[:, *sl].set(g_design.astype(inv_eps.dtype))
        return (grad,)

    phasor_fn.defvjp(phasor_fwd, phasor_bwd)
    return ReciprocityPhasorFn(phasor_fn, single=single, diagnostics=diagnostics)


class ReciprocityPhasorFn:
    """``phasor_fn(inv_permittivities) -> phasors``, from :func:`make_reciprocity_phasor_fn`.

    Attributes:
        diagnostics: ``cond`` of the amplitude solve and the :func:`dft_tail`
            estimates of the latest forward (``objective_tail``,
            ``forward_design_tail``) and backward (``adjoint_design_tail``) solve,
            each a dict name -> value. Updated by every call, including under ``jit``.
    """

    def __init__(self, fn: Callable[[jax.Array], tuple[Any, ...]], *, single: bool, diagnostics: dict[str, Any]):
        self._fn = fn
        self._single = single
        self.diagnostics = diagnostics

    def __call__(self, inv_permittivities: jax.Array) -> Any:
        out = self._fn(inv_permittivities)
        return out[0] if self._single else out
