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

from collections.abc import Sequence
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.adjoint.reciprocity import _design_matrix, assemble_material_gradient, gaussian_window
from fdtdx.adjoint.recording import ChannelRecording, channel_recordings
from fdtdx.adjoint.scene import (
    DESIGN_DETECTOR_SETTINGS,
    canonical_components,
    detector_channels,
    internal_scene,
    is_box_projection,
)
from fdtdx.config import SimulationConfig
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
    cond_limit: float = 1e8,
) -> Callable[[jax.Array], Any]:
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
            a bad window fails at setup rather than inside the VJP.

    Returns:
        ``phasor_fn(inv_permittivities)``. For one plain monitor it returns that
        monitor's phasor array. Otherwise it returns a tuple in the given order,
        whose entries are a phasor array for a plain monitor and the full state
        dict for a box projection detector. Feed a box dict straight to the
        detector's own projection, which is pure JAX, and the gradient follows.

    Raises:
        NotImplementedError: for an unsupported objective configuration, or a
            frozen source overlapping a design region.
        ValueError: on missing objects, mismatched names or frequencies, or an
            ill-conditioned amplitude solve.
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

    omegas = tuple(float(w) for w in obj_dets[0]._angular_frequencies)
    dt = float(config.time_step_duration)
    courant = float(config.courant_number)
    T = int(config.time_steps_total)

    # Flatten every objective detector into channels (one per stored phasor array),
    # each with the transpose of what it records and one adjoint current per block
    # of that transpose's support.
    channels: list[tuple[int, ChannelRecording, tuple[int, ...]]] = []  # (detector, recording, source indices)
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
            channels.append((d_i, rec, tuple(indices)))
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
            "Widen the window's sigma_frac, lengthen the simulation, or space the objective "
            "frequencies further apart."
        )
    A = jnp.asarray(A_np)
    nf = len(omegas)

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
    for d_i, rec, _indices in channels:
        det = obj_dets[d_i]
        magnetic_factor = half_step * time_average if rec.exact else half_step
        is_magnetic = np.asarray([c.startswith("H") for c in canonical_components(det)])
        s = np.where(is_magnetic[None, :], magnetic_factor[:, None], 1.0 + 0.0j) * s_m_per_det[d_i]
        ndim_spatial = len(det.grid_shape)
        signs.append(jnp.asarray(s).reshape(s.shape + (1,) * ndim_spatial))

    des_slices = [d.grid_slice for d in des_dets]

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
        for (d_i, rec, indices), sign in zip(channels, signs):
            ct_det = ct[d_i]
            ct_one = ct_det[rec.state_key] if isinstance(ct_det, dict) else ct_det
            # Cotangent on the recorded values -> cotangent on the raw Yee fields
            # the adjoint current drives: the identity for a raw-field monitor, the
            # transposed co-location stencil (and padding) for an exact one.
            for idx, target in zip(indices, rec.transpose(ct_one[0])):
                new_list[idx] = new_list[idx].aset("amplitudes", _solve_amplitudes(target, sign))
        objects_a = adj_objects.aset("object_list", new_list)
        arrays_a = adj_arrays.aset("inv_permittivities", inv_eps)
        # One solve for every channel: the adjoint currents superpose, and so do
        # their contributions to the design gradient.
        _, out_a = checkpointed_fdtd(arrays_a, objects_a, config, key, show_progress=False)

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

    if single and not returns_dict[0]:

        def single_fn(inv_eps: jax.Array) -> jax.Array:
            return phasor_fn(inv_eps)[0]

        return single_fn
    if single:

        def single_dict_fn(inv_eps: jax.Array):
            return phasor_fn(inv_eps)[0]

        return single_dict_fn
    return phasor_fn
