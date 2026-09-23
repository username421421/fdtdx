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
Both of PhasorDetector's scaling modes are accepted on the objective monitors.

Restrictions, all enforced with an exception rather than silently approximated:

* On the objective monitors: ``dft_subsample`` resolving to stride 1,
  ``reduce_volume=False``, ``exact_interpolation=False``, no apodization, and a
  switch that records every time step.
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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.adjoint.reciprocity import _design_matrix, assemble_material_gradient, gaussian_window
from fdtdx.adjoint.scene import (
    DESIGN_DETECTOR_SETTINGS,
    canonical_components,
    detector_channels,
    internal_scene,
    is_box_projection,
)
from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import checkpointed_fdtd
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.object import INVALID_SLICE_TUPLE_3D
from fdtdx.objects.sources.adjoint import AdjointCurrentSource


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
    if det._dft_stride != 1:
        raise NotImplementedError(
            f"The {role} detector has dft_subsample resolving to stride {det._dft_stride}; "
            "the reciprocity path requires 1."
        )
    if det.reduce_volume:
        raise NotImplementedError(f"The {role} detector must have reduce_volume=False.")
    if det.exact_interpolation:
        raise NotImplementedError(
            f"The {role} detector has exact_interpolation=True. That interpolation runs in "
            "update_detector_states, outside the detector's own update, so a VJP replacing the "
            "whole time loop never sees it and the co-location stencil would have to be "
            "transposed back through the Yee grid by hand. Set it off with\n"
            '    detector = detector.aset("exact_interpolation", False)\n'
            "before placing. The detector then records raw Yee fields rather than co-located "
            "ones, which changes the forward far field by a second-order discretization "
            "amount: measured 1.53e-02 relative at 12 cells per wavelength and 3.39e-03 at 24, "
            "i.e. it converges away as the grid is refined. The gradient itself is unaffected, "
            "matching run_fdtd to 6.4e-07 either way."
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
            returns them.
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

    # Flatten every objective detector into channels, one adjoint current each.
    channels: list[tuple[int, str, int]] = []  # (detector index, state key, source index in object_list)
    returns_dict: list[bool] = []
    for d_i, (name, det, group) in enumerate(zip(det_names, obj_dets, src_groups)):
        chans = detector_channels(det, name)
        if len(chans) != len(group):
            raise ValueError(f"detector {name!r} needs {len(chans)} adjoint source(s) but {len(group)} given")
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
        # The adjoint current must follow the order the phasors are STORED in,
        # which is canonical whatever order `components` was declared in.
        stored = canonical_components(det)
        for (state_key, _slice), src_name in zip(chans, group):
            src = _find(adj_objects.sources, src_name, "source")
            if not isinstance(src, AdjointCurrentSource):
                raise ValueError(f"{src_name!r} is a {type(src).__name__}, not an AdjointCurrentSource")
            if tuple(src.components) != stored:
                raise ValueError(
                    f"adjoint source {src_name!r} components {tuple(src.components)} must match the "
                    f"order detector {name!r} stores its phasors in, {stored}"
                )
            idx = next(i for i, o in enumerate(adj_objects.object_list) if getattr(o, "name", None) == src_name)
            channels.append((d_i, state_key, idx))
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
    s_m_per_det = [float(d._static_scale()) for d in obj_dets]
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
    magnetic_factor = -np.exp(-1j * np.asarray(omegas, dtype=np.float64) * dt / 2.0)
    signs: list[jax.Array] = []
    for d_i, _state_key, _idx in channels:
        det = obj_dets[d_i]
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
        for (d_i, state_key, idx), sign in zip(channels, signs):
            ct_det = ct[d_i]
            ct_one = ct_det[state_key] if isinstance(ct_det, dict) else ct_det
            amplitudes = _solve_amplitudes(ct_one[0], sign)
            new_list[idx] = new_list[idx].aset("amplitudes", amplitudes)
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
