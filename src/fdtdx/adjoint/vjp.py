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

Restrictions, all enforced with an exception rather than silently approximated:

* ``dft_subsample == 1``, ``reduce_volume=False``, ``exact_interpolation=False``
  on both detectors, and the objective detector must be exactly a
  ``PhasorDetector`` (box-mode and projection detectors keep per-face state and
  bypass ``PhasorDetector.update``).
* The design region must not contain a source. FDTDX sources scale their
  injection by the local ``inv_eps``, a term this gradient does not model.
* The returned gradient is zero outside the design detector's region. That is
  what inverse design wants, but it is not the full-grid gradient.

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
from fdtdx.adjoint.scene import detector_channels, is_box_projection
from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import checkpointed_fdtd
from fdtdx.objects.detectors.phasor import PhasorDetector
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
        if getattr(obj, "_grid_slice_tuple", None) is None:
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
    """Accept a plain phasor detector or a box-mode field projection detector."""
    if is_box_projection(det):
        if len(det.components) != 6:
            raise NotImplementedError(
                f"The {role} detector records {len(det.components)} components; box-mode field "
                "projection concatenates E and H and so needs all six "
                "(Ex, Ey, Ez, Hx, Hy, Hz)."
            )
    elif type(det) is not PhasorDetector:
        raise NotImplementedError(
            f"The {role} detector must be a PhasorDetector or a box-mode field projection "
            f"detector, got {type(det).__name__}. Other post-processing subclasses (mode "
            "overlap, diffraction orders) keep state this transpose does not cover; record raw "
            "phasors and do the post-processing in JAX on top of this function instead."
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


def make_reciprocity_phasor_fn(
    forward_arrays: ArrayContainer,
    forward_objects: ObjectContainer,
    adjoint_arrays: ArrayContainer,
    adjoint_objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    objective_detectors: str | Sequence[str],
    design_detector: str,
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

    Args:
        forward_arrays: placed arrays for the forward scene.
        forward_objects: placed objects, carrying the real source, the objective
            monitors and a detector covering the design region.
        adjoint_arrays: placed arrays for the adjoint scene.
        adjoint_objects: placed objects, carrying the adjoint sources and no real
            source. Build it with
            :func:`~fdtdx.adjoint.scene.derive_adjoint_objects`.
        config: shared resolved config; ``gradient_config`` should be ``None``.
        key: PRNG key for both solves.
        objective_detectors: monitor name, or a sequence of them. All must share
            frequencies, since they share one amplitude-solve matrix.
        design_detector: detector covering the design region.
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
        NotImplementedError: for an unsupported detector configuration, or a
            frozen source overlapping the design region.
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
    des_det = _find(forward_objects.detectors, design_detector, "detector")
    des_det_a = _find(adjoint_objects.detectors, design_detector, "detector")

    for name, det in zip(det_names, obj_dets):
        _validate_objective_detector(det, f"objective {name!r}")
    _validate_objective_detector(des_det, "design")
    if is_box_projection(des_det):
        raise NotImplementedError("the design detector must be a plain PhasorDetector")
    if des_det.grid_shape != des_det_a.grid_shape:
        raise ValueError(
            f"design detector shape differs between scenes: forward {des_det.grid_shape} vs "
            f"adjoint {des_det_a.grid_shape}"
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
        for (state_key, _slice), src_name in zip(chans, group):
            src = _find(adjoint_objects.sources, src_name, "source")
            if not isinstance(src, AdjointCurrentSource):
                raise ValueError(f"{src_name!r} is a {type(src).__name__}, not an AdjointCurrentSource")
            if tuple(src.components) != tuple(det.components):
                raise ValueError(
                    f"adjoint source {src_name!r} components {tuple(src.components)} must match "
                    f"detector {name!r}'s {tuple(det.components)}"
                )
            idx = next(i for i, o in enumerate(adjoint_objects.object_list) if getattr(o, "name", None) == src_name)
            channels.append((d_i, state_key, idx))
        returns_dict.append(is_box_projection(det))

    # A stock source overlapping the design region would make this kernel disagree
    # with run_fdtd; refuse rather than silently pick a convention.
    design_slice_tuple = des_det.grid_slice_tuple
    _reject_frozen_sources_in_design(forward_objects, design_slice_tuple)
    all_src_names = tuple(n for g in src_groups for n in g)
    _reject_frozen_sources_in_design(adjoint_objects, design_slice_tuple, skip=all_src_names)

    win = gaussian_window(T) if window is None else window
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
    for d_i, state_key, _idx in channels:
        det = obj_dets[d_i]
        is_magnetic = np.asarray([c.startswith("H") for c in det.components])
        s = np.where(is_magnetic[None, :], magnetic_factor[:, None], 1.0 + 0.0j)
        ndim_spatial = len(det.grid_shape)
        signs.append(jnp.asarray(s).reshape(s.shape + (1,) * ndim_spatial))

    des_slice = des_det.grid_slice

    def _solve_amplitudes(target: jax.Array, sign: jax.Array) -> jax.Array:
        """Trace-safe solve_adjoint_amplitudes with the matrix precomputed."""
        target = target * sign
        tail = target.shape[1:]
        flat = target.reshape(nf, -1)
        rhs = jnp.concatenate([jnp.real(flat), jnp.imag(flat)], axis=0)
        sol = jnp.linalg.solve(A, rhs)
        return (sol[:nf] + 1j * sol[nf:]).reshape(nf, *tail)

    def _forward(inv_eps: jax.Array):
        arrays = forward_arrays.aset("inv_permittivities", inv_eps)
        _, out = checkpointed_fdtd(arrays, forward_objects, config, key, show_progress=False)
        outs = []
        for name, wants_dict in zip(det_names, returns_dict):
            state = out.detector_states[name]
            outs.append(dict(state) if wants_dict else state["phasor"])
        return tuple(outs), out.detector_states[design_detector]["phasor"]

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
        new_list = list(adjoint_objects.object_list)
        for (d_i, state_key, idx), sign in zip(channels, signs):
            ct_det = ct[d_i]
            ct_one = ct_det[state_key] if isinstance(ct_det, dict) else ct_det
            amplitudes = _solve_amplitudes(ct_one[0], sign)
            new_list[idx] = new_list[idx].aset("amplitudes", amplitudes)
        objects_a = adjoint_objects.aset("object_list", new_list)
        arrays_a = adjoint_arrays.aset("inv_permittivities", inv_eps)
        # One solve for every channel: the adjoint currents superpose, and so do
        # their contributions to the design gradient.
        _, out_a = checkpointed_fdtd(arrays_a, objects_a, config, key, show_progress=False)
        lam = out_a.detector_states[design_detector]["phasor"]

        g_design = assemble_material_gradient(
            adjoint_phasors=lam,
            forward_phasors=F,
            inv_permittivities=inv_eps[:, *des_slice],
            angular_frequencies=omegas,
            dt=dt,
            courant_number=courant,
        )
        grad = jnp.zeros_like(inv_eps).at[:, *des_slice].set(g_design.astype(inv_eps.dtype))
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
