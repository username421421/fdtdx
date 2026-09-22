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

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.adjoint.reciprocity import _design_matrix, assemble_material_gradient, gaussian_window
from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import checkpointed_fdtd
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.sources.adjoint import AdjointCurrentSource


def _require_plain_phasor_detector(det, role: str) -> None:
    if type(det) is not PhasorDetector:
        raise NotImplementedError(
            f"The {role} detector must be exactly a PhasorDetector, got {type(det).__name__}. "
            "Subclasses that post-process (mode overlap, field projection, diffraction orders) "
            "keep per-face state and bypass PhasorDetector.update, so the reciprocity transpose "
            "does not apply to them directly. Record raw phasors with a PhasorDetector and do the "
            "post-processing in JAX on top of this function instead -- that composes and is the "
            "whole point of putting the VJP boundary at the raw phasors."
        )
    if det._dft_stride != 1:
        raise NotImplementedError(
            f"The {role} detector has dft_subsample resolving to stride {det._dft_stride}; the "
            "reciprocity path requires 1. The exact adjoint of a decimated accumulator injects "
            "over the same decimated step set, which is not implemented yet."
        )
    if det.reduce_volume:
        raise NotImplementedError(f"The {role} detector must have reduce_volume=False.")
    if det.exact_interpolation:
        raise NotImplementedError(
            f"The {role} detector must have exact_interpolation=False. The co-location stencil "
            "would have to be transposed back through the Yee grid, which is not implemented yet."
        )


def _slices_overlap(a, b) -> bool:
    """Do two ``((lo, hi), (lo, hi), (lo, hi))`` grid slice tuples intersect?

    All three axes must overlap. ``SimulationObject.check_overlap`` is not used
    here: it reports True when the projections overlap on any single axis, which
    calls a source plane sitting above a design block an overlap.
    """
    return all(lo_a < hi_b and lo_b < hi_a for (lo_a, hi_a), (lo_b, hi_b) in zip(a, b))


def _reject_frozen_sources_in_design(objects, design_slice_tuple, skip=()):
    """Refuse a source that overlaps the design region but freezes its inv_eps.

    FDTDX injects an impressed current as ``E += -courant * inv_eps * J``, so a
    source inside the design region contributes a term to the design gradient.
    Every stock source suppresses that term:
    :class:`~fdtdx.objects.sources.dipole.PointDipoleSource` caches the factor in
    a private field during ``apply()`` (``dipole.py``), and the TFSF plane
    sources wrap their injection in ``jax.lax.stop_gradient`` (``tfsf.py``).

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


def make_reciprocity_phasor_fn(
    forward_arrays: ArrayContainer,
    forward_objects: ObjectContainer,
    adjoint_arrays: ArrayContainer,
    adjoint_objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    objective_detector: str,
    design_detector: str,
    adjoint_source: str,
    window: jax.Array | None = None,
    cond_limit: float = 1e8,
) -> Callable[[jax.Array], jax.Array]:
    """Build a differentiable phasor function backed by a reciprocity gradient.

    Args:
        forward_arrays: placed arrays for the forward scene.
        forward_objects: placed objects for the forward scene, carrying the real
            source, the objective detector and the design detector.
        adjoint_arrays: placed arrays for the adjoint scene.
        adjoint_objects: placed objects for the adjoint scene. Must carry an
            :class:`AdjointCurrentSource` named ``adjoint_source``, placed on the
            objective detector's cells, plus a design detector of the same name
            and shape as the forward one. It must NOT carry the real source.
        config: shared simulation config. ``gradient_config`` should be ``None``:
            both runs are plain forward solves.
        key: PRNG key passed to both runs.
        objective_detector: name of the monitor the FoM reads.
        design_detector: name of the detector covering the design region.
        adjoint_source: name of the :class:`AdjointCurrentSource`.
        window: adjoint excitation envelope. Defaults to
            :func:`gaussian_window` over the full run.
        cond_limit: conditioning ceiling for the amplitude solve, checked once
            here rather than inside the VJP so a bad window fails loudly at
            setup.

    Returns:
        ``phasor_fn(inv_permittivities) -> objective phasors``, differentiable.

    Raises:
        NotImplementedError: if either detector falls outside the supported
            configuration.
        ValueError: on a missing object, a mismatched design detector, or an
            ill-conditioned amplitude solve.
    """
    obj_det = _find(forward_objects.detectors, objective_detector, "detector")
    des_det = _find(forward_objects.detectors, design_detector, "detector")
    des_det_a = _find(adjoint_objects.detectors, design_detector, "detector")
    adj_src = _find(adjoint_objects.sources, adjoint_source, "source")

    _require_plain_phasor_detector(obj_det, "objective")
    _require_plain_phasor_detector(des_det, "design")
    if not isinstance(adj_src, AdjointCurrentSource):
        raise ValueError(f"{adjoint_source!r} is a {type(adj_src).__name__}, not an AdjointCurrentSource")
    if des_det.grid_shape != des_det_a.grid_shape:
        raise ValueError(
            f"design detector shape differs between scenes: forward {des_det.grid_shape} vs "
            f"adjoint {des_det_a.grid_shape}"
        )
    if tuple(adj_src.components) != tuple(obj_det.components):
        raise ValueError(
            f"adjoint source components {tuple(adj_src.components)} must match the objective "
            f"detector's {tuple(obj_det.components)}"
        )

    omegas = tuple(float(w) for w in obj_det._angular_frequencies)
    # Compared with a tolerance rather than exactly: on a float32 run the
    # detector stores its frequencies in float32, so a value round-tripped
    # through the detector no longer equals the Python float it came from.
    if len(adj_src.angular_frequencies) != len(omegas) or not np.allclose(
        np.asarray(adj_src.angular_frequencies, dtype=np.float64),
        np.asarray(omegas, dtype=np.float64),
        rtol=1e-6,
        atol=0.0,
    ):
        raise ValueError(
            f"adjoint source frequencies {tuple(adj_src.angular_frequencies)} must match the "
            f"objective detector's {omegas}"
        )

    # A stock source overlapping the design region would make this kernel disagree
    # with run_fdtd; refuse rather than silently pick a convention.
    design_slice_tuple = des_det.grid_slice_tuple
    _reject_frozen_sources_in_design(forward_objects, design_slice_tuple)
    _reject_frozen_sources_in_design(adjoint_objects, design_slice_tuple, skip=(adjoint_source,))

    dt = float(config.time_step_duration)
    courant = float(config.courant_number)
    T = int(config.time_steps_total)

    win = gaussian_window(T) if window is None else window
    # Precompute the amplitude-solve matrix on concrete values so the solve
    # itself is a plain jnp.linalg.solve and works under trace inside the VJP.
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
    des_slice = des_det.grid_slice

    # ObjectContainer.sources is a filtered view of object_list, so the source has
    # to be swapped by its index in that list. Replacing one traced leaf leaves the
    # PyTreeDef untouched, which is what keeps the FDTD loop from recompiling.
    adj_idx = next(i for i, o in enumerate(adjoint_objects.object_list) if getattr(o, "name", None) == adjoint_source)

    def _solve_amplitudes(target: jax.Array) -> jax.Array:
        """Trace-safe version of solve_adjoint_amplitudes with A precomputed."""
        tail = target.shape[1:]
        flat = target.reshape(nf, -1)
        rhs = jnp.concatenate([jnp.real(flat), jnp.imag(flat)], axis=0)
        sol = jnp.linalg.solve(A, rhs)
        return (sol[:nf] + 1j * sol[nf:]).reshape(nf, *tail)

    def _forward(inv_eps: jax.Array) -> tuple[jax.Array, jax.Array]:
        arrays = forward_arrays.aset("inv_permittivities", inv_eps)
        _, out = checkpointed_fdtd(arrays, forward_objects, config, key, show_progress=False)
        return out.detector_states[objective_detector]["phasor"], out.detector_states[design_detector]["phasor"]

    @jax.custom_vjp
    def phasor_fn(inv_eps: jax.Array) -> jax.Array:
        return _forward(inv_eps)[0]

    def phasor_fwd(inv_eps: jax.Array):
        P, F = _forward(inv_eps)
        return P, (inv_eps, F)

    def phasor_bwd(res, ct_P: jax.Array):
        inv_eps, F = res
        # Build the adjoint container HERE, not in a closure: closing over a
        # traced source leaf is what raises UnexpectedTracerError.
        amplitudes = _solve_amplitudes(ct_P[0])
        new_list = list(adjoint_objects.object_list)
        new_list[adj_idx] = new_list[adj_idx].aset("amplitudes", amplitudes)
        objects_a = adjoint_objects.aset("object_list", new_list)
        arrays_a = adjoint_arrays.aset("inv_permittivities", inv_eps)
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
    return phasor_fn
