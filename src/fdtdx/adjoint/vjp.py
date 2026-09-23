"""The ``jax.custom_vjp``: a reciprocity gradient from two plain forward solves.

The forward rule runs the scene with the objective monitors and one internal design detector
per region, and returns the monitors' stored phasors. The backward rule maps the cotangent of
every stored channel to the target DFT of its adjoint currents (the recording transpose, the
scale and magnetic factors and the lossy divisor of :mod:`fdtdx.adjoint.objective`), solves
for their amplitudes, runs one adjoint solve with all of them (currents superpose, so several
monitors and every face of a box cost one solve), and pairs the adjoint and forward design
phasors into the ``inv_permittivities`` gradient (:mod:`fdtdx.adjoint.kernel`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from fdtdx.adjoint import validation
from fdtdx.adjoint.design import internal_scene
from fdtdx.adjoint.kernel import TailReport, amplitude_matrix, assemble_material_gradient, dft_tail, solve_amplitudes
from fdtdx.adjoint.objective import (
    ChannelRecording,
    LossyInjection,
    adjoint_sources,
    canonical_components,
    channel_recordings,
    lossy_injection,
    raw_scale,
    stored_fields,
    target_factor,
)
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.default_key import default_key
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import checkpointed_fdtd
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.sources.adjoint import AdjointCurrentSource
from fdtdx.objects.sources.source import Source


class ReciprocityPhasorFn:
    """``phasor_fn(inv_permittivities) -> phasors``, from :func:`~fdtdx.adjoint.reciprocity_phasor_fn`.

    Attributes:
        diagnostics: ``cond`` of the amplitude solve, and name -> :func:`~fdtdx.adjoint.kernel.dft_tail`
            of the latest forward (``objective_tail``, ``forward_design_tail``) and backward
            (``adjoint_design_tail``) solve, updated by every call, also under ``jit``.
    """

    def __init__(self, fn: Callable[[jax.Array], tuple[Any, ...]], *, single: bool, diagnostics: dict[str, Any]):
        self._fn = fn
        self._single = single
        self.diagnostics = diagnostics

    def __call__(self, inv_permittivities: jax.Array) -> Any:
        out = self._fn(inv_permittivities)
        return out[0] if self._single else out


@dataclass(frozen=True)
class _Channel:
    """One stored phasor array of an objective monitor, and what its cotangent drives."""

    detector: int  # index into the objective names
    recording: ChannelRecording
    sources: tuple[int, ...]  # its adjoint currents' indices in the adjoint object list, one per block
    losses: tuple[LossyInjection | None, ...]  # per block
    factor: jax.Array  # target_factor, broadcast over the cells
    scale: float  # raw_scale of the detector
    components: tuple[str, ...]
    label: str


def _adjoint_objects(objects, names, detectors, recordings, config, window, key):
    """``objects`` with every Source replaced by the monitors' adjoint currents, and the source names per monitor."""
    sources, groups = [], []
    for name, det, recs in zip(names, detectors, recordings):
        made = adjoint_sources(name, det, recs, config, window, default_key(key))
        sources.extend(made)
        groups.append(tuple(s.name for s in made))
    kept = [o for o in objects.object_list if not isinstance(o, Source)]
    volume_idx = next(i for i, o in enumerate(kept) if o is objects.volume)
    return ObjectContainer(object_list=[*kept, *sources], volume_idx=volume_idx), tuple(groups)


def derive_adjoint_objects(
    objects: ObjectContainer,
    config: SimulationConfig,
    objective_detectors: str | Sequence[str],
    window: jax.Array,
    key: jax.Array | None = None,
) -> tuple[ObjectContainer, tuple[tuple[str, ...], ...]]:
    """The adjoint scene the backward rule runs, before its design detectors are added.

    Every Source is replaced by zero-amplitude adjoint currents: per monitor, per stored
    channel (a box has one per face), per block of the cells the channel reads. Placed with
    ``place_on_grid``, so grid, materials and Device parameters are the forward scene's.

    Returns:
        ``(adjoint_objects, source_names)``, ``source_names[i]`` listing monitor ``i``'s currents.
    """
    names = validation.as_names(objective_detectors)
    detectors = validation.objective_detectors(objects, names)
    recordings = [channel_recordings(d, objects, config) for d in detectors]
    return _adjoint_objects(objects, names, detectors, recordings, config, window, key)


def make_phasor_fn(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    *,
    names: tuple[str, ...],
    detectors: Sequence[PhasorDetector],
    single: bool,
    design: str | Sequence[str] | None,
    window: jax.Array,
    cond_limit: float,
    tail_tolerance: float | None,
) -> ReciprocityPhasorFn:
    """The differentiable phasor function of a validated scene (see :func:`~fdtdx.adjoint.reciprocity_phasor_fn`)."""
    omegas = tuple(float(w) for w in detectors[0]._angular_frequencies)
    dt = float(config.time_step_duration)
    courant = float(config.courant_number)
    wave_characters = tuple(detectors[0].wave_characters)
    # precomputed on concrete values, so the solve inside the VJP is a plain jnp.linalg.solve
    matrix_np, cond = amplitude_matrix(omegas, dt, window, cond_limit)
    matrix = jnp.asarray(matrix_np)

    recordings = [channel_recordings(d, objects, config) for d in detectors]
    adjoint, groups = _adjoint_objects(objects, names, detectors, recordings, config, window, key)
    fwd_arrays, fwd_objects, des_names = internal_scene(
        arrays, objects, config, key, keep_detectors=names, design=design, wave_characters=wave_characters
    )
    adj_arrays, adj_objects, _ = internal_scene(
        arrays, adjoint, config, key, keep_detectors=(), design=design, wave_characters=wave_characters
    )
    des_slices = [fwd_objects[n].grid_slice for n in des_names]
    # the design scale rides on both the forward and the adjoint phasors
    des_scale_sq = [raw_scale(fwd_objects[n]) * raw_scale(adj_objects[n]) for n in des_names]

    index = {o.name: i for i, o in enumerate(adj_objects.object_list) if isinstance(o, AdjointCurrentSource)}
    channels: list[_Channel] = []
    for d_i, (name, det, recs, group) in enumerate(zip(names, detectors, recordings, groups)):
        comps = canonical_components(det)
        src_names = iter(group)
        for rec in recs:
            factor = target_factor(det, rec.exact, omegas, dt)
            channels.append(
                _Channel(
                    detector=d_i,
                    recording=rec,
                    sources=tuple(index[next(src_names)] for _ in rec.blocks),
                    # the conductivity of the arrays the adjoint solve injects into
                    losses=tuple(lossy_injection(adj_arrays, courant, b, comps) for b in rec.blocks),
                    factor=jnp.asarray(factor).reshape(factor.shape + (1,) * len(det.grid_shape)),
                    scale=raw_scale(det),
                    components=comps,
                    label=name if rec.state_key == "phasor" else f"{name}[{rec.state_key}]",
                )
            )
    # a monitor storing several channels (a box) returns its whole state dict
    returns_dict = [len(recs) > 1 for recs in recordings]

    diagnostics: dict[str, Any] = {
        "cond": cond,
        "tail_tolerance": tail_tolerance,
        "objective_tail": {},
        "forward_design_tail": {},
        "adjoint_design_tail": {},
    }
    report = TailReport(diagnostics, tail_tolerance)
    labels = tuple(ch.label for ch in channels)

    def design_tails(out):
        return tuple(
            dft_tail(out.fields.E[:, *sl], out.detector_states[n]["phasor"][0], omegas, dt)
            for n, sl in zip(des_names, des_slices)
        )

    def forward(inv_eps: jax.Array):
        _, out = checkpointed_fdtd(
            fwd_arrays.aset("inv_permittivities", inv_eps), fwd_objects, config, key, show_progress=False
        )
        objective_tails = []
        for ch in channels:
            raw = out.detector_states[names[ch.detector]][ch.recording.state_key][0] / ch.scale
            left = stored_fields(out.fields.E, out.fields.H, ch.components, ch.recording.channel_slice)
            objective_tails.append(dft_tail(left, raw, omegas, dt))
        # host callbacks, so the diagnostics also update under jit
        jax.debug.callback(partial(report, "objective_tail", labels), tuple(objective_tails))
        jax.debug.callback(partial(report, "forward_design_tail", des_names), design_tails(out))
        outs = tuple(
            dict(out.detector_states[name]) if wants_dict else out.detector_states[name]["phasor"]
            for name, wants_dict in zip(names, returns_dict)
        )
        return outs, tuple(out.detector_states[n]["phasor"] for n in des_names)

    @jax.custom_vjp
    def phasor_fn(inv_eps: jax.Array):
        return forward(inv_eps)[0]

    def phasor_fwd(inv_eps: jax.Array):
        outs, F = forward(inv_eps)
        return outs, (inv_eps, F)

    def phasor_bwd(res, ct):
        inv_eps, F = res
        # the adjoint container is built here, not closed over: a closed-over traced leaf
        # raises UnexpectedTracerError
        new_list = list(adj_objects.object_list)
        for ch in channels:
            ct_det = ct[ch.detector]
            ct_one = ct_det[ch.recording.state_key] if isinstance(ct_det, dict) else ct_det
            # cotangent on the recorded values -> on the raw Yee fields the currents drive
            for idx, target, loss in zip(ch.sources, ch.recording.transpose(ct_one[0]), ch.losses):
                if loss is not None:
                    target = target / loss.divisor(inv_eps)[None]
                new_list[idx] = new_list[idx].aset("amplitudes", solve_amplitudes(matrix, target * ch.factor))
        _, out_a = checkpointed_fdtd(
            adj_arrays.aset("inv_permittivities", inv_eps),
            adj_objects.aset("object_list", new_list),
            config,
            key,
            show_progress=False,
        )
        jax.debug.callback(partial(report, "adjoint_design_tail", des_names), design_tails(out_a))
        grad = jnp.zeros_like(inv_eps)
        for name, F_i, sl, scale_sq in zip(des_names, F, des_slices, des_scale_sq):
            g_design = assemble_material_gradient(
                adjoint_phasors=out_a.detector_states[name]["phasor"],
                forward_phasors=F_i,
                inv_permittivities=inv_eps[:, *sl],
                angular_frequencies=omegas,
                dt=dt,
                courant_number=courant,
            )
            # divided by a Python float so a float32 run keeps full precision; set, not
            # add: overlapping regions compute the same value from the same fields
            grad = grad.at[:, *sl].set((g_design / scale_sq).astype(inv_eps.dtype))
        return (grad,)

    phasor_fn.defvjp(phasor_fwd, phasor_bwd)
    return ReciprocityPhasorFn(phasor_fn, single=single, diagnostics=diagnostics)
