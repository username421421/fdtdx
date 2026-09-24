"""The ``jax.custom_vjp``: a reciprocity gradient from two plain forward solves.

The forward rule runs the scene with the objective monitors and one internal design detector
per region, and returns the monitors' stored phasors. The backward rule,
:class:`AdjointSolve`, maps the cotangent of every stored channel to the target DFT of its
adjoint currents (the recording transpose, the scale and magnetic factors and the lossy
divisor of :mod:`fdtdx.adjoint.objective`), solves for their amplitudes, runs one adjoint
solve with all of them (currents superpose, so several monitors and every face of a box cost
one solve), and pairs the adjoint and forward design phasors into the ``inv_permittivities``
gradient and, when it is an input, the ``electric_conductivity`` one
(:mod:`fdtdx.adjoint.kernel`). :func:`make_phasor_fn` wraps it for the entry points,
:mod:`fdtdx.adjoint.dropin` for ``run_fdtd``.
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
from fdtdx.adjoint.kernel import (
    TailReport,
    amplitude_matrix,
    assemble_conductivity_gradient,
    assemble_material_gradient,
    dft_tail,
    solve_amplitudes,
)
from fdtdx.adjoint.objective import (
    ChannelRecording,
    LossyInjection,
    adjoint_sources,
    canonical_components,
    channel_recordings,
    lossy_injection,
    pml_weight,
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
    """``phasor_fn(inv_permittivities, electric_conductivity=None) -> phasors``.

    From :func:`~fdtdx.adjoint.reciprocity_phasor_fn`. ``electric_conductivity``, in the units
    ``ArrayContainer`` stores, replaces the scene's and is differentiated too; ``None`` keeps
    the scene's, as a constant.

    Attributes:
        diagnostics: ``cond`` of the amplitude solve, and name -> :func:`~fdtdx.adjoint.kernel.dft_tail`
            of the latest forward (``objective_tail``, ``forward_design_tail``) and backward
            (``adjoint_design_tail``) solve, updated by every call, also under ``jit``.
    """

    def __init__(
        self, fn: Callable[[jax.Array, jax.Array | None], tuple[Any, ...]], *, single: bool, diagnostics: dict[str, Any]
    ):
        self._fn = fn
        self._single = single
        self.diagnostics = diagnostics

    def __call__(self, inv_permittivities: jax.Array, electric_conductivity: jax.Array | None = None) -> Any:
        out = self._fn(inv_permittivities, electric_conductivity)
        return out[0] if self._single else out


@dataclass(frozen=True)
class _Channel:
    """One stored phasor array of an objective monitor, and what its cotangent drives."""

    detector: int  # index into the objective names
    recording: ChannelRecording
    sources: tuple[int, ...]  # its adjoint currents' indices in the adjoint object list, one per block
    losses: tuple[LossyInjection | None, ...]  # per block
    pml: tuple[Any, ...]  # per block: local PML strength per cell, or None
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


def with_materials(arrays: ArrayContainer, inv_eps: jax.Array, sigma: jax.Array | None) -> ArrayContainer:
    """``arrays`` with ``inv_eps`` and, unless ``None``, ``sigma`` as its electric conductivity."""
    arrays = arrays.aset("inv_permittivities", inv_eps)
    if sigma is None:
        return arrays
    if arrays.electric_conductivity is None:
        # the objective monitors' lossy correction is built from the scene's conductivity
        raise ValueError(
            "electric_conductivity was passed, but the scene stores none. Place the scene with a "
            "conductive material, or pass None."
        )
    return arrays.aset("electric_conductivity", sigma)


def design_tails(out: ArrayContainer, names, slices, omegas, dt) -> tuple[jax.Array, ...]:
    """:func:`~fdtdx.adjoint.kernel.dft_tail` of each design detector in the output of a solve."""
    return tuple(
        dft_tail(out.fields.E[:, *sl], out.detector_states[n]["phasor"][0], omegas, dt) for n, sl in zip(names, slices)
    )


class AdjointSolve:
    """The backward rule for the objective monitors ``names``: one adjoint solve on the placed scene.

    Set up eagerly by :func:`make_phasor_fn`, and by ``run_fdtd`` with
    ``GradientConfig("reciprocity")`` when its backward rule is traced, on traced arrays and
    under ``jax.ensure_compile_time_eval``: what is concrete (monitor transposes and their
    support, the amplitude matrix, the placed adjoint currents and design detectors) is
    evaluated then, also under ``jit``, never per call.

    Args: as :func:`make_phasor_fn`; ``report`` receives the diagnostics, ``forward_scales``
        is the ``raw_scale`` of each forward design detector, in :func:`internal_scene` order.
    """

    def __init__(
        self,
        arrays: ArrayContainer,
        objects: ObjectContainer,
        config: SimulationConfig,
        key: jax.Array,
        *,
        names: tuple[str, ...],
        detectors: Sequence[PhasorDetector],
        design: str | Sequence[str] | None,
        window: jax.Array,
        cond_limit: float,
        report: TailReport,
        forward_scales: Sequence[float],
    ):
        self.names, self.config, self.key, self.report = names, config, key, report
        self.omegas = tuple(float(w) for w in detectors[0]._angular_frequencies)
        self.dt = float(config.time_step_duration)
        self.courant = float(config.courant_number)
        # precomputed on concrete values, so the solve inside the VJP is a plain jnp.linalg.solve
        matrix, self.cond = amplitude_matrix(self.omegas, self.dt, window, cond_limit)
        self.matrix = jnp.asarray(matrix)
        self.recordings = [channel_recordings(d, objects, config) for d in detectors]
        adjoint, groups = _adjoint_objects(objects, names, detectors, self.recordings, config, window, key)
        self.arrays, self.objects, self.design_names = internal_scene(
            arrays, adjoint, config, key, keep_detectors=(), design=design, wave_characters=detectors[0].wave_characters
        )
        self.design_slices = [self.objects[n].grid_slice for n in self.design_names]
        # the design scale rides on both the forward and the adjoint phasors
        self.design_scale_sq = [f * raw_scale(self.objects[n]) for f, n in zip(forward_scales, self.design_names)]

        index = {o.name: i for i, o in enumerate(self.objects.object_list) if isinstance(o, AdjointCurrentSource)}
        self.channels: list[_Channel] = []
        for d_i, (name, det, recs, group) in enumerate(zip(names, detectors, self.recordings, groups)):
            comps = canonical_components(det)
            src_names = iter(group)
            for rec in recs:
                factor = target_factor(det, rec.exact, self.omegas, self.dt)
                self.channels.append(
                    _Channel(
                        detector=d_i,
                        recording=rec,
                        sources=tuple(index[next(src_names)] for _ in rec.blocks),
                        # the conductivity of the arrays the adjoint solve injects into
                        losses=tuple(lossy_injection(self.arrays, self.courant, b, comps) for b in rec.blocks),
                        pml=tuple(pml_weight(objects, b) for b in rec.blocks),
                        factor=jnp.asarray(factor).reshape(factor.shape + (1,) * len(det.grid_shape)),
                        scale=raw_scale(det),
                        components=comps,
                        label=name if rec.state_key == "phasor" else f"{name}[{rec.state_key}]",
                    )
                )
        self.labels = tuple(ch.label for ch in self.channels)
        self.reads_pml = any(m is not None for ch in self.channels for m in ch.pml)

    def objective_tails(self, E: jax.Array, H: jax.Array, states: dict[str, Any]) -> None:
        """Report the DFT tail of every objective channel of a forward solve; ``states`` are its detector states."""
        tails = []
        for ch in self.channels:
            raw = states[self.names[ch.detector]][ch.recording.state_key][0] / ch.scale
            left = stored_fields(E, H, ch.components, ch.recording.channel_slice)
            tails.append(dft_tail(left, raw, self.omegas, self.dt))
        # host callbacks, so the diagnostics also update under jit
        jax.debug.callback(partial(self.report, "objective_tail", self.labels), tuple(tails))

    def __call__(
        self,
        inv_eps: jax.Array,
        sigma: jax.Array | None,
        forward_design: Sequence[jax.Array],
        cotangents: Sequence[dict[str, jax.Array]],
    ) -> tuple[jax.Array, jax.Array | None]:
        """``(d/d inv_eps, d/d sigma)`` from the forward design phasors and, per monitor, ``{state_key: cotangent}``."""
        live_sigma = self.arrays.electric_conductivity if sigma is None else sigma
        # the adjoint container is built here, not closed over: a closed-over traced leaf
        # raises UnexpectedTracerError
        new_list = list(self.objects.object_list)
        shares = []
        for ch in self.channels:
            ct_one = cotangents[ch.detector][ch.recording.state_key]
            inside = total = 0.0
            # cotangent on the recorded values -> on the raw Yee fields the currents drive
            for idx, target, loss, weight in zip(ch.sources, ch.recording.transpose(ct_one[0]), ch.losses, ch.pml):
                if loss is not None:
                    target = target / loss.divisor(inv_eps, live_sigma)[None]
                power = jnp.abs(target) ** 2
                total = total + jnp.sum(power)
                if weight is not None:
                    inside = inside + jnp.sum(power * weight**2)
                new_list[idx] = new_list[idx].aset("amplitudes", solve_amplitudes(self.matrix, target * ch.factor))
            shares.append(jnp.sqrt(inside / jnp.maximum(total, jnp.finfo(power.dtype).tiny)))
        if self.reads_pml:
            jax.debug.callback(partial(self.report, "objective_pml_share", self.labels), tuple(shares))
        _, out_a = checkpointed_fdtd(
            with_materials(self.arrays, inv_eps, sigma),
            self.objects.aset("object_list", new_list),
            self.config,
            self.key,
            show_progress=False,
        )
        tails = design_tails(out_a, self.design_names, self.design_slices, self.omegas, self.dt)
        jax.debug.callback(partial(self.report, "adjoint_design_tail", self.design_names), tails)
        grad = jnp.zeros_like(inv_eps)
        grad_sigma = None if sigma is None else jnp.zeros_like(sigma)
        for name, F_i, sl, scale_sq in zip(self.design_names, forward_design, self.design_slices, self.design_scale_sq):
            lam = out_a.detector_states[name]["phasor"]
            g_design = assemble_material_gradient(lam, F_i, inv_eps[:, *sl], self.omegas, self.dt, self.courant)
            # divided by a Python float so a float32 run keeps full precision; set, not
            # add: overlapping regions compute the same value from the same fields
            grad = grad.at[:, *sl].set((g_design / scale_sq).astype(inv_eps.dtype))
            if grad_sigma is not None:
                g_sigma = assemble_conductivity_gradient(lam, F_i, grad_sigma.shape[0], self.omegas, self.dt)
                grad_sigma = grad_sigma.at[:, *sl].set((g_sigma / scale_sq).astype(grad_sigma.dtype))
        return grad, grad_sigma


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
    diagnostics: dict[str, Any] = {
        "cond": None,
        "tail_tolerance": tail_tolerance,
        "objective_tail": {},
        "forward_design_tail": {},
        "adjoint_design_tail": {},
        "objective_pml_share": {},
    }
    report = TailReport(diagnostics, tail_tolerance)
    fwd_arrays, fwd_objects, des_names = internal_scene(
        arrays, objects, config, key, keep_detectors=names, design=design, wave_characters=detectors[0].wave_characters
    )
    solve = AdjointSolve(
        arrays,
        objects,
        config,
        key,
        names=names,
        detectors=detectors,
        design=design,
        window=window,
        cond_limit=cond_limit,
        report=report,
        forward_scales=[raw_scale(fwd_objects[n]) for n in des_names],
    )
    diagnostics["cond"] = solve.cond
    # a monitor storing several channels (a box) returns its whole state dict
    returns_dict = [len(recs) > 1 for recs in solve.recordings]

    def forward(inv_eps: jax.Array, sigma: jax.Array | None):
        _, out = checkpointed_fdtd(
            with_materials(fwd_arrays, inv_eps, sigma), fwd_objects, config, key, show_progress=False
        )
        solve.objective_tails(out.fields.E, out.fields.H, out.detector_states)
        tails = design_tails(out, des_names, solve.design_slices, solve.omegas, solve.dt)
        jax.debug.callback(partial(report, "forward_design_tail", des_names), tails)
        outs = tuple(
            dict(out.detector_states[name]) if wants_dict else out.detector_states[name]["phasor"]
            for name, wants_dict in zip(names, returns_dict)
        )
        return outs, tuple(out.detector_states[n]["phasor"] for n in des_names)

    @jax.custom_vjp
    def phasor_fn(inv_eps: jax.Array, sigma: jax.Array | None):
        return forward(inv_eps, sigma)[0]

    def phasor_fwd(inv_eps: jax.Array, sigma: jax.Array | None):
        outs, F = forward(inv_eps, sigma)
        return outs, (inv_eps, sigma, F)

    def phasor_bwd(res, ct):
        inv_eps, sigma, F = res
        return solve(inv_eps, sigma, F, [c if isinstance(c, dict) else {"phasor": c} for c in ct])

    phasor_fn.defvjp(phasor_fwd, phasor_bwd)
    return ReciprocityPhasorFn(phasor_fn, single=single, diagnostics=diagnostics)
