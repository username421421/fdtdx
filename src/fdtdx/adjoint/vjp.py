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

import importlib.util
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.adjoint import validation
from fdtdx.adjoint.design import internal_scene, late_windows
from fdtdx.adjoint.kernel import (
    TailReport,
    amplitude_matrix,
    assemble_conductivity_gradient,
    assemble_material_gradient,
    dft_tail,
    dft_tail_cells,
    gradient_error_estimate,
    leapfrog_kernel,
    refuse_unconverged,
    row_norms,
    solve_amplitudes,
    truncation_norms,
)
from fdtdx.adjoint.objective import (
    ChannelRecording,
    LossyInjection,
    adjoint_sources,
    canonical_components,
    channel_recordings,
    lossy_injection,
    objective_frequencies,
    raw_scale,
    stored_fields,
    target_factor,
)
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.default_key import default_key
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.fdtd import custom_fdtd_forward
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.sources.adjoint import AdjointCurrentSource
from fdtdx.objects.sources.source import Source


class ReciprocityPhasorFn:
    """``phasor_fn(inv_permittivities, electric_conductivity=None) -> phasors``.

    From :func:`~fdtdx.adjoint.reciprocity_phasor_fn`. ``electric_conductivity``, in the units
    ``ArrayContainer`` stores, replaces the scene's and is differentiated too; ``None`` keeps
    the scene's, as a constant, and is refused where ``apply_params`` writes another one into
    a Device (``conductive_devices``).

    Attributes:
        diagnostics: ``cond`` of the amplitude solve, and the convergence estimates of the latest
            gradient (``objective_tail``, ``forward_design_tail``, ``adjoint_design_tail``: name ->
            :func:`~fdtdx.adjoint.kernel.dft_tail`; ``gradient_error_estimate``), written by the
            backward rule, also under ``jit``; a call without a gradient leaves them as they were.
    """

    def __init__(
        self,
        fn: Callable[[jax.Array, jax.Array | None], tuple[Any, ...]],
        *,
        single: bool,
        diagnostics: dict[str, Any],
        conductive_devices: Sequence[str] = (),
    ):
        self._fn = fn
        self._single = single
        self.diagnostics = diagnostics
        self.conductive_devices = tuple(conductive_devices)

    def __call__(self, inv_permittivities: jax.Array, electric_conductivity: jax.Array | None = None) -> Any:
        if electric_conductivity is None and self.conductive_devices:
            raise ValueError(
                f"apply_params writes the conductivity of Device(s) {list(self.conductive_devices)} (a lossy material, "
                "or loss placed under a Device), so the placed one is not the design's. Pass both arrays apply_params "
                "returned, phasor_fn(arrays.inv_permittivities, arrays.electric_conductivity), or pass the placed "
                "conductivity explicitly to keep it."
            )
        out = self._fn(inv_permittivities, electric_conductivity)
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
    """``objects`` with every Source replaced by the monitors' adjoint currents, and the source names per monitor.

    Every current carries the union of the monitors' frequencies (:func:`objective_frequencies`).
    """
    omegas = objective_frequencies(detectors)[0]
    sources, groups = [], []
    for name, det, recs in zip(names, detectors, recordings):
        made = adjoint_sources(name, det, recs, config, window, default_key(key), angular_frequencies=omegas)
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


#: A read channel and frequency carrying less than this share of the figure of merit's first-order
#: change is left out of the objective check.
_NEGLIGIBLE_SHARE = 1e-6

#: Detector states at the three late-window starts of a solve, for the convergence estimate.
Snapshots = tuple[dict[str, Any], ...]


class _Progress:
    """One progress report over the segments of a solve: one tqdm bar and ``callback(step, total)``, as
    ``run_fdtd`` gives (each segment's own report restarted at 0 and ended at its own length)."""

    def __init__(self, total: int, show: bool, callback: Callable[[int, int], None] | None):
        self.total, self.callback, self.bar = total, callback, None
        self.show = show and importlib.util.find_spec("tqdm") is not None

    def segment(self, start: int) -> Callable[[int, int], None]:
        def report(step: int, _: int) -> None:
            done = start + int(step)
            if self.show:
                if self.bar is None:
                    from tqdm.auto import tqdm

                    self.bar = tqdm(total=self.total, desc="FDTD (forward)", unit="step", dynamic_ncols=True)
                self.bar.n = done
                self.bar.refresh()
                if done >= self.total:
                    self.bar.close()
                    self.bar = None
            if self.callback is not None:
                self.callback(done, self.total)

        return report


def segmented_solve(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    *,
    show_progress: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[tuple[jax.Array, ArrayContainer], Snapshots]:
    """A plain forward solve, in segments split at the late windows, and its detector states there.

    The same step body as ``checkpointed_fdtd``'s forward (every detector recorded, no boundary
    recording), so the output is that solve's; the snapshots cost three copies of the detector
    states and nothing per step.
    """
    bounds = (0, *late_windows(config), int(config.time_steps_total))
    progress = _Progress(bounds[-1], show_progress, progress_callback) if show_progress or progress_callback else None
    snapshots = []
    step, out = jnp.asarray(0, dtype=jnp.int32), arrays
    for i, (start, end) in enumerate(itertools.pairwise(bounds)):
        if i:
            snapshots.append(dict(out.detector_states))
        step, out = custom_fdtd_forward(
            out,
            objects,
            config,
            key,
            reset_container=i == 0,
            record_detectors=True,
            start_time=start,
            end_time=end,
            show_progress=False,
            progress_callback=None if progress is None else progress.segment(start),
        )
    return (step, out), tuple(snapshots)


def late_phasors(states: dict[str, Any], snapshots: Snapshots, name: str, key: str = "phasor") -> tuple:
    """The growth of detector ``name``'s stored ``key`` over each late window, oldest first."""
    marks = [s[name][key][0] for s in snapshots] + [states[name][key][0]]
    return tuple(later - earlier for earlier, later in itertools.pairwise(marks))


def design_tails(out: ArrayContainer, snapshots: Snapshots, names, slices, omegas, dt) -> tuple:
    """:func:`~fdtdx.adjoint.kernel.dft_tail_cells` of each design detector in the output of a solve:
    ``(regions, nf)`` relative tails, and per region its tails cell by cell, ``(nf, *cells)``."""
    states = out.detector_states
    each = [
        dft_tail_cells(out.fields.E[:, *sl], states[n]["phasor"][0], late_phasors(states, snapshots, n), omegas, dt)
        for n, sl in zip(names, slices)
    ]
    return jnp.stack([relative for relative, _ in each]), tuple(cells for _, cells in each)


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
        # one solve over every monitor's frequencies; rows[d] are monitor d's in it
        self.omegas, self.wave_characters, self.rows = objective_frequencies(detectors)
        validation.check_distinct_frequencies(self.omegas, names, self.rows)
        self.detector_omegas = [tuple(float(w) for w in d._angular_frequencies) for d in detectors]
        self.dt = float(config.time_step_duration)
        self.courant = float(config.courant_number)
        # precomputed on concrete values, so the solve inside the VJP is a plain jnp.linalg.solve
        matrix, self.cond = amplitude_matrix(self.omegas, self.dt, window, cond_limit)
        self.matrix = jnp.asarray(matrix)
        self.recordings = [channel_recordings(d, objects, config) for d in detectors]
        validation.check_objectives_outside_pml(
            objects,
            [
                (name if rec.state_key == "phasor" else f"{name}[{rec.state_key}]", block)
                for name, recs in zip(names, self.recordings)
                for rec in recs
                for block in rec.blocks
            ],
        )
        folded = validation.check_source_waveforms(
            objects, config, detectors, names, convergence=report.tolerance is not None
        )
        # per solve frequency, whether a monitor's stride folds foreign spectrum onto it: refused where read
        limit = validation.alias_limit(config)
        self.aliased = [np.isin(np.arange(len(self.omegas)), rows[f > limit]) for f, rows in zip(folded, self.rows)]
        self.alias_message = validation.alias_message(names, detectors, limit)
        adjoint, groups = _adjoint_objects(objects, names, detectors, self.recordings, config, window, key)
        self.arrays, self.objects, self.design_names = internal_scene(
            arrays, adjoint, config, key, keep_detectors=(), design=design, wave_characters=self.wave_characters
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
                factor = target_factor(det, rec.exact, self.detector_omegas[d_i], self.dt)
                self.channels.append(
                    _Channel(
                        detector=d_i,
                        recording=rec,
                        sources=tuple(index[next(src_names)] for _ in rec.blocks),
                        # the conductivity of the arrays the adjoint solve injects into
                        losses=tuple(lossy_injection(self.arrays, self.courant, b, comps) for b in rec.blocks),
                        factor=jnp.asarray(factor).reshape(factor.shape + (1,) * len(det.grid_shape)),
                        scale=raw_scale(det),
                        components=comps,
                        label=name if rec.state_key == "phasor" else f"{name}[{rec.state_key}]",
                    )
                )
        self.labels = tuple(ch.label for ch in self.channels)

    def objective_tails(
        self, E: jax.Array, H: jax.Array, states: dict[str, Any], snapshots: Snapshots
    ) -> tuple[jax.Array, jax.Array]:
        """``(channels, nf)`` :func:`~fdtdx.adjoint.kernel.dft_tail` of every objective channel of a forward
        solve, and the size of its stored phasors, per frequency (their weight in the objective check).

        ``states`` are its detector states, ``snapshots`` those at its late windows; ``nf`` are
        the solve's frequencies, 0 at those a channel does not record.
        """
        tails, sizes = [], []
        for ch in self.channels:
            name, key = self.names[ch.detector], ch.recording.state_key
            stored = states[name][key][0]
            late = tuple(x / ch.scale for x in late_phasors(states, snapshots, name, key))
            left = stored_fields(E, H, ch.components, ch.recording.channel_slice)
            tail = dft_tail(left, stored / ch.scale, late, self.detector_omegas[ch.detector], self.dt)
            # a frequency recorded twice is one phasor: its tail is not counted twice
            tails.append(self._to_solve(ch.detector, tail, reduce="max"))
            sizes.append(self._to_solve(ch.detector, row_norms(stored), reduce="max"))
        return jnp.stack(tails), jnp.stack(sizes)

    def _to_solve(self, detector: int, x: jax.Array, reduce: str = "add") -> jax.Array:
        """``x`` over monitor ``detector``'s frequencies, placed at its rows of the solve's (0 elsewhere)."""
        rows = self.rows[detector]
        if len(rows) == len(self.omegas) and np.array_equal(rows, np.arange(len(rows))):
            return x
        # added: a monitor recording one frequency twice drives it with both cotangents
        placed = jnp.zeros((len(self.omegas), *x.shape[1:]), x.dtype).at[rows]
        return placed.add(x) if reduce == "add" else placed.max(x)

    def __call__(
        self,
        inv_eps: jax.Array,
        sigma: jax.Array | None,
        forward_design: Sequence[jax.Array],
        cotangents: Sequence[dict[str, jax.Array]],
        forward_tails: jax.Array,
        forward_cells: Sequence[jax.Array],
        objective_tails: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array | None]:
        """``(d/d inv_eps, d/d sigma)`` from the forward design phasors and, per monitor, ``{state_key: cotangent}``.

        ``forward_tails`` ``(regions, nf)`` and ``forward_cells`` are the forward solve's
        :func:`design_tails`, ``objective_tails`` its :meth:`objective_tails` (tails and sizes);
        with the adjoint solve's they give the gradient's estimated distance from the converged
        one, which is refused above the tolerance (:func:`~fdtdx.adjoint.kernel.refuse_unconverged`).
        """
        live_sigma = self.arrays.electric_conductivity if sigma is None else sigma
        # the adjoint container is built here, not closed over: a closed-over traced leaf
        # raises UnexpectedTracerError
        new_list = list(self.objects.object_list)
        read = []
        for ch in self.channels:
            ct_one = cotangents[ch.detector][ch.recording.state_key]
            # how strongly the figure of merit reads each of the solve's frequencies on this channel
            read.append(self._to_solve(ch.detector, row_norms(ct_one[0])))
            # cotangent on the recorded values -> on the raw Yee fields the currents drive
            for idx, target, loss in zip(ch.sources, ch.recording.transpose(ct_one[0]), ch.losses):
                if loss is not None:
                    target = target / loss.divisor(inv_eps, live_sigma)[None]
                target = self._to_solve(ch.detector, target * ch.factor)
                new_list[idx] = new_list[idx].aset("amplitudes", solve_amplitudes(self.matrix, target))
        (_, out_a), snapshots = segmented_solve(
            with_materials(self.arrays, inv_eps, sigma),
            self.objects.aset("object_list", new_list),
            self.config,
            self.key,
        )
        adjoint_tails, adjoint_cells = design_tails(
            out_a, snapshots, self.design_names, self.design_slices, self.omegas, self.dt
        )
        # a frequency the figure of merit does not read has no adjoint current: its adjoint phasor's converged
        # value is 0, and what the solve leaves there (6e-8 to 7e-5 of a read one) is noise whose "tail" was
        # infinite and refused converged gradients
        read_freq = jnp.sum(jnp.stack(read), axis=0) > 0
        adjoint_tails = jnp.where(read_freq[None, :], adjoint_tails, 0.0)
        adjoint_cells = tuple(jnp.where(read_freq.reshape((-1,) + (1,) * (c.ndim - 1)), c, 0.0) for c in adjoint_cells)
        kern = leapfrog_kernel(self.omegas, self.dt, self.courant)
        grad = jnp.zeros_like(inv_eps)
        grad_sigma = None if sigma is None else jnp.zeros_like(sigma)
        truncation = 0.0
        regions = zip(
            self.design_names, forward_design, forward_cells, adjoint_cells, self.design_slices, self.design_scale_sq
        )
        for name, F_i, dF_i, dL_i, sl, scale_sq in regions:
            lam = out_a.detector_states[name]["phasor"]
            lam = jnp.where(read_freq.reshape((1, -1) + (1,) * (lam.ndim - 2)), lam, 0.0)
            g_design = assemble_material_gradient(lam, F_i, inv_eps[:, *sl], self.omegas, self.dt, self.courant)
            truncation = truncation + truncation_norms(lam, F_i, dL_i, dF_i, inv_eps[:, *sl], kern) / scale_sq
            # divided by a Python float so a float32 run keeps full precision; set, not
            # add: overlapping regions compute the same value from the same fields
            grad = grad.at[:, *sl].set((g_design / scale_sq).astype(inv_eps.dtype))
            if grad_sigma is not None:
                g_sigma = assemble_conductivity_gradient(lam, F_i, grad_sigma.shape[0], self.omegas, self.dt)
                grad_sigma = grad_sigma.at[:, *sl].set((g_sigma / scale_sq).astype(grad_sigma.dtype))
        # the objective phasors' truncation: the largest tail over the read channels and frequencies whose share
        # of the figure of merit's first-order change, ||cotangent|| ||phasor||, is not negligible. Unweighted, a
        # read channel carrying 1e-17 of it refused a figure of merit converged to 1e-7; averaged by share, a
        # penalty term near its target (share 2e-2, tail 2e-2) passed with the gradient 11-28% off
        tails, sizes = objective_tails
        weight = jnp.stack(read) * sizes
        read_tails = jnp.where(weight > 0, tails, 0.0)
        total = jnp.sum(weight)
        share = weight / jnp.where(total > 0, total, 1.0)
        objective = jnp.max(jnp.where(share > _NEGLIGIBLE_SHARE, read_tails, 0.0))
        estimate = gradient_error_estimate(jnp.asarray(truncation), row_norms(grad[None])[0], objective)
        # host callbacks, so the diagnostics also update under jit
        jax.debug.callback(self._record, forward_tails, adjoint_tails, read_tails, estimate)
        # a read row a stride folds foreign spectrum onto: the transpose cannot model it (an exactness refusal,
        # independent of tail_tolerance)
        rows_aliased = np.stack([self.aliased[ch.detector] for ch in self.channels])
        if rows_aliased.any():
            grad = eqx.error_if(grad, jnp.any((jnp.stack(read) > 0) & jnp.asarray(rows_aliased)), self.alias_message)
        grad = refuse_unconverged(grad, estimate, objective, self.report.tolerance)
        if grad_sigma is not None:
            grad_sigma = refuse_unconverged(grad_sigma, estimate, objective, self.report.tolerance)
        return grad, grad_sigma

    def _record(self, forward_tails, adjoint_tails, objective_tails, estimate) -> None:
        self.report("forward_design_tail", self.design_names, forward_tails)
        self.report("adjoint_design_tail", self.design_names, adjoint_tails)
        self.report("objective_tail", self.labels, objective_tails)
        self.report.diagnostics["gradient_error_estimate"] = float(estimate)


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
        "gradient_error_estimate": None,
    }
    report = TailReport(diagnostics, tail_tolerance)
    fwd_arrays, fwd_objects, des_names = internal_scene(
        arrays,
        objects,
        config,
        key,
        keep_detectors=names,
        design=design,
        wave_characters=objective_frequencies(detectors)[1],
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
        (_, out), snapshots = segmented_solve(with_materials(fwd_arrays, inv_eps, sigma), fwd_objects, config, key)
        tails = (
            *design_tails(out, snapshots, des_names, solve.design_slices, solve.omegas, solve.dt),
            solve.objective_tails(out.fields.E, out.fields.H, out.detector_states, snapshots),
        )
        outs = tuple(
            dict(out.detector_states[name]) if wants_dict else out.detector_states[name]["phasor"]
            for name, wants_dict in zip(names, returns_dict)
        )
        return outs, tuple(out.detector_states[n]["phasor"] for n in des_names), tails

    @jax.custom_vjp
    def phasor_fn(inv_eps: jax.Array, sigma: jax.Array | None):
        return forward(inv_eps, sigma)[0]

    def phasor_fwd(inv_eps: jax.Array, sigma: jax.Array | None):
        outs, F, tails = forward(inv_eps, sigma)
        return outs, (inv_eps, sigma, F, tails)

    def phasor_bwd(res, ct):
        inv_eps, sigma, F, tails = res
        return solve(inv_eps, sigma, F, [c if isinstance(c, dict) else {"phasor": c} for c in ct], *tails)

    phasor_fn.defvjp(phasor_fwd, phasor_bwd)
    return ReciprocityPhasorFn(
        phasor_fn,
        single=single,
        diagnostics=diagnostics,
        conductive_devices=validation.conductive_devices(objects, arrays),
    )
