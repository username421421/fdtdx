"""``run_fdtd`` with ``GradientConfig(method="reciprocity")``: the stock pipeline, differentiated by reciprocity.

An inverse-design script keeps ``apply_params -> run_fdtd -> figure of merit on
arrays.detector_states -> jax.grad`` and changes only the method string. The forward is
``run_fdtd``'s, bit for bit. The backward rule is :class:`~fdtdx.adjoint.vjp.AdjointSolve`
with adjoint currents at every phasor detector whose state carries a nonzero cotangent
(``symbolic_zeros``: a monitor the figure of merit does not read costs nothing).

* Differentiated inputs: ``inv_permittivities`` and ``electric_conductivity``, everything
  ``apply_params`` writes from Device parameters. The design region is every Device, so the
  gradient is exact for Device parameters and zero outside the Devices.
* Raised at trace time instead of a silent zero: a nonzero cotangent on anything but a phasor
  detector's state (a time-domain detector, the fields, ...), a differentiated input it does not
  cover, and every refusal of :mod:`fdtdx.adjoint.validation`.
* The setup (monitor transposes, adjoint scene, amplitude matrix) runs when the backward rule is
  traced: once under ``jax.jit``, on every call without it, where recompiling the time loops
  costs far more.
* ``ConvergenceWarning`` and ``PmlWarning`` report as for ``reciprocity_param_fn``, at the
  default tolerance; ``warnings.filterwarnings`` silences them.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero, custom_vjp_primal_tree_values

from fdtdx.adjoint import validation
from fdtdx.adjoint.design import design_regions, internal_scene
from fdtdx.adjoint.kernel import DEFAULT_COND_LIMIT, DEFAULT_TAIL_TOLERANCE, TailReport, gaussian_window
from fdtdx.adjoint.objective import raw_scale
from fdtdx.adjoint.vjp import AdjointSolve, design_tails
from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.fdtd import checkpointed_fdtd
from fdtdx.objects.detectors.phasor import PhasorDetector


def _traced(leaf) -> bool:
    """The leaves passed through the ``custom_vjp``; Python and NumPy values stay static."""
    return isinstance(leaf, jax.Array)


def _is_zero(ct) -> bool:
    return isinstance(ct, SymbolicZero) or ct.dtype == jax.dtypes.float0


def _scene_waves(objects: ObjectContainer) -> tuple:
    """One wave character per distinct frequency of the scene's phasor detectors, any of which the loss may read."""
    waves = {}
    for det in objects.detectors:
        if isinstance(det, PhasorDetector):
            for wc in det.wave_characters:
                waves.setdefault(float(wc.get_frequency()), wc)
    if not waves:
        raise ValueError(
            "GradientConfig(method='reciprocity') differentiates phasor detectors, and the scene has none. Record "
            "a PhasorDetector and compute the figure of merit on its phasors, or use method='checkpointed'."
        )
    return tuple(waves.values())


def _refuse_differentiated(arrays: ArrayContainer, objects: ObjectContainer) -> None:
    """Refuse a differentiated input whose gradient ``run_fdtd(checkpointed)`` has and this would drop.

    Fields and detector states are reset by the run, and the dispersion coefficients
    ``apply_params`` writes are design-independent for the non-dispersive Device materials
    admitted, so neither contributes in either method.
    """

    def perturbed(tree) -> bool:
        return any(getattr(leaf, "perturbed", False) for leaf in jax.tree.leaves(tree))

    wrong = [f"arrays.{n}" for n in ("inv_permeabilities", "magnetic_conductivity") if perturbed(getattr(arrays, n))]
    wrong += [f"object {o.name!r}" for o in objects.object_list if perturbed(o)]
    if wrong:
        raise NotImplementedError(
            f"GradientConfig(method='reciprocity') differentiates inv_permittivities and electric_conductivity only, "
            f"but {', '.join(wrong)} depend(s) on the differentiated parameters, and that gradient would be dropped. "
            "Use method='checkpointed'."
        )


def _objectives(ct: ArrayContainer) -> tuple[str, ...]:
    """The detectors whose state the figure of merit reads; refuse a read of any other output."""
    names, read = {}, []
    for path, leaf in jax.tree_util.tree_flatten_with_path(ct, is_leaf=lambda x: isinstance(x, SymbolicZero))[0]:
        if _is_zero(leaf):
            continue
        if getattr(path[0], "name", None) == "detector_states":
            names[path[1].key] = None
        else:
            read.append(jax.tree_util.keystr(path))
    if read:
        raise NotImplementedError(
            f"The figure of merit depends on the run_fdtd output(s) {', '.join(read)}, which "
            "GradientConfig(method='reciprocity') does not differentiate: it differentiates phasor detector states "
            "only. Compute the figure of merit from phasor detectors, or use method='checkpointed'."
        )
    return tuple(names)


def reciprocity_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    """``run_fdtd`` for ``GradientConfig(method="reciprocity")``: :func:`checkpointed_fdtd`, differentiated by reciprocity."""
    rest = arrays.aset("inv_permittivities", None).aset("electric_conductivity", None)
    dynamic, static = eqx.partition((rest, objects, key), _traced)
    report = TailReport({}, DEFAULT_TAIL_TOLERANCE)
    # set by whichever forward runs: the output's static leaves, the forward design detectors
    cell: dict = {}

    def scene(inv_eps, sigma, dyn):
        arrs, objs, k = eqx.combine(dyn, static)
        return arrs.aset("inv_permittivities", inv_eps).aset("electric_conductivity", sigma), objs, k

    def forward(arrs, objs, k):
        return checkpointed_fdtd(
            arrs, objs, config, k, show_progress=show_progress, progress_callback=progress_callback
        )

    def finish(state):
        out, cell["static"] = eqx.partition(state, _traced)
        return out

    @jax.custom_vjp
    def run(inv_eps, sigma, dyn):
        return finish(forward(*scene(inv_eps, sigma, dyn)))

    def run_fwd(inv_eps, sigma, dyn):
        _refuse_differentiated(*eqx.combine(dyn, static)[:2])
        inv_eps, sigma, dyn = custom_vjp_primal_tree_values((inv_eps, sigma, dyn))
        arrs, objs, k = scene(inv_eps, sigma, dyn)
        with jax.ensure_compile_time_eval():
            validation.check_scene(objs, arrs, config)
            validation.check_device_materials(objs)
            regions = design_regions(objs, None)
            validation.check_outside_pml(objs, regions)
            validation.check_sources_outside(objs, regions)
            waves = _scene_waves(objs)
            fwd_arrays, fwd_objects, names = internal_scene(
                arrs,
                objs,
                config,
                k,
                keep_detectors=[d.name for d in objs.detectors],
                design=None,
                wave_characters=waves,
            )
            omegas = tuple(float(w) for w in cast(PhasorDetector, fwd_objects[names[0]])._angular_frequencies)
        slices = [fwd_objects[n].grid_slice for n in names]
        cell.update(scales=[raw_scale(fwd_objects[n]) for n in names], omegas=np.asarray(omegas))
        step, out = forward(fwd_arrays, fwd_objects, k)
        tails = design_tails(out, names, slices, omegas, float(config.time_step_duration))
        jax.debug.callback(partial(report, "forward_design_tail", names), tails)
        states = dict(out.detector_states)
        design = tuple(states.pop(n)["phasor"] for n in names)
        out = finish((step, out.aset("detector_states", states)))
        return out, (inv_eps, sigma, dyn, design, out)

    def run_bwd(res, ct):
        inv_eps, sigma, dyn, design, (_, out) = res
        names = _objectives(ct[1])
        if not names:
            return None, None, None
        arrs, objs, k = scene(inv_eps, sigma, dyn)
        with jax.ensure_compile_time_eval():
            adjoint = AdjointSolve(
                arrs,
                objs,
                config,
                k,
                names=names,
                detectors=validation.objective_detectors(objs, names),
                design=None,
                window=gaussian_window(int(config.time_steps_total), dtype=config.dtype),
                cond_limit=DEFAULT_COND_LIMIT,
                report=report,
                forward_scales=cell["scales"],
            )
        # the forward design detectors record every phasor frequency of the scene; keep the objectives'
        rows = [int(np.argmin(np.abs(cell["omegas"] - w))) for w in adjoint.omegas]
        out = eqx.combine(out, cell["static"][1])
        adjoint.objective_tails(out.fields.E, out.fields.H, out.detector_states)
        cts = [
            {s: jnp.zeros(v.aval.shape, v.aval.dtype) if isinstance(v, SymbolicZero) else v for s, v in state.items()}
            for state in (ct[1].detector_states[n] for n in names)
        ]
        grad, grad_sigma = adjoint(inv_eps, sigma, [f[:, rows] for f in design], cts)
        return grad, grad_sigma, None

    run.defvjp(run_fwd, run_bwd, symbolic_zeros=True)
    return eqx.combine(run(arrays.inv_permittivities, arrays.electric_conductivity, dynamic), cell["static"])
