"""``run_fdtd`` with ``GradientConfig(method="reciprocity")``: the stock forward, differentiated by reciprocity.

An inverse-design script keeps ``apply_params -> run_fdtd -> figure of merit on
arrays.detector_states -> jax.grad`` and changes only the method string. The forward is
``checkpointed_fdtd``'s, bit for bit. The backward is one adjoint solve
(:class:`~fdtdx.adjoint.solve.AdjointSolve`) with adjoint currents at every phasor detector
whose state carries a nonzero cotangent (``symbolic_zeros``: a monitor the figure of merit does
not read costs nothing), and gives the gradient with respect to ``inv_permittivities`` inside
the Devices, which is where ``apply_params`` writes the Device parameters: exact for them, and
zero outside the Devices.

The dispersion coefficients count as constants: ``apply_params`` writes those of the Device
materials as a blend of equal rows, whose derivative is zero, and a dispersive Device material
is refused. Raised when the gradient is traced, instead of a wrong one: a figure of merit
reading anything but phasor detector states, another differentiated input, and the
configurations of :mod:`fdtdx.adjoint.validation`. Raised when it is computed: an estimated
truncation error above ``GradientConfig.tail_tolerance`` (``None`` switches that check off).
The setup runs when the backward rule is traced, so once under ``jax.jit``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero, custom_vjp_primal_tree_values

from fdtdx.adjoint import validation
from fdtdx.adjoint.design import internal_scene
from fdtdx.adjoint.objective import raw_scale
from fdtdx.adjoint.solve import AdjointSolve, design_tails, segmented_solve
from fdtdx.config import SimulationConfig
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.fdtd import checkpointed_fdtd
from fdtdx.objects.detectors.phasor import PhasorDetector

#: Array fields a differentiated value may reach without being refused: the fields and detector
#: states the run resets, and the dispersion coefficients (see the module docstring).
_CONSTANT_FIELDS = ("fields", "detector_states", "dispersive_c1", "dispersive_c2", "dispersive_c3")


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
    """Refuse a differentiated input other than ``inv_permittivities``, whose gradient would be dropped."""

    def perturbed(tree) -> bool:
        return any(getattr(leaf, "perturbed", False) for leaf in jax.tree.leaves(tree))

    wrong = [
        f"arrays{jax.tree_util.keystr(path)}"
        for path, leaf in jax.tree_util.tree_flatten_with_path(arrays)[0]
        if getattr(leaf, "perturbed", False) and getattr(path[0], "name", None) not in _CONSTANT_FIELDS
    ]
    wrong += [f"object {o.name!r}" for o in objects.object_list if perturbed(o)]
    if wrong:
        raise NotImplementedError(
            f"GradientConfig(method='reciprocity') differentiates inv_permittivities only, but {', '.join(wrong)} "
            "depend(s) on the differentiated parameters, and that gradient would be dropped. Use "
            "method='checkpointed'."
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
    """``run_fdtd`` for ``GradientConfig(method="reciprocity")``: :func:`~fdtdx.fdtd.fdtd.checkpointed_fdtd`,
    differentiated by reciprocity with respect to ``inv_permittivities`` inside the Devices."""
    assert config.gradient_config is not None  # run_fdtd dispatches here for method="reciprocity" only
    tolerance = config.gradient_config.tail_tolerance
    dynamic, static = eqx.partition((arrays.aset("inv_permittivities", None), objects, key), _traced)
    # set by whichever forward runs: the output's static leaves, the forward design detectors
    cell: dict = {}

    def scene(inv_eps, dyn):
        arrs, objs, k = eqx.combine(dyn, static)
        return arrs.aset("inv_permittivities", inv_eps), objs, k

    def finish(state):
        out, cell["static"] = eqx.partition(state, _traced)
        return out

    @jax.custom_vjp
    def run(inv_eps, dyn):
        arrs, objs, k = scene(inv_eps, dyn)
        return finish(
            checkpointed_fdtd(arrs, objs, config, k, show_progress=show_progress, progress_callback=progress_callback)
        )

    def run_fwd(inv_eps, dyn):
        _refuse_differentiated(*eqx.combine(dyn, static)[:2])
        inv_eps, dyn = custom_vjp_primal_tree_values((inv_eps, dyn))
        arrs, objs, k = scene(inv_eps, dyn)
        with jax.ensure_compile_time_eval():
            stretched = validation.check_scene(objs, arrs, config)
            fwd_arrays, fwd_objects, names = internal_scene(
                arrs,
                objs,
                config,
                k,
                keep_detectors=[d.name for d in objs.detectors],
                wave_characters=_scene_waves(objs),
            )
            omegas = tuple(float(w) for w in cast(PhasorDetector, fwd_objects[names[0]])._angular_frequencies)
        slices = [fwd_objects[n].grid_slice for n in names]
        cell.update(scales=[raw_scale(fwd_objects[n]) for n in names], omegas=np.asarray(omegas))
        (step, out), snapshots = segmented_solve(
            fwd_arrays, fwd_objects, config, k, show_progress=show_progress, progress_callback=progress_callback
        )
        cells = design_tails(out, snapshots, names, slices, omegas, float(config.time_step_duration))
        states = dict(out.detector_states)
        design = tuple(states.pop(n)["phasor"] for n in names)
        # the late-window snapshots of the monitors a figure of merit may read, the phasor detectors
        readable = {d.name for d in objs.detectors if isinstance(d, PhasorDetector)}
        snapshots = tuple({n: s for n, s in snap.items() if n in readable and n not in names} for snap in snapshots)
        out = finish((step, out.aset("detector_states", states)))
        return out, (inv_eps, dyn, design, cells, snapshots, out, stretched)

    def run_bwd(res, ct):
        inv_eps, dyn, design, cells, snapshots, (_, out), stretched = res
        names = _objectives(ct[1])
        if not names:
            return None, None
        arrs, objs, k = scene(inv_eps, dyn)
        with jax.ensure_compile_time_eval():
            adjoint = AdjointSolve(
                arrs, objs, config, k, names=names, tolerance=tolerance, forward_scales=cell["scales"]
            )
        # the forward design detectors record every phasor frequency of the scene; keep the objectives'
        rows = np.asarray([int(np.argmin(np.abs(cell["omegas"] - w))) for w in adjoint.omegas])
        out = eqx.combine(out, cell["static"][1])
        objective_tails = adjoint.objective_tails(out.fields.E, out.fields.H, out.detector_states, snapshots)
        cts = [
            {s: jnp.zeros(v.aval.shape, v.aval.dtype) if isinstance(v, SymbolicZero) else v for s, v in state.items()}
            for state in (ct[1].detector_states[n] for n in names)
        ]
        grad = adjoint(inv_eps, [f[:, rows] for f in design], cts, [c[rows] for c in cells], objective_tails)
        if stretched is not None:
            grad = eqx.error_if(grad, jnp.any(stretched), validation.STRETCHED_GRID.format(axes="an axis"))
        return grad, None

    run.defvjp(run_fwd, run_bwd, symbolic_zeros=True)
    return eqx.combine(run(arrays.inv_permittivities, dynamic), cell["static"])
