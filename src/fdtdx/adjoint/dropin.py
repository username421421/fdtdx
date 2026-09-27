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
* Raised on every call instead of a wrong gradient: an estimated truncation error above
  ``GradientConfig.tail_tolerance`` (phasors that have not converged), as for
  ``reciprocity_param_fn``; ``tail_tolerance=None`` switches that check off.
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
from fdtdx.adjoint.design import design_regions, internal_scene
from fdtdx.adjoint.kernel import DEFAULT_COND_LIMIT, TailReport, gaussian_window
from fdtdx.adjoint.objective import raw_scale
from fdtdx.adjoint.vjp import AdjointSolve, design_tails, segmented_solve
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.utils import is_jax_tracer
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, SimulationState
from fdtdx.fdtd.fdtd import checkpointed_fdtd
from fdtdx.fdtd.initialization import AppliedRecord, applied_record
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

    Everything but ``inv_permittivities`` and ``electric_conductivity``, except the fields and
    detector states, which the run resets (no gradient in either method). ``apply_params``
    writes the dispersion coefficients of the non-dispersive Device materials admitted as
    constants, so a differentiated coefficient is the caller's (zero here, exact natively).
    """

    def perturbed(tree) -> bool:
        return any(getattr(leaf, "perturbed", False) for leaf in jax.tree.leaves(tree))

    reset = ("fields", "detector_states")
    wrong = [
        f"arrays{jax.tree_util.keystr(path)}"
        for path, leaf in jax.tree_util.tree_flatten_with_path(arrays)[0]
        if getattr(leaf, "perturbed", False) and getattr(path[0], "name", None) not in reset
    ]
    wrong += [f"object {o.name!r}" for o in objects.object_list if perturbed(o)]
    if wrong:
        raise NotImplementedError(
            f"GradientConfig(method='reciprocity') differentiates inv_permittivities and electric_conductivity only, "
            f"but {', '.join(wrong)} depend(s) on the differentiated parameters, and that gradient would be dropped. "
            "Use method='checkpointed'."
        )


def _refuse_unwritten(inv_eps, sigma, records: tuple, objects: ObjectContainer) -> None:
    """Refuse a differentiated material array ``apply_params`` did not write, or wrote for other Devices.

    The gradient is computed inside the Devices only (zero elsewhere), which is exact for
    Device parameters. Differentiated directly, or edited after ``apply_params`` (a background
    parameter, a blur), the array's sensitivity outside the Devices would be a silent zero:
    97-99% of the gradient norm in a test scene, and a blur corrupted the Device part too (0.15).
    ``records`` (:class:`~fdtdx.fdtd.initialization.AppliedRecord` or ``None``) are read here,
    when the gradient is taken, not when ``run_fdtd`` is called (see there); their flags belong to
    one differentiation, and are cleared after it (:func:`_clear_flags`), since ``jax.jit`` reuses a
    traced program and its records for the next. A Device written but missing from ``objects``
    would get no gradient (its part was 0 against checkpointed's).
    """
    current = {d.grid_slice_tuple for d in objects.devices}
    offending, missing = [], []
    for name, primal, record in (
        ("inv_permittivities", inv_eps, records[0]),
        ("electric_conductivity", sigma, records[1]),
    ):
        if primal is None or not getattr(primal, "perturbed", False):
            continue
        if record is None or record.differentiated:
            offending.append(name)
        else:
            missing += [s for s in record.devices if s not in current and s not in missing]
    if missing or offending:
        _clear_flags(records)
    if missing:
        raise NotImplementedError(
            f"apply_params wrote Devices at grid slices {missing} that the objects passed to run_fdtd do not contain, "
            "so GradientConfig(method='reciprocity') would give them no gradient. Pass run_fdtd the objects "
            "apply_params returned, or use method='checkpointed'."
        )
    if offending:
        raise NotImplementedError(
            f"GradientConfig(method='reciprocity') differentiates arrays.{' and arrays.'.join(offending)} inside the "
            "Devices only, which is exact for Device parameters, but the array was not written by apply_params from "
            "undifferentiated material arrays: it is differentiated directly, or edited before or after apply_params "
            "(a background parameter, a smoothing), or it passed through anything between the two calls (its own "
            "jax.jit around apply_params, a lax.scan or lax.cond carry, even an astype): the check follows the very "
            "array apply_params returned. Its sensitivity outside the Devices would be dropped. Differentiate Device "
            "parameters only, with apply_params and run_fdtd in the same traced function, or use "
            "method='checkpointed'."
        )


def _clear_flags(records: tuple) -> None:
    """End one differentiation's reading of the ``apply_params`` records (see :func:`_refuse_unwritten`)."""
    for record in records:
        if record is not None:
            record.differentiated.clear()


def _jvp_tracer(leaf) -> bool:
    """Forward mode: a ``JVPTracer`` reaches ``run_fdtd``. Reverse mode reaches it as a ``LinearizeTracer``,
    unless JAX linearizes by JVP (``jax_use_direct_linearize=False``), when its tracer is a JVPTracer too."""
    return type(leaf).__name__ == "JVPTracer" and bool(getattr(jax.config, "jax_use_direct_linearize", True))


def _refuse_untraceable(arrays: ArrayContainer, objects: ObjectContainer) -> list:
    """Refuse forward-mode tracers, and differentiated tracers in frozen object fields, before the
    ``custom_vjp`` sees them.

    A ``custom_vjp`` has no forward-mode rule (``jax.jvp``, ``jacfwd``: a generic ``TypeError``,
    which is what they still give under ``jax.jit``, where the tracer seen here is a staging
    one); forward-over-reverse (``jax.hessian``, ``jax.jvp(jax.grad(f))``) works, as natively. A
    differentiated value stored in a frozen field sits in the tree structure, so the rule closes
    over it (``UnexpectedTracerError``); an undifferentiated ``jit`` argument there is exact
    (rel 2.4e-10) and is let through, and returned, for the rule to refuse it where the jitted
    function is differentiated from outside (:func:`_refuse_outer_staging`).
    """
    leaves = jax.tree.leaves((arrays, objects))
    if any(_jvp_tracer(leaf) for leaf in leaves):
        raise NotImplementedError(
            "Forward-mode differentiation (jax.jvp, jax.jacfwd) through run_fdtd with "
            "GradientConfig(method='reciprocity') is not supported: the reciprocity gradient is a reverse-mode rule. "
            "Use reverse mode (jax.grad, jax.vjp; jax.hessian or jax.jvp(jax.grad(f)) for second derivatives, "
            "which work), or method='checkpointed'."
        )
    leaf_ids = {id(leaf) for leaf in leaves}
    found: list[str] = []
    staged: list = []
    seen: set[int] = set()

    def visit(value, path: str, depth: int) -> None:
        if depth > 6 or id(value) in seen:
            return
        seen.add(id(value))
        if is_jax_tracer(value):
            # a staging (jit) tracer is a constant of the rule unless differentiated, which JAX then refuses itself
            if id(value) not in leaf_ids:
                (staged if type(value).__name__ == "DynamicJaxprTracer" else found).append(
                    path if type(value).__name__ != "DynamicJaxprTracer" else value
                )
            return
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(item, f"{path}[{i}]", depth + 1)
        elif isinstance(value, dict):
            for k, item in value.items():
                visit(item, f"{path}[{k!r}]", depth + 1)
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            for k in vars(value):
                visit(getattr(value, k, None), f"{path}.{k}", depth + 1)

    for obj in objects.object_list:
        visit(obj, repr(getattr(obj, "name", type(obj).__name__)), 0)
    if found:
        raise NotImplementedError(
            f"{', '.join(found[:4])} {'is' if len(found) == 1 else 'are'} traced but stored in frozen object fields "
            "(for example a source amplitude under jax.grad). GradientConfig(method='reciprocity') differentiates "
            "inv_permittivities and electric_conductivity only, which is everything apply_params writes from Device "
            "parameters. Keep other object fields constant, or use method='checkpointed'."
        )
    return staged


def _refuse_outer_staging(staged: list, inv_eps) -> None:
    """Refuse ``jit`` values kept in object fields when the jitted function is differentiated from outside.

    Under ``jax.grad(jax.jit(f))`` the rule is traced when the staged program is differentiated,
    in another trace than the one those values belong to, and lowering it failed with a bare
    ``TypeError: No constant handler`` (``jax.jit(jax.grad(f))`` traces the rule in theirs, and
    is exact).
    """
    current = getattr(inv_eps, "_trace", None)
    if current is not None and any(getattr(value, "_trace", None) is not current for value in staged):
        raise NotImplementedError(
            "A value traced by jax.jit is stored in an object field (a source amplitude passed to the jitted "
            "function, for example), and the jitted function is differentiated from outside (jax.grad(jax.jit(f))). "
            "GradientConfig(method='reciprocity') cannot carry it into its rule. Differentiate inside the jit "
            "(jax.jit(jax.grad(f))), keep the field constant, or use method='checkpointed'."
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
    staged = _refuse_untraceable(arrays, objects)
    # read when the gradient is taken: under jax.grad(jax.jit(f)) the probe's flags are filled only then
    records: tuple[AppliedRecord | None, AppliedRecord | None] = (
        applied_record(arrays.inv_permittivities),
        None if arrays.electric_conductivity is None else applied_record(arrays.electric_conductivity),
    )
    for record in records:
        if record is not None:
            record.consumed.append(True)
    rest = arrays.aset("inv_permittivities", None).aset("electric_conductivity", None)
    dynamic, static = eqx.partition((rest, objects, key), _traced)
    assert config.gradient_config is not None  # run_fdtd dispatches here for method="reciprocity" only
    report = TailReport({}, config.gradient_config.tail_tolerance)
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
        try:
            return forward_rule(inv_eps, sigma, dyn)
        except BaseException:
            # the flags belong to this differentiation, also when it was refused (a stale flag refused the next)
            _clear_flags(records)
            raise

    def forward_rule(inv_eps, sigma, dyn):
        _refuse_outer_staging(staged, custom_vjp_primal_tree_values(inv_eps))
        _refuse_differentiated(*eqx.combine(dyn, static)[:2])
        _refuse_unwritten(inv_eps, sigma, records, eqx.combine(dyn, static)[1])
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
        (step, out), snapshots = segmented_solve(
            fwd_arrays, fwd_objects, config, k, show_progress=show_progress, progress_callback=progress_callback
        )
        tails, cells = design_tails(out, snapshots, names, slices, omegas, float(config.time_step_duration))
        states = dict(out.detector_states)
        design = tuple(states.pop(n)["phasor"] for n in names)
        # the late-window snapshots of the monitors a figure of merit may read (phasor detectors: a time-domain
        # detector's history is refused as an objective, and three copies of it would be kept to the backward)
        readable = {d.name for d in objs.detectors if isinstance(d, PhasorDetector)}
        snapshots = tuple({n: s for n, s in snap.items() if n in readable and n not in names} for snap in snapshots)
        out = finish((step, out.aset("detector_states", states)))
        return out, (inv_eps, sigma, dyn, design, tails, cells, snapshots, out)

    def run_bwd(res, ct):
        _clear_flags(records)
        inv_eps, sigma, dyn, design, tails, cells, snapshots, (_, out) = res
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
        objective_tails = adjoint.objective_tails(out.fields.E, out.fields.H, out.detector_states, snapshots)
        cts = [
            {s: jnp.zeros(v.aval.shape, v.aval.dtype) if isinstance(v, SymbolicZero) else v for s, v in state.items()}
            for state in (ct[1].detector_states[n] for n in names)
        ]
        grad, grad_sigma = adjoint(
            inv_eps,
            sigma,
            [f[:, rows] for f in design],
            cts,
            tails[:, rows],
            [c[np.asarray(rows)] for c in cells],
            objective_tails,
        )
        return grad, grad_sigma, None

    run.defvjp(run_fwd, run_bwd, symbolic_zeros=True)
    return eqx.combine(run(arrays.inv_permittivities, arrays.electric_conductivity, dynamic), cell["static"])
