"""Reciprocity gradients: ``run_fdtd`` with ``GradientConfig(method="reciprocity")``.

Instead of differentiating through the time loop, the backward pass places adjoint currents at
the phasor detectors the figure of merit reads and runs one more forward solve, as Meep's
adjoint does, then pairs the two solves' phasors in every Device. It costs about two forward
solves and stores no time history::

    config = config.aset("gradient_config", fdtdx.GradientConfig(method="reciprocity"))

    def loss(params):
        arrays_p, objects_p, _ = fdtdx.apply_params(arrays, objects, params, key)
        _, out = fdtdx.run_fdtd(arrays_p, objects_p, config, key)
        return -jnp.sum(jnp.abs(out.detector_states["mon"]["phasor"]) ** 2)

    value, grad = jax.value_and_grad(loss)(params)

The gradient equals ``GradientConfig(method="checkpointed")``'s once the fields have decayed;
when they have not, it raises instead (``GradientConfig.tail_tolerance``). Configurations it
would get wrong raise when the gradient is traced (:mod:`fdtdx.adjoint.validation`).

Modules: :mod:`~fdtdx.adjoint.reciprocity` (the ``custom_vjp`` behind ``run_fdtd``),
:mod:`~fdtdx.adjoint.solve` (the adjoint solve), :mod:`~fdtdx.adjoint.objective` (monitor
channels, their transposes and adjoint currents), :mod:`~fdtdx.adjoint.design` (design
detectors and the solves' scenes), :mod:`~fdtdx.adjoint.kernel` (amplitude solve, gradient
kernel, convergence estimate), :mod:`~fdtdx.adjoint.validation` (refusals),
:mod:`~fdtdx.adjoint.source` (the adjoint current). Theory and measurements: ``notes/adjoint/``.
"""

from fdtdx.adjoint.reciprocity import reciprocity_fdtd

__all__ = ["reciprocity_fdtd"]
