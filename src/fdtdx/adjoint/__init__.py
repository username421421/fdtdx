"""Reciprocity gradients: the gradient of any figure of merit on phasor monitors, in two forward solves.

Instead of differentiating through the time loop, the backward pass places adjoint currents
at the objective monitors and runs a second forward solve, as Meep does. The scene needs
nothing added: no adjoint source and no design-region detector.

Example, with ``objects, arrays, params, config`` straight from ``fdtdx.place_objects``, a
``PhasorDetector`` named ``"mon"`` and one or more ``Device``::

    import jax
    import jax.numpy as jnp
    from fdtdx.adjoint import reciprocity_param_fn

    param_fn = reciprocity_param_fn(arrays, objects, config, key, objective_detectors="mon")

    def loss(params):
        phasors = param_fn(params)  # (1, num_frequencies, num_components, *cells), as run_fdtd records
        return -jnp.sum(jnp.abs(phasors) ** 2)

    value, grad = jax.value_and_grad(loss)(params)  # grad is a ParameterContainer, like params

The gradient equals ``jax.grad`` of ``apply_params`` then ``run_fdtd`` with
``GradientConfig("checkpointed")`` once the fields have decayed (``param_fn.diagnostics`` and a
:class:`ConvergenceWarning` report when they have not). Keyword arguments to ``param_fn`` go to
``apply_params`` (``param_fn(params, beta=beta)``). :func:`reciprocity_phasor_fn` is the same one
level down, as a function of ``inv_permittivities``. Configurations it would get wrong raise at
setup (:mod:`fdtdx.adjoint.validation`).

Modules: :mod:`~fdtdx.adjoint.api` (entry points), :mod:`~fdtdx.adjoint.vjp` (the custom VJP),
:mod:`~fdtdx.adjoint.objective` (monitor channels and their transposes),
:mod:`~fdtdx.adjoint.design` (design regions and the internal scene),
:mod:`~fdtdx.adjoint.kernel` (window, amplitude solve, gradient kernel, convergence estimate),
:mod:`~fdtdx.adjoint.validation` (refusals). Theory and measurements: ``notes/adjoint/``.
"""

from fdtdx.adjoint.api import ReciprocityParamFn, reciprocity_param_fn, reciprocity_phasor_fn
from fdtdx.adjoint.kernel import ConvergenceWarning, dft_tail, gaussian_window
from fdtdx.adjoint.vjp import ReciprocityPhasorFn

__all__ = [
    "ConvergenceWarning",
    "ReciprocityParamFn",
    "ReciprocityPhasorFn",
    "dft_tail",
    "gaussian_window",
    "reciprocity_param_fn",
    "reciprocity_phasor_fn",
]
