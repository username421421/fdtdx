"""Reciprocity-based gradients: place an adjoint source, run a second forward solve.

FDTDX's built-in gradients differentiate through the time loop, which on an
RTX 3080 Ti costs about 8.9x a forward solve for the reversible path and 52x for
the checkpointed one. Reciprocity replaces that with a second forward solve, the
way Meep does, measured at 2.56x.

Start with :func:`reciprocity_param_fn` if you are optimizing a design, or
:func:`reciprocity_phasor_fn` if you want to differentiate raw permittivities.

See ``notes/adjoint/01-reciprocity-physics.md`` for the theory and
``notes/adjoint/02-implementation.md`` for the measured discrete factors and the
traps behind them.
"""

from fdtdx.adjoint.api import (
    ReciprocityParamFn,
    apply_objects_once,
    design_region_slice,
    reciprocity_param_fn,
    reciprocity_phasor_fn,
)
from fdtdx.adjoint.reciprocity import (
    assemble_material_gradient,
    gaussian_window,
    leapfrog_kernel,
    solve_adjoint_amplitudes,
)
from fdtdx.adjoint.scene import derive_adjoint_objects
from fdtdx.adjoint.vjp import make_reciprocity_phasor_fn

__all__ = [
    "ReciprocityParamFn",
    "apply_objects_once",
    "assemble_material_gradient",
    "derive_adjoint_objects",
    "design_region_slice",
    "gaussian_window",
    "leapfrog_kernel",
    "make_reciprocity_phasor_fn",
    "reciprocity_param_fn",
    "reciprocity_phasor_fn",
    "solve_adjoint_amplitudes",
]
