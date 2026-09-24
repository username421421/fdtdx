"""The numerical kernel: adjoint window, amplitude solve, gradient kernel, convergence estimate.

Measurements behind every constant and factor here are in ``notes/adjoint/02-implementation.md``
and ``notes/adjoint/05-guards.md``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.adjoint.validation import check_conditioning
from fdtdx.constants import eta0

#: Ceiling on the condition number of the amplitude-solve matrix. The float32 solve error is
#: about ``1e-8 * cond``, so 1e4 keeps it near 1e-4.
DEFAULT_COND_LIMIT = 1e4

#: :func:`dft_tail` above which the phasors are reported as unconverged. The gradient error
#: measured 2-5x the largest tail, so 1e-2 flags errors of a few percent.
DEFAULT_TAIL_TOLERANCE = 1e-2


class ConvergenceWarning(UserWarning):
    """The phasors a reciprocity gradient is built from have not converged (see :func:`dft_tail`)."""


class PmlWarning(UserWarning):
    """Part of an objective's adjoint current lies in a PML, where the reciprocal pairing does not hold."""


def gaussian_window(
    time_steps_total: int,
    center_frac: float = 0.22,
    sigma_frac: float = 0.055,
    dtype=jnp.float64,
) -> jax.Array:
    """Envelope of the adjoint excitation, shape ``(time_steps_total,)``.

    A decaying pulse, not a CW source, because reciprocity holds only once both DFTs
    have converged. The defaults peak a fifth of the way in and leave the back two
    thirds of the run for the field to leave the domain. ``sigma_frac`` must span
    several optical periods, or the amplitude solve becomes ill-conditioned.
    """
    n = jnp.arange(time_steps_total, dtype=dtype)
    center = center_frac * time_steps_total
    sigma = sigma_frac * time_steps_total
    return jnp.exp(-0.5 * ((n - center) / sigma) ** 2)


def amplitude_matrix(
    angular_frequencies: Sequence[float],
    dt: float,
    window: jax.Array,
    cond_limit: float = DEFAULT_COND_LIMIT,
) -> tuple[np.ndarray, float]:
    """The real ``(2 nf, 2 nf)`` map from sinusoid coefficients to the windowed DFT, and its ``cond``.

    Columns are ``[alpha_0.., beta_0..]`` of ``sum_g w_n (alpha_g cos(w_g t_n) - beta_g sin(w_g t_n))``;
    rows are the real and imaginary parts of ``sum_n exp(+i w_f t_n) J(t_n)``. Injecting
    ``Re[a exp(+i w t)]`` does not give a DFT of ``a``, so the amplitudes are solved for.

    Raises:
        ValueError: if ``cond`` exceeds ``cond_limit``.
    """
    omega = np.asarray(angular_frequencies, dtype=np.float64)
    w = np.asarray(jax.device_get(window), dtype=np.float64)
    phase = omega[:, None] * (np.arange(len(w)) * float(dt))[None, :]
    kern = np.exp(1j * phase)
    C = kern @ (w[None, :] * np.cos(phase)).T
    S = kern @ (w[None, :] * np.sin(phase)).T
    matrix = np.block([[C.real, -S.real], [C.imag, -S.imag]])
    cond = float(np.linalg.cond(matrix))
    check_conditioning(cond, cond_limit)
    return matrix, cond


def solve_amplitudes(matrix, target, xp=jnp):
    """Amplitudes ``(nf, nc, *cells)`` whose windowed sum has DFT ``target``; trace-safe for ``xp=jnp``."""
    nf = target.shape[0]
    flat = target.reshape(nf, -1)
    rhs = xp.concatenate([xp.real(flat), xp.imag(flat)], axis=0)
    sol = xp.linalg.solve(matrix, rhs)
    return (sol[:nf] + 1j * sol[nf:]).reshape(target.shape)


def solve_adjoint_amplitudes(
    target_phasors: jax.Array,
    angular_frequencies: tuple[float, ...],
    dt: float,
    window: jax.Array,
    cond_limit: float = DEFAULT_COND_LIMIT,
) -> tuple[jax.Array, dict[str, float]]:
    """Amplitudes for an :class:`~fdtdx.objects.sources.adjoint.AdjointCurrentSource` with DFT ``target_phasors``.

    Args:
        target_phasors: requested DFT, ``(nf, nc, *cells)``, complex.
        angular_frequencies: the ``nf`` frequencies, rad/s.
        dt: ``config.time_step_duration``.
        window: the source's envelope, e.g. :func:`gaussian_window`.
        cond_limit: see :func:`amplitude_matrix`.

    Returns:
        ``(amplitudes, {"cond": ..., "residual": ...})``, solved in float64.
    """
    nf = len(angular_frequencies)
    if target_phasors.shape[0] != nf:
        raise ValueError(f"target_phasors.shape[0]={target_phasors.shape[0]} != nf={nf}")
    matrix, cond = amplitude_matrix(angular_frequencies, dt, window, cond_limit)
    target = np.asarray(jax.device_get(target_phasors))
    amplitudes = solve_amplitudes(matrix, target, xp=np)

    def packed(z):
        return np.concatenate([z.real, z.imag]).reshape(2 * nf, -1)

    rhs = packed(target)
    residual = float(np.linalg.norm(matrix @ packed(amplitudes) - rhs) / (np.linalg.norm(rhs) + 1e-300))
    return jnp.asarray(amplitudes), {"cond": cond, "residual": residual}


def leapfrog_kernel(angular_frequencies: Sequence[float], dt: float, courant_number: float) -> np.ndarray:
    """``(exp(+i w dt) - 1) / courant``: the leapfrog's own stand-in for the continuum ``i w``.

    Perturbing ``inv_eps`` in ``E_{n+1} = E_n + courant * inv_eps * curl_H`` moves the field
    by ``d(inv_eps) (E_{n+1} - E_n) / inv_eps``, whose DFT is ``(1 - exp(+i w dt)) F`` for the
    post-step phasor ``F``; the injection law supplies ``-1 / courant``. The continuum factor
    gives the wrong sign at the ``w dt`` FDTD runs at.
    """
    omega = np.asarray(angular_frequencies, dtype=np.float64)
    return (np.exp(1j * omega * dt) - 1.0) / courant_number


def conductivity_kernel(angular_frequencies: Sequence[float], dt: float) -> np.ndarray:
    """``eta0 (1 + exp(+i w dt)) / 2``: the conductivity's counterpart of :func:`leapfrog_kernel`.

    ``update_E`` steps ``E_{n+1} = ((1 - a) E_n + courant * inv_eps * curl_H) / (1 + a)``,
    ``a = courant * sigma * eta0 * inv_eps / 2``, so a perturbation moves the field by
    ``d(inv_eps) (E_{n+1} - E_n) / (inv_eps (1 + a))`` or ``-d(sigma) courant eta0 inv_eps
    (E_{n+1} + E_n) / (2 (1 + a))``: the time sum ``(1 + exp(+i w dt)) F`` replaces the difference,
    and ``inv_eps`` cancels. ``sigma`` in FDTDX's stored units (conductivity times grid spacing).
    """
    omega = np.asarray(angular_frequencies, dtype=np.float64)
    return float(eta0) * (1.0 + np.exp(1j * omega * dt)) / 2.0


def _pair(adjoint_phasors: jax.Array, forward_phasors: jax.Array, kern: np.ndarray, rows: int) -> jax.Array:
    """``Re[sum_f kern(w_f) sum_c Lambda_c F_c]``, components summed for a one-row (isotropic) material."""
    lam = adjoint_phasors[0]
    fwd = forward_phasors[0]
    overlap = jnp.sum(lam * fwd, axis=1, keepdims=True) if rows == 1 else lam * fwd
    kern_b = jnp.asarray(kern).reshape((-1,) + (1,) * (overlap.ndim - 1))
    return jnp.real(jnp.sum(kern_b * overlap, axis=0))


def assemble_material_gradient(
    adjoint_phasors: jax.Array,
    forward_phasors: jax.Array,
    inv_permittivities: jax.Array,
    angular_frequencies: Sequence[float],
    dt: float,
    courant_number: float,
) -> jax.Array:
    """``d loss / d inv_permittivities`` on one design region.

    ``Re[sum_f K(w_f) sum_c Lambda_c F_c] / inv_eps^2`` with ``K`` the :func:`leapfrog_kernel`.
    The component axis is summed for an isotropic ``(1, ...)`` permittivity, since one value
    per cell drives all three E components. Exact with loss in the cells too: the ``1 + a`` of
    the lossy update cancels against the lossy injection of the reciprocal source.

    Args:
        adjoint_phasors: adjoint-solve design phasors, ``(1, nf, 3, *cells)``, raw scale.
        forward_phasors: forward-solve design phasors, same shape.
        inv_permittivities: the region's slice, ``(1 | 3, *cells)``.
        angular_frequencies: the ``nf`` frequencies, rad/s.
        dt: ``config.time_step_duration``.
        courant_number: ``config.courant_number``.

    Returns:
        Real gradient shaped like ``inv_permittivities``.
    """
    kern = leapfrog_kernel(angular_frequencies, dt, courant_number)
    eps_sq = 1.0 / (inv_permittivities**2)
    return _pair(adjoint_phasors, forward_phasors, kern, inv_permittivities.shape[0]) * eps_sq


def assemble_conductivity_gradient(
    adjoint_phasors: jax.Array,
    forward_phasors: jax.Array,
    rows: int,
    angular_frequencies: Sequence[float],
    dt: float,
) -> jax.Array:
    """``d loss / d electric_conductivity`` on one design region, ``(rows, *cells)``, :func:`conductivity_kernel`."""
    return _pair(adjoint_phasors, forward_phasors, conductivity_kernel(angular_frequencies, dt), rows)


def dft_tail(
    fields: jax.Array,
    phasors: jax.Array,
    angular_frequencies: Sequence[float],
    dt: float,
    rel_floor: float = 1e-3,
) -> jax.Array:
    """Estimated DFT truncation error of ``phasors``, relative to their size.

    ``eta(w) = ||fields|| / (|1 - exp(i w dt)| ||phasors(w)||)``: the geometric tail the DFT
    misses if the field left at the last step, ``fields`` ``(nc, *cells)``, neither decayed
    nor oscillated, against the raw-scale ``phasors`` ``(nf, nc, *cells)``. An estimate, not a
    bound. Frequencies whose phasor is below ``rel_floor`` times the largest are skipped.

    Returns:
        The largest ``eta(w)``.
    """
    w = jnp.asarray(np.asarray(angular_frequencies, dtype=np.float64))
    per_freq = jnp.sqrt(jnp.sum(jnp.abs(phasors.reshape(phasors.shape[0], -1)) ** 2, axis=1))
    left = jnp.sqrt(jnp.sum(jnp.abs(fields) ** 2))
    gap = jnp.abs(1.0 - jnp.exp(1j * w * dt)).astype(per_freq.dtype)
    kept = per_freq >= rel_floor * jnp.max(per_freq)
    tiny = jnp.finfo(per_freq.dtype).tiny
    eta = jnp.where(kept, left / (gap * jnp.maximum(per_freq, tiny)), 0.0)
    return jnp.max(eta)


def _silence(category: type[Warning]) -> str:
    return (
        "Silence it with tail_tolerance=None or, under GradientConfig(method='reciprocity'), with "
        f"warnings.filterwarnings('ignore', category=fdtdx.adjoint.{category.__name__})."
    )


_TAIL_WHAT = {
    "objective_tail": "objective phasors",
    "forward_design_tail": "forward design-region phasors",
    "adjoint_design_tail": "adjoint design-region phasors",
}


class TailReport:
    """Receives :func:`dft_tail` values from host callbacks, records them, warns once per stage."""

    def __init__(self, diagnostics: dict, tolerance: float | None):
        self.diagnostics = diagnostics
        self.tolerance = tolerance
        self.warned: set[str] = set()

    def __call__(self, stage: str, labels: tuple[str, ...], values) -> None:
        tails = {label: float(v) for label, v in zip(labels, values)}
        self.diagnostics[stage] = tails
        if self.tolerance is None or stage in self.warned:
            return
        worst = max(tails, key=lambda k: tails[k])
        if tails[worst] > self.tolerance:
            self.warned.add(stage)
            if stage == "objective_pml_share":
                warnings.warn(
                    f"A share {tails[worst]:.1e} of the adjoint current for {worst!r}, weighted by the local PML "
                    "strength, lies inside a PML, where the reciprocal pairing does not hold; the gradient error "
                    "is of that order or below. Keep objective monitors out of the PML (its first cell is "
                    f"harmless). {_silence(PmlWarning)}",
                    PmlWarning,
                    stacklevel=2,
                )
                return
            if stage == "objective_tail":
                meaning = "The returned phasors, and a figure of merit on them, carry an error of about that size."
            else:
                meaning = "The gradient is exact only once both solves have converged; its error was 2-5x this."
            listed = ", ".join(f"{k}={v:.2e}" for k, v in tails.items())
            warnings.warn(
                f"The {_TAIL_WHAT[stage]} have not converged: DFT tail estimate {tails[worst]:.2e} for {worst!r} exceeds "
                f"tail_tolerance={self.tolerance:.1e} (all: {listed}). {meaning} Lengthen the simulation so "
                f"the fields leave the domain. {_silence(ConvergenceWarning)}",
                ConvergenceWarning,
                stacklevel=2,
            )
