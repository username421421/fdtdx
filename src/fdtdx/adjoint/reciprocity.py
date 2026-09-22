"""Pieces of the reciprocity gradient: window, amplitude solve, gradient kernel.

The reciprocity route computes a photonic design gradient with two forward FDTD
solves instead of differentiating through the time loop:

  1. forward run  -> objective phasors ``P`` and design-region phasors ``F``
  2. JAX hands the VJP a cotangent ``ct_P`` for ``P``
  3. build an adjoint current whose DFT equals ``ct_P``
  4. adjoint run  -> design-region phasors ``Lambda``
  5. combine ``Lambda`` and ``F`` pointwise into the material gradient

Steps 3 and 5 live here. Both were calibrated against ``checkpointed`` autodiff
rather than trusted from a derivation, and both had a trap worth recording.

Step 3 has two independent traps:

* A constant-amplitude adjoint source never stops radiating, so its DFT over the
  simulation window does not converge and the reconstructed gradient is wrong by
  order one. The excitation has to be a decaying pulse. Measured, not assumed.
* Injecting a real waveform ``Re[a e^{+i w t}]`` does **not** produce a DFT of
  ``a``. Under the detector's ``e^{+i w t}`` accumulation it gives roughly
  ``(T/2) conj(a)``, plus leakage between frequencies spaced closer than ``1/T``.

:func:`solve_adjoint_amplitudes` handles both: given a window it solves exactly
for the sinusoid coefficients whose windowed sum has the requested DFT at every
objective frequency. The system is square (``2 nf`` real equations in ``2 nf``
real unknowns), the matrix is shared by every spatial site, and the conditioning
and residual are reported so a bad window is loud rather than silent. This is the
same problem Meep solves with ``FilteredSource``.

Step 5's trap is geometric, not algebraic: see
:func:`assemble_material_gradient`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def gaussian_window(
    time_steps_total: int,
    center_frac: float = 0.22,
    sigma_frac: float = 0.055,
    dtype=jnp.float64,
) -> jax.Array:
    """Smooth pulse envelope on ``[0, time_steps_total)``.

    The defaults put the peak about a fifth of the way in and leave roughly the
    back two thirds of the run for the excited field to leave the domain, which
    is what the adjoint DFT needs in order to converge.

    Args:
        time_steps_total: length of the returned array.
        center_frac: peak position as a fraction of the run.
        sigma_frac: Gaussian sigma as a fraction of the run. It must span several
            optical periods or the amplitude solve goes ill-conditioned, which
            :func:`solve_adjoint_amplitudes` reports.
        dtype: output dtype.

    Returns:
        The envelope, shape ``(time_steps_total,)``.
    """
    n = jnp.arange(time_steps_total, dtype=dtype)
    center = center_frac * time_steps_total
    sigma = sigma_frac * time_steps_total
    return jnp.exp(-0.5 * ((n - center) / sigma) ** 2)


def _design_matrix(angular_frequencies: np.ndarray, dt: float, window: np.ndarray) -> np.ndarray:
    """Real ``(2 nf, 2 nf)`` map from sinusoid coefficients to the phasor DFT.

    Columns are ``[alpha_0..alpha_{nf-1}, beta_0..beta_{nf-1}]`` for the waveform
    ``sum_g w_n (alpha_g cos(w_g t_n) - beta_g sin(w_g t_n))``. Rows are the real
    and imaginary parts of ``sum_n e^{+i w_f t_n} J(t_n)``.
    """
    n = np.arange(len(window))
    t = n * dt
    phase = angular_frequencies[:, None] * t[None, :]
    kern = np.exp(1j * phase)
    C = kern @ (window[None, :] * np.cos(phase)).T
    S = kern @ (window[None, :] * np.sin(phase)).T
    top = np.concatenate([C.real, -S.real], axis=1)
    bot = np.concatenate([C.imag, -S.imag], axis=1)
    return np.concatenate([top, bot], axis=0)


def solve_adjoint_amplitudes(
    target_phasors: jax.Array,
    angular_frequencies: tuple[float, ...],
    dt: float,
    window: jax.Array,
    cond_limit: float = 1e8,
) -> tuple[jax.Array, dict[str, float]]:
    """Sinusoid amplitudes whose windowed sum has DFT ``target_phasors``.

    Args:
        target_phasors: requested DFT, shape ``(nf, nc, *grid)``, complex.
        angular_frequencies: the ``nf`` objective frequencies, rad/s.
        dt: ``config.time_step_duration``.
        window: envelope from :func:`gaussian_window`, shape ``(T,)``.
        cond_limit: raise if the design matrix is worse conditioned than this. A
            bad condition number means the window is too short to separate the
            requested frequencies, so the amplitudes would be meaningless.

    Returns:
        ``(amplitudes, diagnostics)``. ``amplitudes`` matches ``target_phasors``'
        shape and feeds :class:`fdtdx.objects.sources.adjoint.AdjointCurrentSource`.
        ``diagnostics`` carries ``cond`` and ``residual``; the residual should sit
        at machine precision because the system is square.

    Raises:
        ValueError: on a shape mismatch, or if the solve is ill-conditioned.
    """
    omega = np.asarray(angular_frequencies, dtype=np.float64)
    nf = len(omega)
    if target_phasors.shape[0] != nf:
        raise ValueError(f"target_phasors.shape[0]={target_phasors.shape[0]} != nf={nf}")

    w = np.asarray(jax.device_get(window), dtype=np.float64)
    A = _design_matrix(omega, float(dt), w)
    cond = float(np.linalg.cond(A))
    if not np.isfinite(cond) or cond > cond_limit:
        raise ValueError(
            f"Adjoint amplitude solve is ill-conditioned (cond={cond:.3e} > {cond_limit:.1e}). "
            "The window cannot resolve the requested frequencies: widen sigma_frac, lengthen the "
            "simulation, or space the objective frequencies further apart."
        )

    target = np.asarray(jax.device_get(target_phasors))
    tail = target.shape[1:]
    flat = target.reshape(nf, -1)
    rhs = np.concatenate([flat.real, flat.imag], axis=0)
    sol = np.linalg.solve(A, rhs)
    amplitudes = (sol[:nf] + 1j * sol[nf:]).reshape(nf, *tail)

    residual = float(np.linalg.norm(A @ sol - rhs) / (np.linalg.norm(rhs) + 1e-300))
    return jnp.asarray(amplitudes), {"cond": cond, "residual": residual}


def leapfrog_kernel(
    angular_frequencies: tuple[float, ...],
    dt: float,
    courant_number: float,
) -> np.ndarray:
    """Per-frequency factor relating the adjoint-forward overlap to the gradient.

    Perturbing ``inv_eps`` inside FDTDX's own update

        E_{n+1} = E_n + courant * inv_eps * curl_H

    moves the field by ``d(inv_eps) * courant * curl_H``, which equals
    ``d(inv_eps) * (E_{n+1} - E_n) / inv_eps``. The detector records the field
    *after* each step, so with ``F = sum_n exp(+i w t_n) E_{n+1}`` the increment
    transforms as ``sum_n exp(+i w t_n) (E_{n+1} - E_n) = (1 - exp(+i w dt)) F``,
    and the source injection law supplies the remaining ``-1 / courant``.

    The result is the leapfrog's own stand-in for the textbook continuum ``i w``;
    they agree only as ``w dt -> 0``, so the continuum factor degrades agreement
    at coarse resolution and near Nyquist.

    Calibration: fitting a free complex constant to this expression against
    ``checkpointed`` autodiff on a separated-geometry scene gave
    ``-1.7492 - 0.0006j`` times ``(1 - exp(+i w dt))``, against the predicted
    ``-1 / courant = -1.74954``. That is agreement to the run's DFT truncation
    floor, so the form below is used with no fitted constant.
    """
    omega = np.asarray(angular_frequencies, dtype=np.float64)
    return (np.exp(1j * omega * dt) - 1.0) / courant_number


def assemble_material_gradient(
    adjoint_phasors: jax.Array,
    forward_phasors: jax.Array,
    inv_permittivities: jax.Array,
    angular_frequencies: tuple[float, ...],
    dt: float,
    courant_number: float,
    scale: complex | jax.Array = 1.0,
) -> jax.Array:
    """Gradient with respect to ``inv_permittivities`` over the design region.

    Args:
        adjoint_phasors: adjoint-run design phasors, ``(1, nf, nc, *grid)``.
        forward_phasors: forward-run design phasors, same shape.
        inv_permittivities: design-region slice of the inverse permittivity.
        angular_frequencies: the ``nf`` objective frequencies, rad/s.
        dt: ``config.time_step_duration``.
        courant_number: ``config.courant_number``.
        scale: optional extra factor; leave at 1 for the calibrated kernel.

    Returns:
        Real gradient shaped like ``inv_permittivities``.

    Notes:
        The component axis is **summed** when ``inv_permittivities`` is isotropic
        (FDTDX stores that as ``(1, Nx, Ny, Nz)``): one scalar per cell drives all
        three E components, so perturbing it moves every component at once.

        **The design region must not contain a source.** FDTDX sources scale their
        injected amplitude by the local ``inv_eps``, which contributes a gradient
        term this expression does not model. One offending cell is enough to
        dominate the gradient norm, so the symptom is a global-looking failure
        rather than a single bad cell: during development a dipole sitting inside
        the design region held the relative residual at 0.86 across every
        candidate kernel, and separating the geometry dropped it to 3e-4.
    """
    lam = adjoint_phasors[0]
    fwd = forward_phasors[0]
    if inv_permittivities.shape[0] == 1:
        overlap = jnp.sum(lam * fwd, axis=1, keepdims=True)
    else:
        overlap = lam * fwd

    kern = jnp.asarray(leapfrog_kernel(angular_frequencies, dt, courant_number)) * jnp.asarray(scale)
    kern = kern.reshape((-1,) + (1,) * (overlap.ndim - 1))
    eps_sq = 1.0 / (inv_permittivities**2)
    return jnp.real(jnp.sum(kern * overlap, axis=0)) * eps_sq
