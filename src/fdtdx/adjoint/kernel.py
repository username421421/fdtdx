"""The numerical kernel: adjoint window, amplitude solve, gradient kernel, convergence estimate.

Measurements behind every constant and factor here are in ``notes/adjoint/02-implementation.md``
and ``notes/adjoint/05-guards.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.adjoint.validation import check_conditioning
from fdtdx.constants import eta0

#: Ceiling on the condition number of the amplitude-solve matrix. The float32 solve error is
#: about ``1e-8 * cond``, so 1e4 keeps it near 1e-4.
DEFAULT_COND_LIMIT = 1e4

#: :func:`gradient_error_estimate` above which the gradient is refused.
DEFAULT_TAIL_TOLERANCE = 1e-2


def gaussian_window(
    time_steps_total: int,
    center_frac: float = 0.30,
    sigma_frac: float = 0.055,
    dtype=jnp.float64,
) -> jax.Array:
    """Envelope of the adjoint excitation, shape ``(time_steps_total,)``.

    A decaying pulse, not a CW source, because reciprocity holds only once both DFTs
    have converged. The defaults peak 30% of the way in, where the envelope starts at
    ``exp(-14.9)`` of its peak (at 22% it started at ``exp(-8)``, a DC step that never
    decays: 2e-2 on a weak frequency, 1.5e-5 at 30%), and leave the back two thirds of
    the run for the field to leave the domain. ``sigma_frac`` must span several optical
    periods, or the amplitude solve becomes ill-conditioned.
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


def row_norms(x: jax.Array) -> jax.Array:
    """``||x||`` over all but the leading axis, scaled before squaring so float32 cannot underflow."""
    a = jnp.abs(x.reshape(x.shape[0], -1))
    peak = jnp.max(a, axis=1)
    safe = jnp.where(peak > 0, peak, 1.0)
    return jnp.sqrt(jnp.sum((a / safe[:, None]) ** 2, axis=1)) * peak


#: A late window's ratio is kept this far from 1 (a field that does not decay, on resonance),
#: which extrapolates at most 1000 windows further.
_MAX_LATE_RATIO = 0.999

#: Late content below this fraction of a phasor is roundoff, not a tail.
_LATE_FLOOR = 1e-6

#: The windows are fitted with two modes only with at least this many entries per frequency (cells
#: times components) and first two windows this far from collinear; otherwise with one.
_TWO_MODE_ENTRIES = 8
_COLLINEAR = 1e-10

#: Windows this close to collinear (``||b_perp||^2 / ||b||^2``) also take the one-mode estimate, with
#: its counter-rotating shift: a standing wave on resonance plus a 1e-3 remainder elsewhere switched
#: to the two-mode fit, which mixed the wave's two rotating parts into one mode (0.28 of the tail).
_NEAR_COLLINEAR = 1e-3

#: A last window this many times larger than both earlier ones, or than the middle one while larger
#: than the first, is a field still arriving (an echo from far away, a pulse's front), whose size no
#: continuation can tell: refused. Continued as a series, an echo reaching the monitor in the last
#: window read 4.5e-5 where the figure of merit was 89% off; with a Device's decaying ringing in the
#: first window (n0 = n2 / 9.5), the second test caught what the first missed (80% off). The stock
#: PML's own weak reflection trips it too, in runs converged to 1e-5: the front's size says nothing
#: of the echo behind it.
_ARRIVING = 10.0


def _continued(z: jax.Array) -> jax.Array:
    """``z^3 / (1 - z)``: a mode's windows after the third, summed; ``|1 - z|`` held at 1e-3 or more."""
    gap = 1.0 - z
    size = jnp.abs(gap)
    held = jnp.where(size > 0, gap / jnp.where(size > 0, size, 1.0), 1.0) * (1.0 - _MAX_LATE_RATIO)
    return z**3 / jnp.where(size > 1.0 - _MAX_LATE_RATIO, gap, held)


def _series(z: jax.Array) -> jax.Array:
    """``|z / (1 - z)|``: a mode's windows after the last, over the last; ``|1 - z|`` held at 1e-3 or more.

    Not held below 1 in ``|z|``: on few cells beating modes make a window grow (``|z| > 1``), and
    holding that as a field that does not decay read an objective tail of 6 where the figure of
    merit was 6e-3 off, and 0.05 at 3e-5.
    """
    return jnp.abs(z) / jnp.maximum(jnp.abs(1.0 - z), 1.0 - _MAX_LATE_RATIO)


def dft_tail(
    fields: jax.Array,
    phasors: jax.Array,
    late: tuple[jax.Array, jax.Array, jax.Array],
    angular_frequencies: Sequence[float],
    dt: float,
) -> jax.Array:
    """Estimated DFT truncation error of ``phasors`` per frequency, relative to their size, ``(nf,)``.

    See :func:`dft_tail_cells`, which also gives it cell by cell.
    """
    return dft_tail_cells(fields, phasors, late, angular_frequencies, dt)[0]


def dft_tail_cells(
    fields: jax.Array,
    phasors: jax.Array,
    late: tuple[jax.Array, jax.Array, jax.Array],
    angular_frequencies: Sequence[float],
    dt: float,
) -> tuple[jax.Array, jax.Array]:
    """Estimated DFT truncation error of ``phasors``: relative per frequency, ``(nf,)``, and its size per
    cell, over the components, ``(nf, *cells)`` (raw scale).

    Args:
        fields: the recorded fields at the last step, ``(nc, *cells)``.
        phasors: their raw-scale phasors over the run, ``(nf, nc, *cells)``.
        late: the raw-scale phasors' growth over the last three windows of the run
            (:func:`~fdtdx.adjoint.design.late_windows`), oldest first, each shaped like ``phasors``.
        angular_frequencies: the ``nf`` frequencies, rad/s.
        dt: the time step.

    Two estimates of the part of the transform the run did not see, the larger taken: a static
    remainder, ``||fields|| / |1 - exp(i w dt)|``, and ringing, the windows ``D0, D1, D2``
    continued. A decaying mode grows the phasor by ``v z^k`` over window ``k``. Two modes are fitted
    over the cells (``D2 = -(a0 D0 + a1 D1)`` by least squares, ``z^2 + a1 z + a0 = 0``) and each is
    summed on, ``v z^3 / (1 - z)``; what they leave of ``D2`` is continued like the slowest of them
    and of the last pair's ratio ``z = <D1, D2> / <D1, D1>``, ``|z / (1 - z)|``. (The windows'
    magnitude ratio instead, which assumes no cancellation, read a periodic slab's guided mode that
    does not decay as an objective tail of 0.26 on a figure of merit converged to 1e-5.) With fewer
    than ``_TWO_MODE_ENTRIES`` entries (a one-cell monitor), or collinear windows (one mode, or a
    standing wave's two rotating parts), one mode: ``||D2|| |z| / (|1 - z| - s)``, ``s`` the shift
    of ``z`` the counter-rotating part's boundary terms can make; nearly collinear windows also take
    it. A last window more than 10 times larger than both earlier ones is a field still arriving:
    infinite. An estimate, not a bound: a mode whose windows are hidden under stronger ones is not
    seen (a weak resonance of Q far beyond the run under two faster modes over cells read 0.05 of its
    tail; on one cell, where a real field's two rotating parts already make two modes, under one
    fast detuned mode 0.03, and a floor at the windows' magnitude ratio read single detuned modes up
    to 15x high without catching it). Measured
    against the phasors of runs long enough to converge (notes/adjoint/05-guards.md), true over
    estimated: a single mode off resonance 0.96 to 1.09, an eps-12 Device's mixture of modes 0.72 to
    1.35 (the one-mode estimate 1.5 to 6.7 there), a periodic slab's guided mode 0.62 to 1.01.
    """
    w = jnp.asarray(np.asarray(angular_frequencies, dtype=np.float64))
    full = row_norms(phasors)
    tiny = jnp.finfo(full.dtype).tiny
    gap = jnp.abs(1.0 - jnp.exp(1j * w * dt)).astype(full.dtype)
    static = row_norms(fields[None])[0] / gap
    # scaled rows, so float32 cannot underflow
    d = [x.reshape(x.shape[0], -1) for x in late]
    scale = jnp.max(jnp.stack([jnp.max(jnp.abs(x), axis=1) for x in d]), axis=0)
    scale = jnp.where(scale > 0, scale, 1.0)
    a, b, c = (x / scale[:, None] for x in d)

    def dot(x, y):
        return jnp.sum(jnp.conj(x) * y, axis=1)

    def norm(x):
        return jnp.sqrt(jnp.sum(jnp.abs(x) ** 2, axis=1))

    g00, g11 = jnp.real(dot(a, a)), jnp.real(dot(b, b))
    n0, n2 = jnp.sqrt(g00), norm(c)
    z = dot(b, c) / jnp.where(g11 > 0, g11, 1.0)
    n1 = jnp.sqrt(g11)
    arriving = (n2 > _ARRIVING * jnp.maximum(n0, n1)) | ((n2 > _ARRIVING * n1) & (n2 > n0))

    # two modes: c = -(a0 a + a1 b) by least squares, by modified Gram-Schmidt on a and b (in float32
    # the Gram determinant lost a slow mode under a fast one to cancellation, 0.09 of its tail; one
    # projection still let the fast part of c leak into the slow coefficient, 0.39), then
    # z^2 + a1 z + a0 = 0
    inv00 = 1.0 / jnp.where(g00 > 0, g00, 1.0)
    p = dot(a, b) * inv00
    b_perp = b - p[:, None] * a
    p2 = dot(a, b_perp) * inv00
    b_perp = b_perp - p2[:, None] * a
    gpp = jnp.real(dot(b_perp, b_perp))
    alpha = dot(a, c) * inv00
    c_perp = c - alpha[:, None] * a
    beta = dot(b_perp, c_perp) / jnp.where(gpp > 0, gpp, 1.0)
    a0, a1 = -(alpha - beta * (p + p2)), -beta
    root = jnp.sqrt(a1 * a1 - 4.0 * a0 + 0j)
    z1, z2 = (-a1 + root) / 2.0, (-a1 - root) / 2.0
    two = (
        (a.shape[1] >= _TWO_MODE_ENTRIES)
        & (gpp > _COLLINEAR * g11)
        & (jnp.abs(root) > _COLLINEAR * (1.0 + jnp.abs(a1)))
    )
    split = jnp.where(two, root, 1.0)[:, None]
    v1, v2 = (b - z2[:, None] * a) / split, (z1[:, None] * a - b) / split
    modes_each = v1 * _continued(z1)[:, None] + v2 * _continued(z2)[:, None]
    rest_each = c_perp - beta[:, None] * b_perp
    slowest = jnp.maximum(jnp.maximum(_series(z1), _series(z2)), _series(z))
    two_modes = norm(modes_each) + norm(rest_each) * slowest

    # one mode. A real field's counter-rotating part adds a boundary term of about the static
    # estimate's size to every window, which moves z by up to `shift`; near resonance |1 - z| is as
    # small, and the pessimistic |1 - z| - shift is taken (a standing wave on resonance, decaying by
    # 1% per window, read 3.7x low). Windows no bigger than that hold no ringing to extrapolate (a
    # static remainder), and z is taken as it is.
    size_1 = n1 * scale
    shift = 0.5 * (1.0 + jnp.abs(z)) * static / jnp.where(size_1 > 0, size_1, 1.0)
    shift = jnp.where(shift < 0.5, shift, 0.0)
    one_series = jnp.abs(z) / jnp.maximum(jnp.abs(1.0 - z) - shift, 1.0 - _MAX_LATE_RATIO)
    one_mode = n2 * one_series
    near = gpp <= _NEAR_COLLINEAR * g11

    ringing = jnp.where(two, jnp.where(near, jnp.maximum(two_modes, one_mode), two_modes), one_mode) * scale
    late_2 = n2 * scale
    rings = late_2 > _LATE_FLOOR * full
    ringing = jnp.where(rings, jnp.where(arriving, jnp.inf, ringing), late_2)
    tail = jnp.maximum(static, ringing)
    relative = jnp.where(full > 0, tail / jnp.maximum(full, tiny), jnp.where(tail > 0, jnp.inf, 0.0))

    # the same, entry by entry: the modes' continuation, the rest's size times its continuation
    each_two = jnp.abs(modes_each) + jnp.abs(rest_each) * slowest[:, None]
    each_one = jnp.abs(c) * one_series[:, None]
    each_two = jnp.where(near[:, None], jnp.maximum(each_two, each_one), each_two)
    each = jnp.where(two[:, None], each_two, each_one)
    each = jnp.where(rings[:, None], jnp.where(arriving[:, None], jnp.inf, each), jnp.abs(c)) * scale[:, None]
    each_static = jnp.abs(fields.reshape(1, -1)) / gap[:, None]
    each = jnp.maximum(each, each_static).reshape(phasors.shape)
    return relative, component_norms(each)


def component_norms(x: jax.Array) -> jax.Array:
    """``||x||`` over axis 1 (the components), ``(n, *cells)`` from ``(n, nc, *cells)``, scaled per leading
    row before squaring so float32 cannot underflow (unscaled, adjoint phasors of 1e-21 read 0)."""
    size = jnp.abs(x)
    peak = jnp.max(size.reshape(size.shape[0], -1), axis=1)
    safe = jnp.where(peak > 0, peak, 1.0).reshape((-1,) + (1,) * (size.ndim - 1))
    return jnp.sqrt(jnp.sum((size / safe) ** 2, axis=1)) * safe[:, 0]


def truncation_norms(
    adjoint_phasors: jax.Array,
    forward_phasors: jax.Array,
    adjoint_tails: jax.Array,
    forward_tails: jax.Array,
    inv_permittivities: jax.Array,
    kern: np.ndarray,
) -> jax.Array:
    """``|| |K(w_f)| (|dF| |Lambda| + |F| |dLambda|) / inv_eps^2 ||`` per frequency over one region, ``(nf,)``.

    The gradient term's truncation error cell by cell, over the components (Cauchy-Schwarz), from
    the design phasors ``(1, nf, 3, *cells)`` and their per-cell tails ``(nf, *cells)``
    (:func:`dft_tail_cells`). Cell by cell, not a term's norm times the phasors' relative tails: a
    periodic slab's guided mode rings where the adjoint field is strong, and the product read the
    gradient's truncation 2x to 2.6x low (per cell 1.3x to 2x high).
    """

    kern_b = jnp.abs(jnp.asarray(kern)).reshape((-1,) + (1,) * (adjoint_tails.ndim - 1))
    lam, fwd = component_norms(adjoint_phasors[0]), component_norms(forward_phasors[0])
    # a frequency the figure of merit does not read has no adjoint field, and its tails (infinite for a
    # field still arriving there) count nothing: inf * 0 would be NaN, and refuse
    per_cell = kern_b * (jnp.where(lam > 0, forward_tails * lam, 0.0) + jnp.where(fwd > 0, fwd * adjoint_tails, 0.0))
    return row_norms(per_cell / jnp.min(inv_permittivities, axis=0) ** 2)


def gradient_error_estimate(truncation: jax.Array, gradient_norm: jax.Array, objective: jax.Array) -> jax.Array:
    """Estimated relative distance of the gradient from the converged one.

    ``sum_f truncation_f / ||grad|| + eta_obj``: the design terms' truncation (:func:`truncation_norms`,
    summed over the regions), plus ``objective``, the largest relative truncation of the objective
    phasors the figure of merit reads, which shifts the cotangent the adjoint solve is driven by
    about as much. A frequency the figure of merit does not read has no adjoint current and adds
    nothing; terms that cancel in the sum still count their errors. On a Fabry-Perot cavity off
    resonance (2000 fs) the two parts were 4.1e-3 and 5.2e-3 against a true 9.3e-3.
    """
    tiny = jnp.finfo(truncation.dtype).tiny
    total = jnp.sum(truncation)
    # a non-finite truncation (an overflowing continuation in float32) is no estimate: refused
    part = jnp.where(total > 0, total / jnp.maximum(gradient_norm, tiny), 0.0)
    return jnp.where(jnp.isfinite(total), part, jnp.inf) + objective


def refuse_unconverged(
    gradient: jax.Array, estimate: jax.Array, objective: jax.Array, tolerance: float | None
) -> jax.Array:
    """``gradient``, raising when either truncation estimate is above ``tolerance`` (or not finite).

    ``objective`` is the largest relative truncation of the objective phasors over the channels and
    frequencies the figure of merit reads, ``estimate`` :func:`gradient_error_estimate`, which
    includes it; the objective's own refusal comes first, for its message.
    """
    if tolerance is None:
        return gradient
    gradient = eqx.error_if(
        gradient,
        ~(jnp.isfinite(objective) & jnp.isfinite(estimate)),
        "The reciprocity gradient is refused: a field was still arriving at the end of the run, at an objective "
        "monitor or in a design region (its last late window more than 10 times its earlier ones; the diagnostics "
        "show that recording's tail as inf). How large it will get cannot be told from the run. It is an echo "
        "from far away, or a pulse's front, possibly the PML's own weak reflection coming back. Lengthen the "
        "simulation until it has passed (often 10-30% longer does), or thicken the PML, or use "
        "GradientConfig(method='checkpointed'), which differentiates the truncated run as it is. "
        "tail_tolerance=None switches the check off.",
    )
    gradient = eqx.error_if(
        gradient,
        ~(objective <= tolerance),
        "The reciprocity gradient is refused: the objective monitors' phasors have not converged (their estimated "
        f"truncation exceeds tail_tolerance={tolerance:.1e}), so the figure of merit itself, and every method's "
        "gradient of it, carry that error. The fields were still present at the monitors at the end of the run. "
        "A source carrying DC (a few-cycle pulse) leaves a static remainder no run length removes: make its "
        "carrier DC-free (for a GaussianPulseProfile, total carrier phase = pi/2 - 2*pi*f0*t0 with t0 = 6*sigma_t "
        "and sigma_t = 1/(2*pi*spectral_width)). Otherwise lengthen the simulation. The estimates are in "
        "param_fn.diagnostics (functional API); tail_tolerance=None switches the check off.",
    )
    return eqx.error_if(
        gradient,
        ~(estimate <= tolerance),
        "The reciprocity gradient is refused: the phasors it is built from have not converged, and its estimated "
        f"distance from the converged gradient exceeds tail_tolerance={tolerance:.1e}. The fields were still "
        "present, or ringing (a resonance, a grazing diffraction order, a guided mode, a mode between parallel "
        "walls), at the end of the run. Lengthen the simulation. GradientConfig(method='checkpointed') "
        "differentiates the truncated run exactly instead, but where a mode rings off the objective frequency "
        "that gradient converges more slowly still. The estimates are in param_fn.diagnostics (functional API); "
        "tail_tolerance=None switches the check off.",
    )


class TailReport:
    """Receives the convergence estimates from host callbacks and records them in ``diagnostics``."""

    def __init__(self, diagnostics: dict, tolerance: float | None):
        self.diagnostics = diagnostics
        self.tolerance = tolerance

    def __call__(self, stage: str, labels: tuple[str, ...], values) -> None:
        self.diagnostics[stage] = {label: float(np.max(np.asarray(v))) for label, v in zip(labels, values)}
