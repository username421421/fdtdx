"""The numerical kernel: adjoint window, amplitude solve, gradient kernel, convergence estimate.

The measurements behind each constant are in ``notes/adjoint/02-implementation.md`` and
``notes/adjoint/05-guards.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.adjoint.validation import check_conditioning

#: Ceiling on the condition number of the amplitude-solve matrix. The float32 solve error is
#: about ``1e-8 * cond``, so 1e4 keeps it near 1e-4.
DEFAULT_COND_LIMIT = 1e4


def gaussian_window(
    time_steps_total: int,
    center_frac: float = 0.30,
    sigma_frac: float = 0.055,
    dtype=jnp.float64,
) -> jax.Array:
    """Envelope of the adjoint excitation, shape ``(time_steps_total,)``.

    A decaying pulse, not a CW source, because reciprocity holds only once both DFTs have
    converged. It peaks 30% into the run, where it starts at ``exp(-14.9)`` of its peak (no DC
    step), and leaves the last two thirds for the field to leave the domain. ``sigma_frac`` must
    span several optical periods, or the amplitude solve becomes ill-conditioned.
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


def solve_amplitudes(matrix: jax.Array, target: jax.Array) -> jax.Array:
    """Amplitudes ``(nf, nc, *cells)`` whose windowed current has the DFT ``target``."""
    nf = target.shape[0]
    flat = target.reshape(nf, -1)
    rhs = jnp.concatenate([jnp.real(flat), jnp.imag(flat)], axis=0)
    sol = jnp.linalg.solve(matrix, rhs)
    return (sol[:nf] + 1j * sol[nf:]).reshape(target.shape)


def leapfrog_kernel(angular_frequencies: Sequence[float], dt: float, courant_number: float) -> np.ndarray:
    """``(exp(+i w dt) - 1) / courant``: the leapfrog's own stand-in for the continuum ``i w``.

    Perturbing ``inv_eps`` in ``E_{n+1} = E_n + courant * inv_eps * curl_H`` moves the field
    by ``d(inv_eps) (E_{n+1} - E_n) / inv_eps``, whose DFT is ``(1 - exp(+i w dt)) F`` for the
    post-step phasor ``F``; the injection law supplies ``-1 / courant``. The continuum factor
    gives the wrong sign at the ``w dt`` FDTD runs at.
    """
    omega = np.asarray(angular_frequencies, dtype=np.float64)
    return (np.exp(1j * omega * dt) - 1.0) / courant_number


def assemble_material_gradient(
    adjoint_phasors: jax.Array,
    forward_phasors: jax.Array,
    inv_permittivities: jax.Array,
    angular_frequencies: Sequence[float],
    dt: float,
    courant_number: float,
) -> jax.Array:
    """``d loss / d inv_permittivities`` on one design region, shaped like ``inv_permittivities``.

    ``Re[sum_f K(w_f) sum_c Lambda_c F_c] / inv_eps^2`` with ``K`` the :func:`leapfrog_kernel`, from
    the adjoint and forward design phasors ``(1, nf, 3, *cells)`` (raw scale) and the region's
    ``inv_permittivities`` ``(1 | 3, *cells)``. The components are summed for an isotropic
    permittivity, one value per cell driving all three. Exact with loss in the cells too: the lossy
    update's ``1 + a`` cancels against the lossy injection of the reciprocal source.
    """
    kern = leapfrog_kernel(angular_frequencies, dt, courant_number)
    lam, fwd = adjoint_phasors[0], forward_phasors[0]
    overlap = jnp.sum(lam * fwd, axis=1, keepdims=True) if inv_permittivities.shape[0] == 1 else lam * fwd
    kern_b = jnp.asarray(kern).reshape((-1,) + (1,) * (overlap.ndim - 1))
    return jnp.real(jnp.sum(kern_b * overlap, axis=0)) * (1.0 / (inv_permittivities**2))


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

#: Two modes are fitted only with at least this many entries per frequency (cells times
#: components) and the first two windows this far from collinear; otherwise one.
_TWO_MODE_ENTRIES = 8
_COLLINEAR = 1e-10

#: Windows this close to collinear (``||b_perp||^2 / ||b||^2``) also take the one-mode estimate:
#: a standing wave's two rotating parts are not two modes.
_NEAR_COLLINEAR = 1e-3

#: A last window this many times larger than both earlier ones, or than the middle one while
#: larger than the first, is a field still arriving (an echo, a pulse's front), whose size no
#: continuation can tell.
_ARRIVING = 10.0


def _continued(z: jax.Array) -> jax.Array:
    """``z^3 / (1 - z)``: a mode's windows after the third, summed; ``|1 - z|`` held at 1e-3 or more."""
    gap = 1.0 - z
    size = jnp.abs(gap)
    held = jnp.where(size > 0, gap / jnp.where(size > 0, size, 1.0), 1.0) * (1.0 - _MAX_LATE_RATIO)
    return z**3 / jnp.where(size > 1.0 - _MAX_LATE_RATIO, gap, held)


def _series(z: jax.Array) -> jax.Array:
    """``|z / (1 - z)|``: a mode's windows after the last, over the last; ``|1 - z|`` held at 1e-3 or more.

    Not held below 1 in ``|z|``: on few cells beating modes make a window grow.
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
    cell over the components, ``(nf, *cells)`` (raw scale).

    Two estimates of the part of the transform the run did not see, the larger taken: a static
    remainder, ``||fields|| / |1 - exp(i w dt)|``, and ringing, the late windows ``D0, D1, D2``
    continued. A decaying mode grows the phasor by ``v z^k`` over window ``k``. Two modes are
    fitted over the cells (``D2 = -(a0 D0 + a1 D1)`` by least squares, ``z^2 + a1 z + a0 = 0``) and
    each is summed on, ``v z^3 / (1 - z)``; what they leave of ``D2`` is continued like the slowest
    of them and of the last pair's ratio ``z = <D1, D2> / <D1, D1>``. With fewer than
    ``_TWO_MODE_ENTRIES`` entries, or (nearly) collinear windows, one mode is continued,
    ``||D2|| |z| / (|1 - z| - s)``, ``s`` the shift of ``z`` a counter-rotating part can make. A field
    still arriving (``_ARRIVING``) is infinite. An estimate, not a bound: a weak mode hidden under
    stronger ones in the late windows is not seen.

    Args:
        fields: the recorded fields at the last step, ``(nc, *cells)``.
        phasors: their raw-scale phasors over the run, ``(nf, nc, *cells)``.
        late: the raw-scale phasors' growth over the three late windows
            (:func:`~fdtdx.adjoint.design.late_windows`), oldest first, each shaped like ``phasors``.
        angular_frequencies: the ``nf`` frequencies, rad/s.
        dt: the time step.
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

    # two modes: c = -(a0 a + a1 b) by least squares, by modified Gram-Schmidt on a and b (the Gram
    # determinant loses a slow mode under a fast one in float32), then z^2 + a1 z + a0 = 0
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
    # small, and the pessimistic |1 - z| - shift is taken. Windows no bigger than that hold no
    # ringing to extrapolate (a static remainder), and z is taken as it is.
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
    row before squaring so float32 cannot underflow."""
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
    (:func:`dft_tail_cells`).
    """
    kern_b = jnp.abs(jnp.asarray(kern)).reshape((-1,) + (1,) * (adjoint_tails.ndim - 1))
    lam, fwd = component_norms(adjoint_phasors[0]), component_norms(forward_phasors[0])
    # a frequency the figure of merit does not read has no adjoint field, and its tails (infinite for a
    # field still arriving there) count nothing: inf * 0 would be NaN
    per_cell = kern_b * (jnp.where(lam > 0, forward_tails * lam, 0.0) + jnp.where(fwd > 0, fwd * adjoint_tails, 0.0))
    return row_norms(per_cell / jnp.min(inv_permittivities, axis=0) ** 2)


def gradient_error_estimate(truncation: jax.Array, gradient_norm: jax.Array, objective: jax.Array) -> jax.Array:
    """Estimated relative distance of the gradient from the converged one.

    ``sum_f truncation_f / ||grad|| + objective``: the design terms' truncation
    (:func:`truncation_norms`, summed over the regions), plus the largest relative truncation of
    the objective phasors the figure of merit reads, which shifts the cotangent the adjoint solve is
    driven by about as much. A non-finite truncation is no estimate and reads infinite.
    """
    tiny = jnp.finfo(truncation.dtype).tiny
    total = jnp.sum(truncation)
    part = jnp.where(total > 0, total / jnp.maximum(gradient_norm, tiny), 0.0)
    return jnp.where(jnp.isfinite(total), part, jnp.inf) + objective


def refuse_unconverged(
    gradient: jax.Array, estimate: jax.Array, objective: jax.Array, tolerance: float | None
) -> jax.Array:
    """``gradient``, raising when either truncation estimate is above ``tolerance`` or not finite.

    ``objective`` is the largest relative truncation of the objective phasors the figure of merit
    reads, ``estimate`` the :func:`gradient_error_estimate`, which includes it; the objective's own
    refusal comes first, for its message.
    """
    if tolerance is None:
        return gradient
    switch_off = "GradientConfig(tail_tolerance=None) switches this check off."
    gradient = eqx.error_if(
        gradient,
        ~(jnp.isfinite(objective) & jnp.isfinite(estimate)),
        "The reciprocity gradient is refused: a field was still arriving at an objective monitor or a Device at "
        "the end of the run (an echo, a pulse's front, or the PML's own weak reflection), so how large it gets "
        "cannot be told from the run. Lengthen the simulation until it has passed (often 10-30% longer), thicken "
        f"the PML, or use GradientConfig(method='checkpointed'). {switch_off}",
    )
    gradient = eqx.error_if(
        gradient,
        ~(objective <= tolerance),
        f"The reciprocity gradient is refused: the objective monitors' phasors are more than tail_tolerance="
        f"{tolerance:.1e} from converged, so the figure of merit itself, and every method's gradient of it, carry "
        "that error. Lengthen the simulation. A source carrying DC (a few-cycle pulse) leaves a static remainder "
        "no run removes: make its carrier DC-free (for a GaussianPulseProfile, carrier phase = pi/2 - 2*pi*f0*t0 "
        f"with t0 = 6*sigma_t and sigma_t = 1/(2*pi*spectral_width)). {switch_off}",
    )
    return eqx.error_if(
        gradient,
        ~(estimate <= tolerance),
        f"The reciprocity gradient is refused: its estimated distance from the converged gradient exceeds "
        f"tail_tolerance={tolerance:.1e}, because the fields were still present or ringing (a resonance, a "
        "guided mode, a grazing diffraction order) at the end of the run. Lengthen the simulation, or use "
        "GradientConfig(method='checkpointed'), which differentiates the truncated run as it is. "
        f"{switch_off}",
    )
