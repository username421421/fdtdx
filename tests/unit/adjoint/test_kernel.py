"""The numerical kernel: amplitude-solve conditioning and the convergence estimate, on synthetic records.

No FDTD solves. The estimate's measurements against real runs are in ``notes/adjoint/05-guards.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.adjoint.kernel import (
    DEFAULT_COND_LIMIT,
    amplitude_matrix,
    component_norms,
    dft_tail,
    gaussian_window,
    gradient_error_estimate,
    truncation_norms,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


class TestAmplitudeSolveGuard:
    """At 22 fs the colour splitter's solve had cond 7.2e7, passed the former 1e8
    ceiling, and gave gradient rel 20. float32 solve error is about 1e-8 * cond."""

    def test_default_limit_is_1e4(self):
        assert DEFAULT_COND_LIMIT == 1e4

    def test_mid_range_condition_is_refused_by_default(self):
        # 52 steps (5 fs), 600 and 620 nm: cond 1.9e4, which the former 1e8 default accepted
        dt = 0.99 * 50e-9 / (299792458.0 * np.sqrt(3.0))
        omegas = tuple(2 * np.pi * 299792458.0 / w for w in (600e-9, 620e-9))
        with pytest.raises(ValueError, match="ill-conditioned"):
            amplitude_matrix(omegas, dt, gaussian_window(52))
        _, cond = amplitude_matrix(omegas, dt, gaussian_window(52), cond_limit=1e8)
        assert 1e4 < cond < 1e8


class TestDftTail:
    dt = 1e-16
    omega = 2 * np.pi / (20 * 1e-16)  # 20 steps per period

    def _record(self, modes, T=400, cells=1):
        """What a detector records of damped carriers ``(amplitude, detune, tau_steps, phase)``: the end
        field, the full phasors, their growth over the three late windows, and the true relative tail.

        With ``cells > 1`` every mode has its own random complex shape ``psi`` over the cells, the field
        being ``Re[psi exp(i w' t)]``: its two rotating parts then have shapes ``psi`` and ``conj(psi)``.
        """
        rng = np.random.default_rng(0)
        shapes = [
            rng.standard_normal(cells) + 1j * rng.standard_normal(cells) if cells > 1 else np.ones(1) for _ in modes
        ]

        def field(steps):
            return sum(
                np.real(psi[None, :] * np.exp(1j * (self.omega * (1 + d) * steps * self.dt + p))[:, None])
                * (a * np.exp(-steps / tau))[:, None]
                for psi, (a, d, tau, p) in zip(shapes, modes)
            )

        def dft(steps):
            return np.sum(np.exp(1j * self.omega * steps * self.dt)[:, None] * field(steps), axis=0)

        w = int(0.125 * T)
        full = dft(np.arange(T))
        late = tuple(jnp.asarray(dft(np.arange(T - k * w, T - (k - 1) * w))).reshape(1, 1, cells) for k in (3, 2, 1))
        true = np.linalg.norm(dft(np.arange(T, T + 60 * int(max(m[2] for m in modes))))) / np.linalg.norm(full)
        left = jnp.asarray(field(np.asarray([T]))).reshape(1, cells)
        return left, jnp.asarray(full).reshape(1, 1, cells), late, true

    def _estimate(self, left, full, late):
        return float(dft_tail(left, full, late, (self.omega,), self.dt)[0])

    def test_decayed_field_has_a_negligible_tail(self):
        assert self._estimate(*self._record([(1.0, 0.0, 20.0, 0.0)])[:3]) < 1e-6

    def test_undecayed_field_has_a_large_tail(self):
        # ten periods of an undamped carrier: the late windows do not decay, so the tail has no bound;
        # the estimate saturates far above the default tolerance (1e-2)
        assert self._estimate(*self._record([(1.0, 0.0, 1e6, 0.0)], T=200)[:3]) > 0.1

    def test_resonant_ringing_is_estimated_from_the_late_windows(self):
        """A mode ringing at the frequency: the end-field (static) estimate alone is several times too small.
        On one cell the window ratio is taken with its counter-rotating uncertainty (1.3x high here)."""
        left, full, late, true = self._record([(1.0, 0.0, 150.0, 0.0)])
        gap = abs(1 - np.exp(1j * self.omega * self.dt))
        static = abs(float(left[0, 0])) / (gap * abs(complex(full[0, 0, 0])))
        assert static < true / 5
        eta = self._estimate(left, full, late)
        assert true / 1.1 < eta < 1.5 * true, f"estimate {eta:.3e}, true {true:.3e}"

    @pytest.mark.parametrize("detune, tau, T", [(0.3, 150.0, 400), (0.05, 400.0, 800), (0.1, 300.0, 400)])
    def test_off_resonant_ringing_is_estimated_within_a_factor(self, detune, tau, T):
        """One cell: one mode, which cannot tell a real carrier's two rotating parts apart (1.42x low at
        detune 0.3). Over many cells the two-mode fit separates them."""
        left, full, late, true = self._record([(1.0, detune, tau, 1.0)], T=T)
        eta = self._estimate(left, full, late)
        assert true / 1.5 < eta < 5 * max(true, 1e-9), f"one cell: estimate {eta:.3e}, true {true:.3e}"
        left, full, late, true = self._record([(1.0, detune, tau, 1.0)], T=T, cells=32)
        eta = self._estimate(left, full, late)
        assert true / 1.1 < eta < 5 * max(true, 1e-9), f"32 cells: estimate {eta:.3e}, true {true:.3e}"

    def test_a_slow_mode_under_a_fast_one_is_not_missed(self):
        """Two modes with their own shapes: the fast one dominates the first window, the slow one the tail.
        One mode fitted to the last two windows read the eps-12 Device's mixture 1.5x to 6.7x low."""
        modes = [(3000.0, 0.0, 40.0, 0.0), (1.0, 0.04, 400.0, 0.7)]
        left, full, late, true = self._record(modes, T=400, cells=32)
        eta = self._estimate(left, full, late)
        assert true / 1.3 < eta < 5 * true, f"estimate {eta:.3e}, true {true:.3e}"

    def test_a_standing_wave_on_resonance_is_not_missed(self):
        """A real mode at the frequency, decaying by 1% per window, shape shared by its two rotating
        parts (collinear windows, one mode): their boundary terms moved the ratio by as much as
        ``|1 - z|``, and the estimate read 3.7x low."""
        n, cells = 1574, 64
        w = int(0.125 * n)
        omega = 2 * np.pi / 21.0
        steps = np.arange(0, 80 * n)
        s = np.cos(omega * steps) * np.exp(-5e-5 * steps)
        s[: n // 3] = 0.0
        cum = np.concatenate([[0], np.cumsum(s * np.exp(1j * omega * steps))])
        shape = np.sin(np.linspace(0.1, 3.0, cells))
        late = tuple(
            jnp.asarray(((cum[n - k * w + w] - cum[n - k * w]) * shape).reshape(1, 1, cells)) for k in (3, 2, 1)
        )
        tail = np.linalg.norm((cum[-1] - cum[n]) * shape)
        # a converged part on top, so the true relative tail is about 2e-2
        extra = np.cos(np.linspace(0, 7, cells)) + 0.3j
        phasor = cum[n] * shape + extra * (tail / 2e-2) / np.linalg.norm(extra)
        true = tail / np.linalg.norm(phasor)
        left = jnp.asarray((s[n - 1] * shape).reshape(1, cells))
        eta = float(dft_tail(left, jnp.asarray(phasor.reshape(1, 1, cells)), late, (omega,), 1.0)[0])
        assert eta > true / 1.3, f"estimate {eta:.3e}, true {true:.3e}"

    def test_a_standing_wave_with_a_small_remainder_is_not_missed(self):
        """A standing wave on resonance (collinear windows) plus 1e-3 on another shape switched to the
        two-mode fit, which mixed the wave's two rotating parts into one mode: 0.28 of the tail."""
        n, cells = 1574, 64
        w = int(0.125 * n)
        omega = 2 * np.pi / 21.0
        steps = np.arange(0, 80 * n)
        s = np.cos(omega * steps) * np.exp(-5e-5 * steps)
        s[: n // 3] = 0.0
        other = np.cos(omega * 1.3 * steps) * np.exp(-steps / 400.0) * 1e-3
        cum_s = np.concatenate([[0], np.cumsum(s * np.exp(1j * omega * steps))])
        cum_o = np.concatenate([[0], np.cumsum(other * np.exp(1j * omega * steps))])
        wave = np.sin(np.linspace(0.1, 3.0, cells))
        rest = np.cos(np.linspace(0.0, 5.0, cells))

        def record(lo, hi):
            return (cum_s[hi] - cum_s[lo]) * wave + (cum_o[hi] - cum_o[lo]) * rest

        late = tuple(jnp.asarray(record(n - k * w, n - (k - 1) * w).reshape(1, 1, cells)) for k in (3, 2, 1))
        tail = np.linalg.norm(record(n, len(steps)))
        extra = np.cos(np.linspace(0, 7, cells)) + 0.3j
        phasor = record(0, n) + extra * (tail / 2e-2) / np.linalg.norm(extra)
        true = tail / np.linalg.norm(phasor)
        left = jnp.asarray((s[n - 1] * wave + other[n - 1] * rest).reshape(1, cells))
        eta = float(dft_tail(left, jnp.asarray(phasor.reshape(1, 1, cells)), late, (omega,), 1.0)[0])
        assert eta > true / 1.3, f"estimate {eta:.3e}, true {true:.3e}"

    def test_float32_keeps_a_slow_mode_under_a_fast_one(self):
        """The fit in complex64: a slow mode (z = 0.999) at 3e-3 of a fast one (z = 0.5). The Gram
        determinant read 0.09 of its tail, one Gram-Schmidt projection 0.39."""
        rng = np.random.default_rng(7)
        u, s = (rng.normal(size=1000) + 1j * rng.normal(size=1000) for _ in range(2))
        zs, vs = (0.5, 0.999), (u, 3e-3 * s)
        late = tuple(
            jnp.asarray(sum(v * z**k for v, z in zip(vs, zs)).reshape(1, 1, -1), dtype=jnp.complex64) for k in range(3)
        )
        tail = np.linalg.norm(sum(v * z**3 / (1 - z) for v, z in zip(vs, zs)))
        phasor = rng.normal(size=1000) + 1j * rng.normal(size=1000)
        phasor *= tail / np.linalg.norm(phasor) / 3e-2
        left = jnp.zeros((1, 1000), dtype=jnp.float32)
        full = jnp.asarray(phasor.reshape(1, 1, -1), dtype=jnp.complex64)
        eta = float(dft_tail(left, full, late, (self.omega,), self.dt)[0])
        np.testing.assert_allclose(eta, 3e-2, rtol=0.02)

    def test_cell_tails_do_not_underflow_in_float32(self):
        """Per-cell norms of adjoint phasors of 1e-21 (a cotangent of a figure of merit in SI watts)
        squared to 0 and switched the gradient check off."""
        x = jnp.full((1, 3, 4), 1e-21 + 1e-21j, dtype=jnp.complex64)
        np.testing.assert_allclose(np.asarray(component_norms(x)), np.sqrt(6) * 1e-21, rtol=1e-5)
        lam = jnp.full((1, 1, 3, 4), 1e-21 + 0j, dtype=jnp.complex64)
        fwd = jnp.ones((1, 1, 3, 4), dtype=jnp.complex64)
        tails = jnp.full((1, 4), 1e-3, dtype=jnp.float32)
        inv_eps = jnp.ones((1, 4), dtype=jnp.float32)
        norms = truncation_norms(lam, fwd, tails * 1e-21, tails, inv_eps, np.asarray([1.0 + 0j]))
        assert float(norms[0]) > 0

    def test_an_arriving_field_at_an_unread_frequency_counts_nothing(self):
        """Its forward tails are infinite, but with no adjoint field there (not read) the term is 0, not NaN."""
        lam = jnp.stack([jnp.ones((3, 4)), jnp.zeros((3, 4))])[None].astype(jnp.complex64)
        fwd = jnp.ones((1, 2, 3, 4), dtype=jnp.complex64)
        forward_tails = jnp.asarray([[1e-3] * 4, [jnp.inf] * 4], dtype=jnp.float32)
        adjoint_tails = jnp.asarray([[1e-3] * 4, [0.0] * 4], dtype=jnp.float32)
        norms = truncation_norms(lam, fwd, adjoint_tails, forward_tails, jnp.ones((1, 4)), np.asarray([1.0, 1.0]))
        assert np.isfinite(float(norms[0])) and float(norms[1]) == 0.0

    def test_a_non_finite_truncation_refuses(self):
        """An overflowing continuation (float32) made the truncation NaN, which where(total > 0) read as 0."""
        estimate = gradient_error_estimate(jnp.asarray([jnp.nan, 1.0]), jnp.asarray(1.0), jnp.asarray(0.0))
        assert not bool(estimate <= 1e-2)

    def test_a_field_still_arriving_is_refused(self):
        """An echo reaching the monitor in the last window: a series continuation read 4.5e-5 where the
        figure of merit was 89% off. Zero earlier windows (a front, or float32 flushing them) are the same
        case."""
        rng = np.random.default_rng(3)
        for cells in (1, 32):
            shape = (rng.normal(size=cells) + 1j * rng.normal(size=cells)).reshape(1, 1, cells)
            full = jnp.asarray(10.0 * shape)
            for early in (1e-6, 0.0):
                late = (jnp.asarray(early * shape), jnp.asarray(early * shape), jnp.asarray(1e-2 * shape))
                left = jnp.zeros((1, cells))
                assert not float(dft_tail(left, full, late, (self.omega,), self.dt)[0]) <= 1e-2

    def test_an_arrival_behind_a_decaying_first_window_is_refused(self):
        """A Device's ringing in the first window (n0 = n2 / 9.5) hid an echo arriving in the last one
        (n2 / n1 = 590): accepted with the figure of merit 80% off."""
        rng = np.random.default_rng(4)
        shape = (rng.normal(size=32) + 1j * rng.normal(size=32)).reshape(1, 1, 32)
        other = (rng.normal(size=32) + 1j * rng.normal(size=32)).reshape(1, 1, 32)
        late = (jnp.asarray(2.2e-5 * other), jnp.asarray(3.5e-7 * other), jnp.asarray(2.1e-4 * shape))
        eta = float(dft_tail(jnp.zeros((1, 32)), jnp.asarray(shape), late, (self.omega,), self.dt)[0])
        assert not eta <= 1e-2

    def test_per_frequency_and_float32_safe(self):
        left = jnp.full((1, 1), 1e-25, dtype=jnp.float32)
        full = jnp.asarray([[[1e-20 + 0j]], [[1e-22 + 0j]]], dtype=jnp.complex64)
        late = (jnp.zeros_like(full),) * 3
        eta = np.asarray(dft_tail(left, full, late, (self.omega, 2 * self.omega), self.dt))
        gap = np.abs(1 - np.exp(1j * np.asarray([1.0, 2.0]) * self.omega * self.dt))
        # no underflow to 0 in float32: the static estimate per frequency
        np.testing.assert_allclose(eta, 1e-25 / (gap * np.asarray([1e-20, 1e-22])), rtol=1e-4)
