"""Guards against the silent failure modes found by the hazard and convergence harvests.

No FDTD solves: placement, one update step, or setup only. The gradient parity
behind each guard is in ``tests/simulation/adjoint/test_production.py``. The
names these guards added are imported inside the tests, so each test reports
on its own against a tree without them.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.adjoint import gaussian_window, reciprocity_param_fn, reciprocity_phasor_fn
from fdtdx.adjoint.kernel import solve_adjoint_amplitudes
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.core.grid import QuasiUniformGrid, RectilinearGrid, UniformGrid
from fdtdx.fdtd.initialization import apply_params
from fdtdx.fdtd.update import update_E, update_H
from fdtdx.interfaces.recorder import Recorder
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.sources.adjoint import AdjointCurrentSource

_KEY = jax.random.PRNGKey(0)
_N = 12


@pytest.fixture(autouse=True)
def _enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _scene(
    *,
    devices=(("design", (4, 4, 4), (4, 4, 4)),),
    regions=(),
    blocks=(),
    time=10e-15,
    wavelengths=(600e-9, 700e-9),
    monitor=((10, 6, 6), (1, 1, 1), ("Ez",)),
    grid=None,
    pml=0,
    extra=(),
    device_material=None,
    gradient_config=None,
    return_params=False,
):
    """12^3 scene, no source, and no boundaries unless ``pml`` cells of PML. ``regions`` are
    extra stock detectors ``(name, lower, shape[, wave_characters])``; ``blocks`` are
    ``(name, lower, shape, material)``; ``extra`` are ``(object, lower)``. ``grid`` defaults
    to 50 nm cubes; ``device_material`` replaces the Devices' ``si`` (eps 2.25)."""
    config = SimulationConfig(
        time=time,
        grid=grid or UniformGrid(spacing=50e-9),
        backend="cpu",
        dtype=jnp.float64,
        gradient_config=gradient_config,
    )
    objs, cons = [], []
    vol = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
    objs.append(vol)
    if pml:
        bd, cl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=pml), vol)
        objs.extend(bd.values())
        cons.extend(cl)
    edges = None if grid is None else [config.resolve_grid((_N,) * 3).edges(a) for a in range(3)]

    def at(obj, lo):
        objs.append(obj)
        if edges is None:
            cons.append(obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=lo))
        else:  # index-space placement is refused on a non-uniform grid
            margins = tuple(float(e[i] - e[0]) for e, i in zip(edges, lo))
            cons.append(obj.place_relative_to(vol, (0, 1, 2), (-1,) * 3, (-1,) * 3, margins=margins))

    for name, lo, shape, material in blocks:
        at(fdtdx.UniformMaterialObject(name=name, partial_grid_shape=shape, material=material), lo)
    for name, lo, shape in devices:
        dev = fdtdx.Device(
            name=name,
            partial_grid_shape=shape,
            partial_voxel_grid_shape=(1, 1, 1),
            materials={
                "air": fdtdx.Material(permittivity=1.0),
                "si": device_material or fdtdx.Material(permittivity=2.25),
            },
            param_transforms=[],
        )
        at(dev, lo)
    wcs = [fdtdx.WaveCharacter(wavelength=w) for w in wavelengths]
    mon_lo, mon_shape, mon_comps = monitor
    mon = PhasorDetector(
        name="mon",
        partial_grid_shape=mon_shape,
        wave_characters=wcs,
        components=mon_comps,
        exact_interpolation=False,
    )
    at(mon, mon_lo)
    for name, lo, shape, *own in regions:
        at(PhasorDetector(name=name, partial_grid_shape=shape, wave_characters=own[0] if own else wcs), lo)
    for obj, lo in extra:
        at(obj, lo)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objs, config=config, constraints=cons, key=_KEY
    )
    return (objects, arrays, config, params) if return_params else (objects, arrays, config)


# --------------------------------------------------------------------------- 1. lossy monitor


_LOSSY = fdtdx.Material(permittivity=2.25, permeability=2.0, electric_conductivity=1e5, magnetic_conductivity=6e9)
_LOSSY_SI = fdtdx.Material(permittivity=2.25, electric_conductivity=1e5)


class TestLossyInjection:
    """The adjoint current sits where the objective monitor reads the fields. FDTDX
    adds every source after the lossy division by 1 + a, so in a lossy cell the
    current is 1 + a times its reciprocal partner and the cotangent is divided back.
    Measured before: rel 2.39e-01 at cosine 1.0000000000 (pure scale 1 + a)."""

    def _lossy_scene(self):
        # lossy block on x 5..6 only; the probed block x 4..7 is half lossless
        return _scene(devices=(), blocks=(("loss", (5, 4, 4), (2, 4, 4), _LOSSY),))

    def _fdtdx_factor(self, family):
        """1 + a (E) or 1 + b (H) per cell, read off one FDTDX update with zero curl."""
        objects, arrays, config = self._lossy_scene()
        ones = jnp.ones_like(arrays.fields.E)
        zeros = jnp.zeros_like(arrays.fields.E)
        if family == "E":
            arrays = arrays.aset("fields->E", ones).aset("fields->H", zeros)
            ratio = update_E(jnp.asarray(0), arrays, objects, config, simulate_boundaries=False).fields.E
        else:
            arrays = arrays.aset("fields->H", ones).aset("fields->E", zeros)
            ratio = update_H(jnp.asarray(0), arrays, objects, config, simulate_boundaries=False).fields.H
        # zero curl: X_new = (1 - a) / (1 + a) * X_old, so 1 + a = 2 / (1 + ratio)
        return objects, arrays, config, 2.0 / (1.0 + np.asarray(ratio))

    @pytest.mark.parametrize("family", ["E", "H"])
    def test_divisor_is_fdtdx_own_lossy_factor(self, family):
        from fdtdx.adjoint.objective import lossy_injection

        _, arrays, config, factor = self._fdtdx_factor(family)
        block = ((4, 8), (4, 8), (4, 8))
        comps = ("Ex", "Ey", "Ez") if family == "E" else ("Hx", "Hy", "Hz")
        loss = lossy_injection(arrays, float(config.courant_number), block, comps)
        assert loss is not None
        got = np.asarray(loss.divisor(arrays.inv_permittivities, arrays.electric_conductivity))
        want = factor[:, 4:8, 4:8, 4:8]
        np.testing.assert_allclose(got, want, rtol=1e-12)
        assert want.max() > 1.1  # the lossy cells are really lossy ...
        np.testing.assert_allclose(want[:, 0], 1.0, rtol=1e-12)  # ... and x = 4 really is not

    def test_divisor_follows_the_live_conductivity(self):
        """A Device can make the conductivity design-dependent, so the divisor reads it per call;
        a lossless block divides by exactly one."""
        from fdtdx.adjoint.objective import lossy_injection

        _, arrays, config = self._lossy_scene()
        ie, sigma = arrays.inv_permittivities, arrays.electric_conductivity
        lossless = lossy_injection(arrays, float(config.courant_number), ((8, 10), (4, 8), (4, 8)), ("Ez", "Hx"))
        assert lossless is not None and np.all(np.asarray(lossless.divisor(ie, sigma)) == 1.0)
        loss = lossy_injection(arrays, float(config.courant_number), ((4, 8), (4, 8), (4, 8)), ("Ez",))
        assert loss is not None
        a = np.asarray(loss.divisor(ie, sigma)) - 1.0
        np.testing.assert_allclose(np.asarray(loss.divisor(ie, 3.0 * sigma)) - 1.0, 3.0 * a, rtol=1e-12)
        assert a.max() > 0.1

    def test_sources_are_added_after_the_lossy_division(self):
        """The premise of the correction: FDTDX does not divide a source by 1 + a."""
        objects, arrays, config = self._lossy_scene()
        omega = 2 * np.pi * 3e8 / 600e-9
        src = AdjointCurrentSource(
            name="probe",
            amplitudes=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.complex128),
            window=jnp.ones((int(config.time_steps_total),)),
            angular_frequencies=(omega,),
            components=("Ez",),
            wave_character=fdtdx.WaveCharacter(wavelength=600e-9),
        ).place_on_grid(grid_slice_tuple=((5, 6), (5, 6), (5, 6)), config=config, key=_KEY)
        objects = objects.aset("object_list", [*objects.object_list, src])
        arrays = arrays.aset("fields->E", jnp.zeros_like(arrays.fields.E)).aset(
            "fields->H", jnp.zeros_like(arrays.fields.H)
        )
        E = update_E(jnp.asarray(0), arrays, objects, config, simulate_boundaries=False).fields.E
        inv_eps = float(arrays.inv_permittivities[0, 5, 5, 5])
        expected = -float(config.courant_number) * inv_eps * 1.0  # waveform at t = 0: Re(1) * window 1
        np.testing.assert_allclose(float(E[2, 5, 5, 5]), expected, rtol=1e-12)

    def test_full_conductivity_tensor_is_refused(self):
        objects, arrays, config = _scene()
        tensor = jnp.ones((9, _N, _N, _N), dtype=jnp.float64)
        with pytest.raises(NotImplementedError, match="electric_conductivity as a full 3x3 tensor"):
            reciprocity_phasor_fn(
                arrays.aset("electric_conductivity", tensor), objects, config, _KEY, objective_detectors="mon"
            )

    def test_conductivity_for_a_scene_without_one_is_refused(self):
        """The lossy correction is built from the scene's conductivity; a scene with none has none."""
        objects, arrays, config = _scene()
        phasor_fn = reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")
        with pytest.raises(ValueError, match="the scene stores none"):
            phasor_fn(arrays.inv_permittivities, jnp.zeros_like(arrays.inv_permittivities))

    def test_a_design_writing_the_conductivity_must_pass_it(self):
        """``phasor_fn(inv_eps)`` alone kept the placed conductivity, which ``apply_params`` no longer
        leaves in a lossy Device: FoM off by a factor of 24, silently."""
        objects, arrays, config = _scene(device_material=_LOSSY_SI)
        phasor_fn = reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")
        with pytest.raises(ValueError, match=r"writes the conductivity of Device\(s\) \['design'\]"):
            phasor_fn(arrays.inv_permittivities)

    @pytest.mark.parametrize("lower, listed", [((4, 4, 4), True), ((0, 0, 0), False)], ids=["under", "elsewhere"])
    def test_loss_placed_under_a_device_is_replaced(self, lower, listed):
        """``apply_params`` makes a lossless Device lossless over a lossy block; loss elsewhere stays."""
        from fdtdx.adjoint.validation import conductive_devices

        objects, arrays, _ = _scene(blocks=(("loss", lower, (4, 4, 4), _LOSSY_SI),))
        assert conductive_devices(objects, arrays) == (["design"] if listed else [])


class TestDispersionUnderDevice:
    """A dispersive block under a non-dispersive Device: apply_params zeroes its ADE
    coefficients in the Device cells on every call, the reciprocity solves did not.
    Measured before: forward FoM off by 60%, gradient rel 8.3e-01 at cosine 0.66."""

    def test_zeroing_matches_apply_params(self):
        from fdtdx.adjoint.design import device_dispersion_as_applied
        from fdtdx.fdtd.initialization import apply_params

        pole = fdtdx.LorentzPole(resonance_frequency=4e15, damping=2e14, delta_epsilon=0.5)
        lorentz = fdtdx.Material(permittivity=2.0, dispersion=fdtdx.DispersionModel(poles=(pole,)))
        objects, arrays, _, params = _scene(blocks=(("lorentz", (3, 3, 3), (6, 6, 6), lorentz),), return_params=True)
        sl = objects.devices[0].grid_slice
        assert float(jnp.max(jnp.abs(arrays.dispersive_c3[:, :, *sl]))) > 0
        once = device_dispersion_as_applied(arrays, objects)
        applied, _, _ = apply_params(arrays, objects, params, _KEY)
        for name in ("dispersive_c1", "dispersive_c2", "dispersive_c3"):
            np.testing.assert_array_equal(np.asarray(getattr(once, name)), np.asarray(getattr(applied, name)))
        # outside the Device the block keeps its coefficients
        assert float(jnp.max(jnp.abs(once.dispersive_c3[:, :, 3, 3, 3]))) > 0


# --------------------------------------------------------------------------- 2. conditioning / convergence


class TestAmplitudeSolveGuard:
    """At 22 fs the colour splitter's solve had cond 7.2e7, passed the former 1e8
    ceiling, and gave gradient rel 20. float32 solve error is about 1e-8 * cond."""

    def test_default_limit_is_1e4(self):
        from fdtdx.adjoint.kernel import DEFAULT_COND_LIMIT

        assert DEFAULT_COND_LIMIT == 1e4

    @pytest.mark.parametrize("entry", [reciprocity_phasor_fn, reciprocity_param_fn], ids=["phasor_fn", "param_fn"])
    def test_mid_range_condition_is_refused_by_default(self, entry):
        # 5 fs, 600 and 620 nm: cond 1.9e4, which the former 1e8 default accepted
        objects, arrays, config = _scene(time=5e-15, wavelengths=(600e-9, 620e-9))
        with pytest.raises(ValueError, match="ill-conditioned"):
            entry(arrays, objects, config, _KEY, objective_detectors="mon")
        entry(arrays, objects, config, _KEY, objective_detectors="mon", cond_limit=1e8)

    def test_solve_adjoint_amplitudes_uses_the_same_default(self):
        dt = 0.99 * 50e-9 / (299792458.0 * np.sqrt(3.0))
        T = 52
        omegas = tuple(2 * np.pi * 299792458.0 / w for w in (600e-9, 620e-9))
        target = jnp.ones((2, 1, 1, 1, 1), dtype=jnp.complex128)
        with pytest.raises(ValueError, match="ill-conditioned"):
            solve_adjoint_amplitudes(target, omegas, dt, gaussian_window(T))
        _, info = solve_adjoint_amplitudes(target, omegas, dt, gaussian_window(T), cond_limit=1e8)
        assert 1e4 < info["cond"] < 1e8


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
        from fdtdx.adjoint import dft_tail

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
        ``|1 - z|``, and the estimate read 3.7x low (review 2026-09-25)."""
        from fdtdx.adjoint import dft_tail

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
        eta = float(
            dft_tail(
                jnp.asarray((s[n - 1] * shape).reshape(1, cells)),
                jnp.asarray(phasor.reshape(1, 1, cells)),
                late,
                (omega,),
                1.0,
            )[0]
        )
        assert eta > true / 1.3, f"estimate {eta:.3e}, true {true:.3e}"

    def test_float32_keeps_a_slow_mode_under_a_fast_one(self):
        """The fit in complex64: a slow mode (z = 0.999) at 3e-3 of a fast one (z = 0.5). The Gram
        determinant read 0.09 of its tail, one Gram-Schmidt projection 0.39."""
        from fdtdx.adjoint import dft_tail

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
        from fdtdx.adjoint.kernel import component_norms, truncation_norms

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
        from fdtdx.adjoint.kernel import truncation_norms

        lam = jnp.stack([jnp.ones((3, 4)), jnp.zeros((3, 4))])[None].astype(jnp.complex64)
        fwd = jnp.ones((1, 2, 3, 4), dtype=jnp.complex64)
        forward_tails = jnp.asarray([[1e-3] * 4, [jnp.inf] * 4], dtype=jnp.float32)
        adjoint_tails = jnp.asarray([[1e-3] * 4, [0.0] * 4], dtype=jnp.float32)
        norms = truncation_norms(lam, fwd, adjoint_tails, forward_tails, jnp.ones((1, 4)), np.asarray([1.0, 1.0]))
        assert np.isfinite(float(norms[0])) and float(norms[1]) == 0.0

    def test_a_non_finite_truncation_refuses(self):
        """An overflowing continuation (float32) made the truncation NaN, which where(total > 0) read as 0."""
        from fdtdx.adjoint.kernel import gradient_error_estimate

        estimate = gradient_error_estimate(jnp.asarray([jnp.nan, 1.0]), jnp.asarray(1.0), jnp.asarray(0.0))
        assert not bool(estimate <= 1e-2)

    def test_a_field_still_arriving_is_refused(self):
        """An echo reaching the monitor in the last window: a series continuation read 4.5e-5 where the
        figure of merit was 89% off (review 2026-09-25). Zero earlier windows (a front, or float32
        flushing them) are the same case."""
        from fdtdx.adjoint import dft_tail

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
        (n2 / n1 = 590): accepted with the figure of merit 80% off (review 2026-09-26)."""
        from fdtdx.adjoint import dft_tail

        rng = np.random.default_rng(4)
        shape = (rng.normal(size=32) + 1j * rng.normal(size=32)).reshape(1, 1, 32)
        other = (rng.normal(size=32) + 1j * rng.normal(size=32)).reshape(1, 1, 32)
        late = (jnp.asarray(2.2e-5 * other), jnp.asarray(3.5e-7 * other), jnp.asarray(2.1e-4 * shape))
        eta = float(dft_tail(jnp.zeros((1, 32)), jnp.asarray(shape), late, (self.omega,), self.dt)[0])
        assert not eta <= 1e-2

    def test_a_standing_wave_with_a_small_remainder_is_not_missed(self):
        """A standing wave on resonance (collinear windows) plus 1e-3 on another shape switched to the
        two-mode fit, which mixed the wave's two rotating parts into one mode: 0.28 of the tail."""
        from fdtdx.adjoint import dft_tail

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

    def test_per_frequency_and_float32_safe(self):
        from fdtdx.adjoint import dft_tail

        left = jnp.full((1, 1), 1e-25, dtype=jnp.float32)
        full = jnp.asarray([[[1e-20 + 0j]], [[1e-22 + 0j]]], dtype=jnp.complex64)
        late = (jnp.zeros_like(full),) * 3
        eta = np.asarray(dft_tail(left, full, late, (self.omega, 2 * self.omega), self.dt))
        gap = np.abs(1 - np.exp(1j * np.asarray([1.0, 2.0]) * self.omega * self.dt))
        # no underflow to 0 in float32: the static estimate per frequency
        np.testing.assert_allclose(eta, 1e-25 / (gap * np.asarray([1e-20, 1e-22])), rtol=1e-4)


# --------------------------------------------------------------------------- 3. design coverage


class TestDesignRegionMustCoverDevices:
    """Measured before: rel 8.0e-01 (region one cell short), 4.6e-01 (shifted by
    one), 7.0e-01 (two Devices, region over one), all silent."""

    def test_region_one_cell_short_is_refused(self):
        objects, arrays, config = _scene(regions=(("short", (4, 4, 4), (4, 4, 3)),))
        with pytest.raises(ValueError, match=r"'design' \(16 of 64 cells\)"):
            reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon", design_detector="short")

    def test_shifted_region_is_refused(self):
        objects, arrays, config = _scene(regions=(("shifted", (5, 4, 4), (4, 4, 4)),))
        with pytest.raises(ValueError, match="does not cover every Device"):
            reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon", design_detector="shifted")

    def test_second_device_left_out_is_refused(self):
        objects, arrays, config = _scene(devices=(("a", (1, 1, 1), (3, 3, 3)), ("b", (6, 6, 6), (3, 3, 3))))
        with pytest.raises(ValueError, match=r"'b' \(27 of 27 cells\)"):
            reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon", design_detector="a")

    def test_covering_regions_are_accepted(self):
        from fdtdx.adjoint.validation import uncovered_device_cells

        objects, arrays, config = _scene(
            devices=(("a", (1, 1, 1), (3, 3, 3)), ("b", (6, 6, 6), (3, 3, 3))),
            regions=(("big", (0, 0, 0), (5, 5, 5)),),
        )
        assert uncovered_device_cells(objects, [objects["big"].grid_slice_tuple, objects["b"].grid_slice_tuple]) == []
        reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon", design_detector=("big", "b"))

    def test_split_regions_covering_a_device_are_accepted(self):
        objects, arrays, config = _scene(regions=(("lo", (4, 4, 4), (2, 4, 4)), ("hi", (6, 4, 4), (2, 4, 4))))
        reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon", design_detector=("lo", "hi"))

    def test_phasor_level_warns_loudly(self):
        objects, arrays, config = _scene(regions=(("short", (4, 4, 4), (4, 4, 3)),))
        with pytest.warns(UserWarning, match="does not cover every Device"):
            reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon", design_detector="short")


# --------------------------------------------------------------------------- 4. top-level API


class TestTopLevelExports:
    def test_entry_points_are_exported(self):
        from fdtdx.adjoint import api

        assert fdtdx.reciprocity_param_fn is api.reciprocity_param_fn
        assert fdtdx.reciprocity_phasor_fn is api.reciprocity_phasor_fn
        assert {"reciprocity_param_fn", "reciprocity_phasor_fn"} <= set(fdtdx.__all__)

    def test_fresh_import_has_no_cycle(self):
        """fdtdx/__init__ imports fdtdx.adjoint first, whose import chain reaches the
        PML module; a top-level ``from fdtdx import Color`` there is a circular import."""
        env = {**os.environ, "JAX_PLATFORMS": "cpu"}
        code = "import fdtdx; print(fdtdx.reciprocity_param_fn.__module__)"
        done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
        assert done.returncode == 0, done.stderr[-2000:]
        assert done.stdout.strip() == "fdtdx.adjoint.api"


# --------------------------------------------------------------------------- 5. grid, PML, frequencies


def _stretched_x():
    """x widths varying by +-20%, y and z at 50 nm."""
    widths = 50e-9 * (1.0 + 0.2 * np.sin(2 * np.pi * np.arange(_N) / 9.0))
    u = 50e-9 * np.arange(_N + 1.0)
    return RectilinearGrid(x_edges=np.concatenate([[0.0], np.cumsum(widths)]), y_edges=u, z_edges=u)


class TestSceneRefusals:
    """Measured before the refusals (test_production's 24^3 scene, parameter level), nothing
    raised: x widths varying by +-20% rel 1.5e-01 at cosine 0.992, scale 0.91 (+-5%: 3.8e-02;
    one width per axis: TestGeometry, parity); a Device three cells into the PML rel 2.5e-01
    at scale 0.93, all of it on the PML cells (2.2e-07 on the rest; touching it: 3.0e-07)."""

    def test_varying_cell_widths_are_refused(self):
        objects, arrays, config = _scene(grid=_stretched_x())
        with pytest.raises(NotImplementedError, match="cell widths vary along x"):
            reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")

    def test_jitter_below_the_uniform_tolerance_is_accepted(self):
        """y widths 50 nm (1 -+ 0.9e-4): uniform overall (the curls apply no metric), exact, though
        the axis differs from its own first width by more than the tolerance."""
        u = 50e-9 * np.arange(_N + 1.0)
        y = np.concatenate([[0.0], np.cumsum(50e-9 * (1.0 + 0.9e-4 * (-1.0) ** np.arange(1, _N + 1)))])
        grid = RectilinearGrid(x_edges=u, y_edges=y, z_edges=u)
        assert grid._is_uniform and grid._uniform_axes == (True, False, True)
        objects, arrays, config = _scene(grid=grid)
        reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")

    def test_one_width_per_axis_is_accepted(self):
        objects, arrays, config = _scene(grid=QuasiUniformGrid(dx=50e-9, dy=50e-9, dz=40e-9))
        assert config.has_nonuniform_grid
        reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")

    def _pml_scene(self, x_lo):
        """Two PML cells per face; the Device starts at x = ``x_lo``, the monitor clear of the PML."""
        return _scene(pml=2, devices=(("design", (x_lo, 4, 4), (4, 4, 4)),), monitor=((8, 6, 6), (1, 1, 1), ("Ez",)))

    def test_design_region_in_a_pml_is_refused(self):
        objects, arrays, config = self._pml_scene(1)
        with pytest.raises(NotImplementedError, match=r"'design' \(the min_x PML\) reach into a PML"):
            reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")

    def test_design_region_touching_a_pml_is_accepted(self):
        objects, arrays, config = self._pml_scene(2)
        reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")

    @pytest.mark.parametrize("x, strong", [(0, True), (1, False), (2, False)])
    def test_pml_weight_is_the_local_cpml_strength(self, x, strong):
        """Two PML cells: the outer one is lossy, the interface one (sigma = 0 there) is harmless."""
        from fdtdx.adjoint.objective import pml_weight

        objects, _, _ = _scene(pml=2)
        weight = pml_weight(objects, ((x, x + 1), (6, 7), (6, 7)))
        assert (weight is not None and float(weight.max()) > 0.5) == strong

    @pytest.mark.parametrize("x, reaches", [(1, True), (2, False)])
    def test_exact_stencil_reaches_one_cell_further(self, x, reaches):
        """A stock monitor's support starts one cell below it, so at x=1 it touches the lossy PML cell."""
        from fdtdx.adjoint.objective import channel_recordings, pml_weight

        objects, _, config = _scene(pml=2, regions=(("stock", (x, 6, 6), (1, 1, 1)),))
        recs = channel_recordings(objects["stock"], objects, config)
        weights = [pml_weight(objects, b) for rec in recs for b in rec.blocks]
        assert any(w is not None for w in weights) == reaches

    def test_objective_monitors_at_different_frequencies_share_one_solve(self):
        """One adjoint solve over the union of their frequencies, each monitor's cotangent at its own
        rows (refused before). Parity with checkpointed: TestMonitorsAtDifferentFrequencies."""
        from fdtdx.adjoint.objective import objective_frequencies

        other = [fdtdx.WaveCharacter(wavelength=700e-9), fdtdx.WaveCharacter(wavelength=600e-9)]
        objects, arrays, config = _scene(wavelengths=(600e-9,), regions=(("mon2", (9, 6, 6), (1, 1, 1), other),))
        omegas, _, rows = objective_frequencies([objects["mon"], objects["mon2"]])
        assert len(omegas) == 2 and rows[0].tolist() == [0] and rows[1].tolist() == [1, 0]
        fn = reciprocity_phasor_fn(
            arrays, objects, config, _KEY, objective_detectors=("mon", "mon2"), tail_tolerance=None
        )
        mon, mon2 = fn(arrays.inv_permittivities)
        assert mon.shape[1] == 1 and mon2.shape[1] == 2

    def test_a_monitor_named_twice_is_refused(self):
        """Both names drove one adjoint current, and one cotangent was dropped (review 2026-09-25)."""
        objects, arrays, config = _scene()
        with pytest.raises(ValueError, match="more than once"):
            reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors=("mon", "mon"))

    def test_frequencies_written_differently_are_named(self):
        """600e-9 and float32(600e-9), 3.5e-8 apart: no run separates them, and the amplitude solve's advice
        to run longer could not help (review 2026-09-26)."""
        other = [fdtdx.WaveCharacter(wavelength=float(np.float32(600e-9)))]
        objects, arrays, config = _scene(wavelengths=(600e-9,), regions=(("mon2", (9, 6, 6), (1, 1, 1), other),))
        with pytest.raises(ValueError, match="same WaveCharacter"):
            reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors=("mon", "mon2"))

    def test_the_alias_limit_is_above_float32_round_off(self):
        """A float32 run's waveform, evaluated in float32, folds 1.5e-6 of a row from round-off alone."""
        from fdtdx.adjoint.validation import ALIAS_TOLERANCE, alias_limit

        _, _, config = _scene()
        assert alias_limit(config.aset("dtype", jnp.float32)) > 1e-5
        assert alias_limit(config.aset("dtype", jnp.float64)) == ALIAS_TOLERANCE

    def test_frequencies_merge_only_when_stored_equal(self):
        """At rtol 1e-6, frequencies 5e-7 apart shared one row (3e-5 off, unestimated)."""
        from fdtdx.adjoint.objective import objective_frequencies

        for lead, count in ((600e-9, 1), (600e-9 * (1 + 5e-7), 2)):
            other = [fdtdx.WaveCharacter(wavelength=lead)]
            objects, _, _ = _scene(wavelengths=(600e-9,), regions=(("mon2", (9, 6, 6), (1, 1, 1), other),))
            assert len(objective_frequencies([objects["mon"], objects["mon2"]])[0]) == count

    def test_dispersive_device_material_is_refused_at_phasor_level_too(self):
        """apply_params rewrites its ADE coefficients from the design; phasor_fn kept the placed ones (FoM 1.08x)."""
        objects, arrays, config = _scene(device_material=_LORENTZ)
        with pytest.raises(NotImplementedError, match="dispersive"):
            reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")

    def test_inverse_objective_monitor_is_refused(self):
        """It records nothing in a forward run: FoM 0 and checkpointed's gradient 0, ours was nonzero."""
        inv = PhasorDetector(
            name="inv",
            partial_grid_shape=(1, 1, 1),
            wave_characters=[fdtdx.WaveCharacter(wavelength=600e-9)],
            inverse=True,
        )
        objects, arrays, config = _scene(extra=((inv, (9, 6, 6)),))
        with pytest.raises(NotImplementedError, match="inverse=True"):
            reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="inv")


# --------------------------------------------------------------------------- 6. run_fdtd, method="reciprocity"


def _mon_power(out):
    return -jnp.sum(jnp.abs(out.detector_states["mon"]["phasor"]) ** 2)


_LORENTZ = fdtdx.Material(
    permittivity=2.0,
    dispersion=fdtdx.DispersionModel(
        poles=(fdtdx.LorentzPole(resonance_frequency=4e15, damping=2e14, delta_epsilon=0.8),)
    ),
)


def _dipole():
    return fdtdx.PointDipoleSource(
        name="dip", partial_grid_shape=(1, 1, 1), wave_character=fdtdx.WaveCharacter(wavelength=600e-9), polarization=2
    )


class TestRunFdtdReciprocity:
    """``run_fdtd`` with ``GradientConfig(method="reciprocity")`` differentiates phasor detector
    states with respect to ``inv_permittivities`` and ``electric_conductivity``. A figure of merit
    reading anything else, or parameters reaching any other input, raises when the gradient is
    traced (nothing is solved here) instead of getting the silent zero a ``custom_vjp`` would give
    it. Parity: ``TestRunFdtdReciprocity`` in ``tests/simulation/adjoint/test_production.py``."""

    def _trace(self, fom, perturb=None, **scene_kwargs):
        objects, arrays, config, params = _scene(return_params=True, **scene_kwargs)

        def loss(p):
            # set inside the trace, which makes the config's grid edges tracers
            cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))
            arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
            if perturb is not None:
                arrs = perturb(arrs, jax.tree_util.tree_leaves(p)[0])
            return fom(fdtdx.run_fdtd(arrs, objs, cfg, _KEY, show_progress=False)[1])

        return jax.make_jaxpr(jax.grad(loss))(params)

    def test_phasor_objective_traces(self):
        self._trace(_mon_power)

    @pytest.mark.parametrize("leaf", ["fields.E", "fields.H", "inv_permittivities"])
    def test_other_outputs_are_refused(self, leaf):
        def fom(out):
            value = out
            for part in leaf.split("."):
                value = getattr(value, part)
            return _mon_power(out) + jnp.sum(value**2)

        with pytest.raises(NotImplementedError, match=rf"run_fdtd output\(s\) \.{leaf}\b"):
            self._trace(fom)

    @pytest.mark.parametrize("detector", [fdtdx.EnergyDetector, fdtdx.FieldDetector])
    def test_time_domain_detectors_are_refused(self, detector):
        def fom(out):
            return sum(jnp.sum(x) for x in jax.tree_util.tree_leaves(out.detector_states["td"]))

        td = detector(name="td", partial_grid_shape=(1, 1, 1), dtype=jnp.float64)
        with pytest.raises(NotImplementedError, match="does not accumulate complex phasors"):
            self._trace(fom, extra=((td, (2, 2, 2)),))

    def test_other_differentiated_inputs_are_refused(self):
        def perturb(arrays, p):
            return arrays.aset("inv_permeabilities", jnp.ones_like(arrays.inv_permittivities) / (1 + jnp.mean(p)))

        with pytest.raises(NotImplementedError, match=r"arrays\.inv_permeabilities depend"):
            self._trace(_mon_power, perturb)

    @pytest.mark.parametrize(
        "match, scene_kwargs",
        [
            (
                "reach into a PML",
                dict(pml=2, devices=(("design", (1, 4, 4), (4, 4, 4)),), monitor=((8, 6, 6), (1, 1, 1), ("Ez",))),
            ),
            # also with the GradientConfig set inside the trace, where the grid edges are tracers
            ("cell widths vary along x", dict(grid=_stretched_x())),
            ("dispersive", dict(device_material=_LORENTZ)),
            ("overlap the design region", dict(extra=((_dipole(), (5, 5, 5)),))),
        ],
        ids=["device_in_pml", "stretched_grid", "dispersive_device", "source_in_device"],
    )
    def test_scene_refusals_apply(self, match, scene_kwargs):
        """Each silently wrong without its refusal (measured): stretched rel 2.7e-01, Lorentz Device
        3.0e-01, a dipole in the Device 1.4 at cosine -0.05."""
        with pytest.raises(NotImplementedError, match=match):
            self._trace(_mon_power, **scene_kwargs)

    def test_one_width_per_axis_traces(self):
        """A QuasiUniformGrid resolves to a grid that is not uniform overall; with the GradientConfig
        set inside the trace its edges are tracers, which used to be refused."""
        self._trace(_mon_power, grid=QuasiUniformGrid(dx=50e-9, dy=50e-9, dz=40e-9))

    def test_a_raw_material_array_is_refused(self):
        """The gradient is filled inside the Devices only: w.r.t. inv_permittivities itself, 97.8% of
        the checkpointed gradient norm lay outside them and came back a silent zero (rel 0.99)."""
        objects, arrays, config, _ = _scene(return_params=True)
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(inv_eps):
            return _mon_power(fdtdx.run_fdtd(arrays.aset("inv_permittivities", inv_eps), objects, cfg, _KEY)[1])

        with pytest.raises(NotImplementedError, match="not written by apply_params"):
            jax.make_jaxpr(jax.grad(loss))(arrays.inv_permittivities)

    def test_an_array_edited_after_apply_params_is_refused(self):
        """A background parameter or a blur after apply_params: silent zero outside, and a blur also
        corrupted the Device gradient (rel 0.15)."""

        def perturb(arrays, p):
            return arrays.aset("inv_permittivities", arrays.inv_permittivities * (1.0 + 0.0 * jnp.mean(p)))

        with pytest.raises(NotImplementedError, match="not written by apply_params"):
            self._trace(_mon_power, perturb)

    def test_a_background_written_before_apply_params_is_refused(self):
        """A background parameter written into the arrays before apply_params: apply_params' output is
        the very array it returned, but depends on the parameter outside the Devices (rel 0.96, silent
        before its input probe)."""
        objects, arrays, config, params = _scene(return_params=True)
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(p, background):
            arrs = arrays.aset("inv_permittivities", arrays.inv_permittivities / background)
            arrs, objs, _ = apply_params(arrs, objects, p, _KEY)
            return _mon_power(fdtdx.run_fdtd(arrs, objs, cfg, _KEY)[1])

        with pytest.raises(NotImplementedError, match="not written by apply_params"):
            jax.make_jaxpr(jax.grad(loss, argnums=1))(params, 1.2)
        # the same loss differentiated with respect to the Device parameters only traces
        jax.make_jaxpr(jax.grad(loss, argnums=0))(params, 1.2)

    def test_differentiated_dispersion_coefficients_are_refused(self):
        """A differentiated static Lorentz block: checkpointed and finite differences agree (c3: 1e-6),
        this returned zero for c1, c2 and c3."""

        def perturb(arrays, p):
            return arrays.aset("dispersive_c3", arrays.dispersive_c3 * (1.0 + jnp.mean(p)))

        blocks = (("lorentz", (1, 1, 1), (2, 2, 2), _LORENTZ),)
        with pytest.raises(NotImplementedError, match=r"arrays\.dispersive_c3"):
            self._trace(_mon_power, perturb, blocks=blocks)
        # undifferentiated, a dispersive block beside a non-dispersive Device traces
        self._trace(_mon_power, blocks=blocks)

    def test_a_traced_frozen_object_field_is_refused(self):
        """A source amplitude under jax.grad: an UnexpectedTracerError before this refusal."""
        objects, arrays, config, params = _scene(return_params=True, extra=((_dipole(), (1, 6, 6)),))
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(scale):
            arrs, objs, _ = apply_params(arrays, objects, params, _KEY)
            dip = next(o for o in objs.object_list if isinstance(o, fdtdx.PointDipoleSource))
            objs = objs.aset("object_list", [o.aset("amplitude", scale) if o is dip else o for o in objs.object_list])
            return _mon_power(fdtdx.run_fdtd(arrs, objs, cfg, _KEY)[1])

        with pytest.raises(NotImplementedError, match="frozen object fields"):
            jax.make_jaxpr(jax.grad(loss))(1.0)

    @pytest.mark.parametrize("stage", ["jit", "filter_jit", "checkpoint"])
    def test_a_background_under_a_staged_loss_is_refused(self, stage):
        """jax.grad(jax.jit(loss)): the probe's JVP runs only when the staged program is differentiated,
        after apply_params returned; read then, the flag was empty and d/dbg came back 0 (review 2026-09-25)."""
        import equinox as eqx

        objects, arrays, config, params = _scene(return_params=True)
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(p, background):
            arrs = arrays.aset("inv_permittivities", arrays.inv_permittivities / background)
            arrs, objs, _ = apply_params(arrs, objects, p, _KEY)
            # no progress bar: its ordered host callback cannot be rematerialized (natively as well)
            return _mon_power(fdtdx.run_fdtd(arrs, objs, cfg, _KEY, show_progress=False)[1])

        staged = {"jit": jax.jit, "filter_jit": eqx.filter_jit, "checkpoint": jax.checkpoint}[stage](loss)
        # every time, in any order (JAX memoizes the staged rule: a flag read there fired once at most, and
        # after a Device-parameter gradient never)
        jax.make_jaxpr(jax.grad(staged, argnums=0))(params, 1.2)
        for argnums in (1, 1, (0, 1)):
            with pytest.raises(NotImplementedError, match="apply_params"):
                jax.make_jaxpr(jax.grad(staged, argnums=argnums))(params, 1.2)
        jax.make_jaxpr(jax.grad(staged, argnums=0))(params, 1.2)

    def test_a_refusal_leaves_no_flag_for_the_next_gradient(self):
        """A refused differentiation of inv_permeabilities left the probe's flag set, and the next gradient,
        with respect to the Device parameters only, was refused too (review 2026-09-26)."""
        objects, arrays, config, params = _scene(return_params=True)
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(p, s):
            arrs = arrays.aset("inv_permittivities", arrays.inv_permittivities / s)
            arrs = arrs.aset("inv_permeabilities", arrs.inv_permeabilities / s)
            arrs, objs, _ = apply_params(arrs, objects, p, _KEY)
            return _mon_power(fdtdx.run_fdtd(arrs, objs, cfg, _KEY, show_progress=False)[1])

        staged = jax.jit(loss)
        with pytest.raises(NotImplementedError):
            jax.make_jaxpr(jax.grad(staged, argnums=1))(params, 1.05)
        jax.make_jaxpr(jax.grad(staged, argnums=0))(params, 1.05)

    def test_a_jit_value_in_a_frozen_field_under_grad_of_jit_is_refused_clearly(self):
        """jax.grad(jax.jit(loss)) with a jit argument stored in an object field: a bare 'No constant
        handler' TypeError before (jax.jit(jax.grad(loss)) is exact and traces)."""
        wave = fdtdx.WaveCharacter(wavelength=600e-9)
        pulse = fdtdx.GaussianPulseProfile(center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=2.5e14))
        dipole = fdtdx.PointDipoleSource(
            name="dip", partial_grid_shape=(1, 1, 1), wave_character=wave, polarization=2, temporal_profile=pulse
        )
        objects, arrays, config, params = _scene(return_params=True, extra=((dipole, (1, 6, 6)),))
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(p, scale):
            arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
            dip = next(o for o in objs.object_list if isinstance(o, fdtdx.PointDipoleSource))
            objs = objs.aset("object_list", [o.aset("amplitude", scale) if o is dip else o for o in objs.object_list])
            return _mon_power(fdtdx.run_fdtd(arrs, objs, cfg, _KEY, show_progress=False)[1])

        with pytest.raises(NotImplementedError, match="differentiated from outside"):
            jax.make_jaxpr(jax.grad(jax.jit(loss), argnums=0))(params, 0.8)

    def test_a_device_missing_from_the_objects_is_refused(self):
        """apply_params wrote two Devices, run_fdtd got a container with one: the other's gradient was 0."""
        from fdtdx.fdtd.container import ObjectContainer

        objects, arrays, config, params = _scene(
            return_params=True, devices=(("a", (1, 1, 1), (3, 3, 3)), ("b", (6, 6, 6), (3, 3, 3)))
        )
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(p):
            arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
            kept = [o for o in objs.object_list if o.name != "b"]
            volume = next(i for i, o in enumerate(kept) if o is objs.volume)
            return _mon_power(fdtdx.run_fdtd(arrs, ObjectContainer(object_list=kept, volume_idx=volume), cfg, _KEY)[1])

        with pytest.raises(NotImplementedError, match="do not contain"):
            jax.make_jaxpr(jax.grad(loss))(params)

    def test_an_undifferentiated_jit_argument_in_a_frozen_field_traces(self):
        """A source amplitude passed to jax.jit and stored in the objects, the parameters differentiated:
        exact (rel 2.4e-10), and refused before (review 2026-09-25)."""
        wave = fdtdx.WaveCharacter(wavelength=600e-9)
        pulse = fdtdx.GaussianPulseProfile(center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=2.5e14))
        dipole = fdtdx.PointDipoleSource(
            name="dip", partial_grid_shape=(1, 1, 1), wave_character=wave, polarization=2, temporal_profile=pulse
        )
        objects, arrays, config, params = _scene(return_params=True, extra=((dipole, (1, 6, 6)),))
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(p, scale):
            arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
            dip = next(o for o in objs.object_list if isinstance(o, fdtdx.PointDipoleSource))
            objs = objs.aset("object_list", [o.aset("amplitude", scale) if o is dip else o for o in objs.object_list])
            return _mon_power(fdtdx.run_fdtd(arrs, objs, cfg, _KEY)[1])

        jax.make_jaxpr(jax.jit(jax.grad(loss, argnums=0)))(params, 0.8)

    def test_reverse_mode_when_jax_linearizes_by_jvp(self):
        """With jax_use_direct_linearize off, jax.grad reaches run_fdtd as a JVPTracer: not forward mode."""
        objects, arrays, config, params = _scene(return_params=True)
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(p):
            arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
            return _mon_power(fdtdx.run_fdtd(arrs, objs, cfg, _KEY)[1])

        previous = jax.config.jax_use_direct_linearize
        jax.config.update("jax_use_direct_linearize", False)
        try:
            jax.make_jaxpr(jax.grad(loss))(params)
        finally:
            jax.config.update("jax_use_direct_linearize", previous)

    def test_forward_mode_is_refused(self):
        """jax.jvp and jacfwd: a generic TypeError before this refusal (forward over reverse works)."""
        objects, arrays, config, params = _scene(return_params=True)
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(p):
            arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
            return _mon_power(fdtdx.run_fdtd(arrs, objs, cfg, _KEY)[1])

        with pytest.raises(NotImplementedError, match="Forward-mode"):
            jax.jvp(loss, (params,), (params,))

    def test_a_recorder_is_refused(self):
        """Kept after a one-string switch from reversible, it filled the recording every step (13.7x memory)."""
        recorder = fdtdx.Recorder(modules=[fdtdx.DtypeConversion(dtype=jnp.bfloat16)])
        with pytest.raises(Exception, match="drop it for method='reciprocity'"):
            GradientConfig(method="reciprocity", recorder=recorder)


class TestReversibleConductivity:
    """``method="reversible"`` does not differentiate the conductivity, which ``apply_params`` writes
    for a lossy Device. A gradient through one raises here instead of an UnexpectedTracerError; a
    constant one (a lossless Device, loss elsewhere or under it) and a forward run are unaffected."""

    def _trace(self, grad, **scene_kwargs):
        cfg = GradientConfig(method="reversible", recorder=Recorder(modules=[]))
        objects, arrays, config, params = _scene(return_params=True, gradient_config=cfg, **scene_kwargs)

        def loss(p):
            arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
            return _mon_power(fdtdx.run_fdtd(arrs, objs, config, _KEY, show_progress=False)[1])

        return jax.make_jaxpr(jax.grad(loss) if grad else loss)(params)

    def test_lossy_device_gradient_is_refused(self):
        with pytest.raises(NotImplementedError, match="does not differentiate the electric conductivity"):
            self._trace(True, device_material=_LOSSY_SI)

    def test_lossy_device_forward_runs(self):
        self._trace(False, device_material=_LOSSY_SI)

    @pytest.mark.parametrize("lower", [(4, 4, 4), (0, 0, 0)], ids=["under", "elsewhere"])
    def test_lossless_device_in_a_lossy_scene_differentiates(self, lower):
        self._trace(True, blocks=(("loss", lower, (4, 4, 4), _LOSSY_SI),))

    @pytest.mark.parametrize(
        "lower, refused", [((4, 4, 4), True), ((0, 0, 0), False)], ids=["over_loss", "loss_elsewhere"]
    )
    def test_etched_device_differentiates_unless_it_etches_loss(self, lower, refused):
        """Etching air into a lossless background leaves the conductivity as placed (upstream: exact)."""
        etched = fdtdx.Device(
            name="design",
            partial_grid_shape=(4, 4, 4),
            partial_voxel_grid_shape=(1, 1, 1),
            materials={"air": fdtdx.Material()},
            param_transforms=[],
            use_etching=True,
        )
        kwargs = dict(devices=(), extra=((etched, (4, 4, 4)),), blocks=(("loss", lower, (4, 4, 4), _LOSSY_SI),))
        if refused:
            with pytest.raises(NotImplementedError, match="does not differentiate the electric conductivity"):
                self._trace(True, **kwargs)
        else:
            self._trace(True, **kwargs)

    def test_a_lossy_device_written_after_the_etch_leaves_it_alone(self):
        """apply_params writes Devices in order: a later lossy Device over the etched one overwrites
        the shared cells with its own (constant) conductivity, which the etch never reads."""
        etched = fdtdx.Device(
            name="etch",
            partial_grid_shape=(4, 4, 4),
            partial_voxel_grid_shape=(1, 1, 1),
            materials={"air": fdtdx.Material()},
            param_transforms=[],
            use_etching=True,
        )
        shared = fdtdx.Material(permittivity=4.0, electric_conductivity=1e5)
        lossy = fdtdx.Device(
            name="lossy",
            partial_grid_shape=(4, 4, 4),
            partial_voxel_grid_shape=(1, 1, 1),
            materials={"a": _LOSSY_SI, "b": shared},
            param_transforms=[],
        )
        self._trace(True, devices=(), extra=((etched, (3, 4, 4)), (lossy, (5, 4, 4))))
