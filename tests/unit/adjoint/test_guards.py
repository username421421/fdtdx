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
from fdtdx.adjoint.reciprocity import solve_adjoint_amplitudes
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.update import update_E, update_H
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


def _at(obj, lo):
    return obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=lo)


def _scene(
    *,
    devices=(("design", (4, 4, 4), (4, 4, 4)),),
    regions=(),
    blocks=(),
    time=10e-15,
    wavelengths=(600e-9, 700e-9),
    monitor=((10, 6, 6), (1, 1, 1), ("Ez",)),
    return_params=False,
):
    """12^3 scene, no boundaries and no source. ``regions`` are extra detectors
    ``(name, lower, shape)``; ``blocks`` are ``(name, lower, shape, material)``."""
    config = SimulationConfig(time=time, grid=UniformGrid(spacing=50e-9), backend="cpu", dtype=jnp.float64)
    objs, cons = [], []
    vol = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
    objs.append(vol)
    for name, lo, shape, material in blocks:
        blk = fdtdx.UniformMaterialObject(name=name, partial_grid_shape=shape, material=material)
        cons.append(_at(blk, lo))
        objs.append(blk)
    for name, lo, shape in devices:
        dev = fdtdx.Device(
            name=name,
            partial_grid_shape=shape,
            partial_voxel_grid_shape=(1, 1, 1),
            materials={"air": fdtdx.Material(permittivity=1.0), "si": fdtdx.Material(permittivity=2.25)},
            param_transforms=[],
        )
        cons.append(_at(dev, lo))
        objs.append(dev)
    wcs = [fdtdx.WaveCharacter(wavelength=w) for w in wavelengths]
    mon_lo, mon_shape, mon_comps = monitor
    mon = PhasorDetector(
        name="mon",
        partial_grid_shape=mon_shape,
        wave_characters=wcs,
        components=mon_comps,
        exact_interpolation=False,
    )
    cons.append(_at(mon, mon_lo))
    objs.append(mon)
    for name, lo, shape in regions:
        det = PhasorDetector(name=name, partial_grid_shape=shape, wave_characters=wcs)
        cons.append(_at(det, lo))
        objs.append(det)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objs, config=config, constraints=cons, key=_KEY
    )
    return (objects, arrays, config, params) if return_params else (objects, arrays, config)


# --------------------------------------------------------------------------- 1. lossy monitor


_LOSSY = fdtdx.Material(permittivity=2.25, permeability=2.0, electric_conductivity=1e5, magnetic_conductivity=6e9)


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
        from fdtdx.adjoint.vjp import _lossy_injection

        _, arrays, config, factor = self._fdtdx_factor(family)
        block = ((4, 8), (4, 8), (4, 8))
        comps = ("Ex", "Ey", "Ez") if family == "E" else ("Hx", "Hy", "Hz")
        loss = _lossy_injection(arrays, float(config.courant_number), block, comps)
        assert loss is not None
        got = np.asarray(loss.divisor(arrays.inv_permittivities))
        want = factor[:, 4:8, 4:8, 4:8]
        np.testing.assert_allclose(got, want, rtol=1e-12)
        assert want.max() > 1.1  # the lossy cells are really lossy ...
        np.testing.assert_allclose(want[:, 0], 1.0, rtol=1e-12)  # ... and x = 4 really is not

    def test_lossless_block_needs_no_divisor(self):
        from fdtdx.adjoint.vjp import _lossy_injection

        _, arrays, config = self._lossy_scene()
        assert _lossy_injection(arrays, float(config.courant_number), ((8, 10), (4, 8), (4, 8)), ("Ez", "Hx")) is None

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


class TestDispersionUnderDevice:
    """A dispersive block under a non-dispersive Device: apply_params zeroes its ADE
    coefficients in the Device cells on every call, the reciprocity solves did not.
    Measured before: forward FoM off by 60%, gradient rel 8.3e-01 at cosine 0.66."""

    def test_zeroing_matches_apply_params(self):
        from fdtdx.adjoint.api import device_dispersion_as_applied
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
        from fdtdx.adjoint.reciprocity import DEFAULT_COND_LIMIT

        assert DEFAULT_COND_LIMIT == 1e4

    def test_mid_range_condition_is_refused_by_default(self):
        # 5 fs, 600 and 620 nm: cond 1.9e4, which the former 1e8 default accepted
        objects, arrays, config = _scene(time=5e-15, wavelengths=(600e-9, 620e-9))
        with pytest.raises(ValueError, match="ill-conditioned"):
            reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")
        reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon", cond_limit=1e8)

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

    def _signal(self, tau_steps, T=400):
        n = np.arange(T + 1)
        e = np.cos(self.omega * n * self.dt) * np.exp(-n / tau_steps)
        phasor = np.sum(np.exp(1j * self.omega * n[:T] * self.dt) * e[:T])
        return jnp.asarray(e[T]).reshape(1, 1), jnp.asarray(phasor).reshape(1, 1, 1)

    def test_decayed_field_has_a_negligible_tail(self):
        from fdtdx.adjoint import dft_tail

        left, phasor = self._signal(tau_steps=20.0)
        assert float(dft_tail(left, phasor, (self.omega,), self.dt)) < 1e-6

    def test_undecayed_field_has_a_large_tail(self):
        from fdtdx.adjoint import dft_tail

        # ten periods of an undamped carrier: 1 / (|1 - e^{i w dt}| * T / 2) = 3.2e-02
        left, phasor = self._signal(tau_steps=1e6, T=200)
        assert float(dft_tail(left, phasor, (self.omega,), self.dt)) > 3e-2

    def test_frequencies_without_a_phasor_are_ignored(self):
        from fdtdx.adjoint import dft_tail

        left = jnp.ones((1, 1))
        phasor = jnp.asarray([[[100.0 + 0j]], [[1e-9 + 0j]]])  # second frequency: nothing recorded
        eta = float(dft_tail(left, phasor, (self.omega, 2 * self.omega), self.dt))
        gap = abs(1 - np.exp(1j * self.omega * self.dt))
        np.testing.assert_allclose(eta, 1.0 / (gap * 100.0), rtol=1e-12)


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
        from fdtdx.adjoint.vjp import uncovered_device_cells

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
