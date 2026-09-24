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
    return_params=False,
):
    """12^3 scene, no source, and no boundaries unless ``pml`` cells of PML. ``regions`` are
    extra stock detectors ``(name, lower, shape[, wave_characters])``; ``blocks`` are
    ``(name, lower, shape, material)``; ``extra`` are ``(object, lower)``. ``grid`` defaults
    to 50 nm cubes."""
    config = SimulationConfig(time=time, grid=grid or UniformGrid(spacing=50e-9), backend="cpu", dtype=jnp.float64)
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
            materials={"air": fdtdx.Material(permittivity=1.0), "si": fdtdx.Material(permittivity=2.25)},
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


class TestSceneRefusals:
    """Measured before the refusals (test_production's 24^3 scene, parameter level), nothing
    raised: x widths varying by +-20% rel 1.5e-01 at cosine 0.992, scale 0.91 (+-5%: 3.8e-02;
    one width per axis: TestGeometry, parity); a Device three cells into the PML rel 2.5e-01
    at scale 0.93, all of it on the PML cells (2.2e-07 on the rest; touching it: 3.0e-07)."""

    def test_varying_cell_widths_are_refused(self):
        widths = 50e-9 * (1.0 + 0.2 * np.sin(2 * np.pi * np.arange(_N) / 9.0))
        x = np.concatenate([[0.0], np.cumsum(widths)])
        u = 50e-9 * np.arange(_N + 1.0)
        objects, arrays, config = _scene(grid=RectilinearGrid(x_edges=x, y_edges=u, z_edges=u))
        with pytest.raises(NotImplementedError, match="cell widths vary along x"):
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

    def test_objective_monitors_must_share_frequencies(self):
        """They share one amplitude solve; the second would be driven at the first one's frequency."""
        other = [fdtdx.WaveCharacter(wavelength=650e-9)]
        objects, arrays, config = _scene(wavelengths=(600e-9,), regions=(("mon2", (9, 6, 6), (1, 1, 1), other),))
        with pytest.raises(ValueError, match="must share frequencies"):
            reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors=("mon", "mon2"))


# --------------------------------------------------------------------------- 6. run_fdtd, method="reciprocity"


def _mon_power(out):
    return -jnp.sum(jnp.abs(out.detector_states["mon"]["phasor"]) ** 2)


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

    def test_scene_refusals_apply(self):
        with pytest.raises(NotImplementedError, match="reach into a PML"):
            self._trace(
                _mon_power, pml=2, devices=(("design", (1, 4, 4), (4, 4, 4)),), monitor=((8, 6, 6), (1, 1, 1), ("Ez",))
            )
