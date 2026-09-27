"""What ``GradientConfig(method="reciprocity")`` refuses, and what it must not refuse.

Each refusal was measured silently wrong before it existed (numbers in ``notes/adjoint/05-guards.md``).
They raise when the gradient is traced, so most tests only trace it (``jax.make_jaxpr``) on a 12^3
scene without a source. The gradient parity behind each is in ``tests/simulation/adjoint/test_parity.py``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.adjoint.objective import channel_recordings, lossy_injection, objective_frequencies, pml_weight
from fdtdx.adjoint.source import AdjointCurrentSource
from fdtdx.adjoint.validation import ALIAS_TOLERANCE, alias_limit
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.core.grid import QuasiUniformGrid, RectilinearGrid, UniformGrid
from fdtdx.fdtd.initialization import apply_params
from fdtdx.fdtd.update import update_E, update_H
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.detectors.phasor import PhasorDetector

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
):
    """12^3 scene, no source, and no boundaries unless ``pml`` cells of PML: ``(objects, arrays, config,
    params)``. ``regions`` are extra stock detectors ``(name, lower, shape[, wave_characters])``; ``blocks``
    are ``(name, lower, shape, material)``; ``extra`` are ``(object, lower)``. ``grid`` defaults to 50 nm
    cubes; ``device_material`` replaces the Devices' ``si`` (eps 2.25)."""
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
        name="mon", partial_grid_shape=mon_shape, wave_characters=wcs, components=mon_comps, exact_interpolation=False
    )
    at(mon, mon_lo)
    for name, lo, shape, *own in regions:
        at(PhasorDetector(name=name, partial_grid_shape=shape, wave_characters=own[0] if own else wcs), lo)
    for obj, lo in extra:
        at(obj, lo)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objs, config=config, constraints=cons, key=_KEY
    )
    return objects, arrays, config, params


def _power(name="mon"):
    return lambda out: -jnp.sum(jnp.abs(out.detector_states[name]["phasor"]) ** 2)


def _loss(objects, arrays, config, fom, perturb=None, inside=False):
    """``apply_params -> run_fdtd(reciprocity) -> fom``. ``inside`` sets the GradientConfig inside the trace,
    which makes the config's grid edges tracers; ``perturb(arrays, p)`` edits the applied arrays."""
    reciprocity = GradientConfig(method="reciprocity")
    outside = config.aset("gradient_config", reciprocity)

    def loss(p):
        cfg = config.aset("gradient_config", reciprocity) if inside else outside
        arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
        if perturb is not None:
            arrs = perturb(arrs, jax.tree_util.tree_leaves(p)[0])
        return fom(fdtdx.run_fdtd(arrs, objs, cfg, _KEY, show_progress=False)[1])

    return loss


def _trace(fom=None, perturb=None, inside=False, **scene_kwargs):
    objects, arrays, config, params = _scene(**scene_kwargs)
    loss = _loss(objects, arrays, config, fom or _power(), perturb, inside)
    return jax.make_jaxpr(jax.grad(loss))(params)


def _run(fom=None, inside=False, **scene_kwargs):
    """The gradient under ``jax.jit``, computed: refusals of traced values raise when it runs."""
    objects, arrays, config, params = _scene(**scene_kwargs)
    grad = jax.jit(jax.grad(_loss(objects, arrays, config, fom or _power(), inside=inside)))
    return jax.block_until_ready(grad(params))


def _dipole(**kwargs):
    return fdtdx.PointDipoleSource(
        name="dip",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=600e-9),
        polarization=2,
        **kwargs,
    )


def _stretched_x():
    """x widths varying by +-20%, y and z at 50 nm."""
    widths = 50e-9 * (1.0 + 0.2 * np.sin(2 * np.pi * np.arange(_N) / 9.0))
    u = 50e-9 * np.arange(_N + 1.0)
    return RectilinearGrid(x_edges=np.concatenate([[0.0], np.cumsum(widths)]), y_edges=u, z_edges=u)


_LOSSY = fdtdx.Material(permittivity=2.25, permeability=2.0, electric_conductivity=1e5, magnetic_conductivity=6e9)
_LORENTZ = fdtdx.Material(
    permittivity=2.0,
    dispersion=fdtdx.DispersionModel(
        poles=(fdtdx.LorentzPole(resonance_frequency=4e15, damping=2e14, delta_epsilon=0.8),)
    ),
)


class TestLossyInjection:
    """The adjoint current sits where the objective monitor reads the fields. FDTDX adds every source
    after the lossy division by 1 + a, so in a lossy cell the current is 1 + a times its reciprocal
    partner and the cotangent is divided back. Measured before: rel 2.39e-01 at cosine 1.0 (scale 1 + a)."""

    def _lossy_scene(self):
        # lossy block on x 5..6 only; the probed block x 4..7 is half lossless
        objects, arrays, config, _ = _scene(devices=(), blocks=(("loss", (5, 4, 4), (2, 4, 4), _LOSSY),))
        return objects, arrays, config

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
        return arrays, config, 2.0 / (1.0 + np.asarray(ratio))

    @pytest.mark.parametrize("family", ["E", "H"])
    def test_divisor_is_fdtdx_own_lossy_factor(self, family):
        arrays, config, factor = self._fdtdx_factor(family)
        comps = ("Ex", "Ey", "Ez") if family == "E" else ("Hx", "Hy", "Hz")
        loss = lossy_injection(arrays, float(config.courant_number), ((4, 8), (4, 8), (4, 8)), comps)
        assert loss is not None
        got = np.asarray(loss.divisor(arrays.inv_permittivities, arrays.electric_conductivity))
        want = factor[:, 4:8, 4:8, 4:8]
        np.testing.assert_allclose(got, want, rtol=1e-12)
        assert want.max() > 1.1  # the lossy cells are really lossy ...
        np.testing.assert_allclose(want[:, 0], 1.0, rtol=1e-12)  # ... and x = 4 really is not

    def test_a_lossless_block_divides_by_exactly_one(self):
        _, arrays, config = self._lossy_scene()
        ie, sigma = arrays.inv_permittivities, arrays.electric_conductivity
        lossless = lossy_injection(arrays, float(config.courant_number), ((8, 10), (4, 8), (4, 8)), ("Ez", "Hx"))
        assert lossless is not None and np.all(np.asarray(lossless.divisor(ie, sigma)) == 1.0)

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


class TestSceneRefusals:
    """Measured before the refusals (test_parity's 24^3 scene, parameter level), nothing raised: x widths
    varying by +-20% rel 1.5e-01 at cosine 0.992; a Device three cells into the PML rel 2.5e-01, all of it
    on the PML cells; a Lorentz Device 3.0e-01; a dipole in the Device 1.4 at cosine -0.05; a Bloch wave
    vector 1.24 at cosine 0.35; an objective recording from 20 fs 1.8e-01."""

    def test_varying_cell_widths_are_refused(self):
        with pytest.raises(NotImplementedError, match="cell widths vary along x"):
            _trace(grid=_stretched_x())

    def test_varying_cell_widths_of_a_traced_grid_are_refused(self):
        """``config.aset`` inside the trace makes the grid edges tracers: refused when the gradient runs."""
        with pytest.raises(Exception, match="cell widths vary along an axis"):
            _run(inside=True, grid=_stretched_x())

    @pytest.mark.parametrize("inside", [False, True], ids=["concrete", "traced"])
    def test_one_width_per_axis_is_accepted(self, inside):
        _run(inside=inside, grid=QuasiUniformGrid(dx=50e-9, dy=50e-9, dz=40e-9))

    def test_jitter_below_the_uniform_tolerance_is_accepted(self):
        """y widths 50 nm (1 -+ 0.9e-4): uniform overall (the curls apply no metric), and exact."""
        u = 50e-9 * np.arange(_N + 1.0)
        y = np.concatenate([[0.0], np.cumsum(50e-9 * (1.0 + 0.9e-4 * (-1.0) ** np.arange(1, _N + 1)))])
        grid = RectilinearGrid(x_edges=u, y_edges=y, z_edges=u)
        assert grid._is_uniform
        _trace(grid=grid)

    def _pml_scene(self, x_lo):
        """Two PML cells per face; the Device starts at x = ``x_lo``, the monitor clear of the PML."""
        return dict(pml=2, devices=(("design", (x_lo, 4, 4), (4, 4, 4)),), monitor=((8, 6, 6), (1, 1, 1), ("Ez",)))

    def test_a_device_in_a_pml_is_refused(self):
        with pytest.raises(NotImplementedError, match=r"'design' \(the min_x PML\) reach into a PML"):
            _trace(**self._pml_scene(1))

    def test_a_device_touching_a_pml_is_accepted(self):
        _trace(**self._pml_scene(2))

    @pytest.mark.parametrize("x, strong", [(0, True), (1, False), (2, False)])
    def test_pml_weight_is_the_local_cpml_strength(self, x, strong):
        """Two PML cells: the outer one is lossy, the interface one (sigma = 0 there) is harmless."""
        objects, _, _, _ = _scene(pml=2)
        weight = pml_weight(objects, ((x, x + 1), (6, 7), (6, 7)))
        assert (weight is not None and float(weight.max()) > 0.5) == strong

    @pytest.mark.parametrize("x, reaches", [(1, True), (2, False)])
    def test_exact_stencil_reaches_one_cell_further(self, x, reaches):
        """A stock monitor's support starts one cell below it, so at x=1 it touches the lossy PML cell."""
        objects, _, config, _ = _scene(pml=2, regions=(("stock", (x, 6, 6), (1, 1, 1)),))
        recs = channel_recordings(objects["stock"], objects, config)
        weights = [pml_weight(objects, b) for rec in recs for b in rec.blocks]
        assert any(w is not None for w in weights) == reaches

    def test_a_dispersive_device_material_is_refused(self):
        with pytest.raises(NotImplementedError, match="dispersive"):
            _trace(device_material=_LORENTZ)

    def test_a_stock_source_inside_a_device_is_refused(self):
        """Stock sources freeze the inv_eps of their injection, so run_fdtd's gradient omits that term."""
        with pytest.raises(NotImplementedError, match="inject inside a Device"):
            _trace(extra=((_dipole(), (5, 5, 5)),))

    def test_a_scene_without_devices_is_refused(self):
        """The gradient is taken inside the Devices: without one, inv_permittivities differentiated directly
        would get a zero gradient everywhere."""
        objects, arrays, config, _ = _scene(devices=())
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(inv_eps):
            arrs = arrays.aset("inv_permittivities", inv_eps)
            return _power()(fdtdx.run_fdtd(arrs, objects, cfg, _KEY, show_progress=False)[1])

        with pytest.raises(ValueError, match="the scene has none"):
            jax.make_jaxpr(jax.grad(loss))(arrays.inv_permittivities)

    def test_a_bloch_wave_vector_is_refused(self):
        objects, arrays, config, params = _scene()
        bloch = BlochBoundary(
            name="bloch_min_x", axis=0, direction="-", partial_grid_shape=(1, _N, _N), bloch_vector=(2e6, 0.0, 0.0)
        ).place_on_grid(((0, 1), (0, _N), (0, _N)), config, _KEY)
        objects = objects.aset("object_list", [*objects.object_list, bloch])
        with pytest.raises(NotImplementedError, match="Bloch"):
            jax.make_jaxpr(jax.grad(_loss(objects, arrays, config, _power())))(params)

    @pytest.mark.parametrize("name", ["inv_permittivities", "electric_conductivity"])
    def test_a_full_material_tensor_is_refused(self, name):
        """The adjoint current would inject xx, xy, xz instead of the diagonal."""
        tensor = jnp.ones((9, _N, _N, _N), dtype=jnp.float64)

        def perturb(arrays, p):
            if name == "inv_permittivities":
                return arrays.aset(name, tensor * (1.0 + 0.0 * jnp.mean(p)))
            return arrays.aset(name, tensor)

        with pytest.raises(NotImplementedError, match=f"{name} as a full 3x3 tensor"):
            _trace(perturb=perturb, blocks=(("loss", (0, 0, 0), (2, 2, 2), _LOSSY),))

    def test_an_objective_recording_part_of_the_run_is_refused(self):
        objects, arrays, config, params = _scene()
        mon = objects["mon"]
        late = mon.aset("switch", fdtdx.OnOffSwitch(start_time=5e-15))
        late = late.aset("_grid_slice_tuple", ((-1, -1), (-1, -1), (-1, -1)))
        late = late.place_on_grid(mon.grid_slice_tuple, config, _KEY)
        assert late._num_time_steps_on < config.time_steps_total
        objects = objects.aset("object_list", [late if o is mon else o for o in objects.object_list])
        with pytest.raises(NotImplementedError, match="time steps"):
            jax.make_jaxpr(jax.grad(_loss(objects, arrays, config, _power())))(params)

    def test_an_inverse_objective_monitor_is_refused(self):
        """It records nothing in a forward run: FoM 0 and checkpointed's gradient 0, ours was nonzero."""
        inv = PhasorDetector(
            name="inv",
            partial_grid_shape=(1, 1, 1),
            wave_characters=[fdtdx.WaveCharacter(wavelength=600e-9)],
            inverse=True,
        )
        with pytest.raises(NotImplementedError, match="inverse=True"):
            _trace(_power("inv"), extra=((inv, (9, 6, 6)),))

    def test_mid_range_condition_is_refused(self):
        # 5 fs, 600 and 620 nm: cond 1.9e4, which the former 1e8 default accepted
        with pytest.raises(ValueError, match="ill-conditioned"):
            _trace(time=5e-15, wavelengths=(600e-9, 620e-9))

    def test_a_stride_folding_the_source_spectrum_is_refused(self):
        """A stride folding a source band onto the objective frequency gave cosine -0.63 (rel 1.0, no
        warning); the same illumination recorded every step, rel 1.6e-6."""

        def dipole(frequency):
            wc = fdtdx.WaveCharacter(frequency=frequency)
            return fdtdx.PointDipoleSource(
                name="src",
                partial_grid_shape=(1, 1, 1),
                wave_character=wc,
                temporal_profile=fdtdx.GaussianPulseProfile(
                    center_wave=wc, spectral_width=fdtdx.WaveCharacter(frequency=0.1 * frequency)
                ),
                polarization=2,
            )

        def strided_run(frequency, stride):
            objects, arrays, config, params = _scene(time=200e-15, extra=((dipole(frequency), (1, 6, 6)),))
            mon = objects["mon"]
            det = PhasorDetector(
                name="mon",
                partial_grid_shape=(1, 1, 1),
                wave_characters=mon.wave_characters,
                components=mon.components,
                exact_interpolation=False,
                dft_subsample=stride,
            ).place_on_grid(mon.grid_slice_tuple, config, _KEY)
            objects = objects.aset("object_list", [det if o is mon else o for o in objects.object_list])
            arrays = arrays.aset("detector_states", {**arrays.detector_states, "mon": det.init_state()})
            config = config.aset("gradient_config", GradientConfig(method="reciprocity", tail_tolerance=None))

            def loss(p):
                arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
                return _power()(fdtdx.run_fdtd(arrs, objs, config, _KEY, show_progress=False)[1])

            return jax.block_until_ready(jax.grad(loss)(params)), config

        f0 = 299792458.0 / 600e-9
        stride = 5
        _, config = strided_run(f0, stride)  # a source band clear of every image is fine at the same stride
        # a source band at 1 / (stride dt) - f0 folds onto f0: refused (an exactness check, also at
        # tail_tolerance=None)
        with pytest.raises(Exception, match="dft_subsample"):
            strided_run(1.0 / (stride * float(config.time_step_duration)) - f0, stride)


class TestFrequencies:
    def test_objective_monitors_at_different_frequencies_share_one_solve(self):
        """One adjoint solve over the union of their frequencies, each monitor's cotangent at its own rows.
        Parity with checkpointed: TestMonitorsAtDifferentFrequencies in test_parity.py."""
        other = [fdtdx.WaveCharacter(wavelength=700e-9), fdtdx.WaveCharacter(wavelength=600e-9)]
        scene = dict(wavelengths=(600e-9,), regions=(("mon2", (9, 6, 6), (1, 1, 1), other),))
        objects, _, _, _ = _scene(**scene)
        omegas, _, rows = objective_frequencies([objects["mon"], objects["mon2"]])
        assert len(omegas) == 2 and rows[0].tolist() == [0] and rows[1].tolist() == [1, 0]
        _trace(lambda out: _power()(out) + _power("mon2")(out), **scene)

    def test_frequencies_written_differently_are_named(self):
        """600e-9 and float32(600e-9), 3.5e-8 apart: no run separates them, and the amplitude solve's advice
        to run longer could not help."""
        other = [fdtdx.WaveCharacter(wavelength=float(np.float32(600e-9)))]
        with pytest.raises(ValueError, match="same WaveCharacter"):
            _trace(
                lambda out: _power()(out) + _power("mon2")(out),
                wavelengths=(600e-9,),
                regions=(("mon2", (9, 6, 6), (1, 1, 1), other),),
            )

    def test_frequencies_merge_only_when_stored_equal(self):
        """At rtol 1e-6, frequencies 5e-7 apart shared one row (3e-5 off, unestimated)."""
        for lead, count in ((600e-9, 1), (600e-9 * (1 + 5e-7), 2)):
            other = [fdtdx.WaveCharacter(wavelength=lead)]
            objects, _, _, _ = _scene(wavelengths=(600e-9,), regions=(("mon2", (9, 6, 6), (1, 1, 1), other),))
            assert len(objective_frequencies([objects["mon"], objects["mon2"]])[0]) == count

    def test_the_alias_limit_is_above_float32_round_off(self):
        """A float32 run's waveform, evaluated in float32, folds 1.5e-6 of a row from round-off alone."""
        _, _, config, _ = _scene()
        assert alias_limit(config.aset("dtype", jnp.float32)) > 1e-5
        assert alias_limit(config.aset("dtype", jnp.float64)) == ALIAS_TOLERANCE


class TestRunFdtdReciprocity:
    """``run_fdtd`` with ``GradientConfig(method="reciprocity")`` differentiates phasor detector states with
    respect to ``inv_permittivities``. A figure of merit reading anything else, or parameters reaching any
    other input, raises when the gradient is traced instead of getting the silent zero a ``custom_vjp``
    would give it."""

    def test_phasor_objective_traces(self):
        _trace()

    @pytest.mark.parametrize("leaf", ["fields.E", "fields.H", "inv_permittivities"])
    def test_other_outputs_are_refused(self, leaf):
        def fom(out):
            value = out
            for part in leaf.split("."):
                value = getattr(value, part)
            return _power()(out) + jnp.sum(value**2)

        with pytest.raises(NotImplementedError, match=rf"run_fdtd output\(s\) \.{leaf}\b"):
            _trace(fom)

    @pytest.mark.parametrize("detector", [fdtdx.EnergyDetector, fdtdx.FieldDetector])
    def test_time_domain_detectors_are_refused(self, detector):
        def fom(out):
            return sum(jnp.sum(x) for x in jax.tree_util.tree_leaves(out.detector_states["td"]))

        td = detector(name="td", partial_grid_shape=(1, 1, 1), dtype=jnp.float64)
        with pytest.raises(NotImplementedError, match="does not accumulate complex phasors"):
            _trace(fom, extra=((td, (2, 2, 2)),))

    @pytest.mark.parametrize("name", ["inv_permeabilities", "electric_conductivity"])
    def test_other_differentiated_inputs_are_refused(self, name):
        def perturb(arrays, p):
            return arrays.aset(name, getattr(arrays, name) / (1 + jnp.mean(p)))

        with pytest.raises(NotImplementedError, match=rf"arrays\.{name} depend"):
            _trace(perturb=perturb, blocks=(("loss", (0, 0, 0), (2, 2, 2), _LOSSY),))

    def test_a_continuous_device_in_a_dispersive_scene_traces(self):
        """apply_params writes the Device cells' dispersion coefficients as a blend of equal rows, traced but
        with zero derivative: not refused."""
        _trace(blocks=(("lorentz", (1, 1, 1), (2, 2, 2), _LORENTZ),))

    @pytest.mark.parametrize("stage", ["jit", "filter_jit", "checkpoint"])
    def test_staged_losses_trace(self, stage):
        objects, arrays, config, params = _scene()
        loss = _loss(objects, arrays, config, _power())
        staged = {"jit": jax.jit, "filter_jit": eqx.filter_jit, "checkpoint": jax.checkpoint}[stage](loss)
        jax.make_jaxpr(jax.grad(staged))(params)

    def test_an_undifferentiated_jit_argument_in_a_frozen_field_traces(self):
        """A source amplitude passed to jax.jit and stored in the objects, the parameters differentiated:
        exact (rel 2.4e-10)."""
        wave = fdtdx.WaveCharacter(wavelength=600e-9)
        pulse = fdtdx.GaussianPulseProfile(center_wave=wave, spectral_width=fdtdx.WaveCharacter(frequency=2.5e14))
        objects, arrays, config, params = _scene(extra=((_dipole(temporal_profile=pulse), (1, 6, 6)),))
        cfg = config.aset("gradient_config", GradientConfig(method="reciprocity"))

        def loss(p, scale):
            arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
            dip = next(o for o in objs.object_list if isinstance(o, fdtdx.PointDipoleSource))
            objs = objs.aset("object_list", [o.aset("amplitude", scale) if o is dip else o for o in objs.object_list])
            return _power()(fdtdx.run_fdtd(arrs, objs, cfg, _KEY, show_progress=False)[1])

        jax.make_jaxpr(jax.jit(jax.grad(loss, argnums=0)))(params, 0.8)

    def test_reverse_mode_when_jax_linearizes_by_jvp(self):
        previous = jax.config.jax_use_direct_linearize
        jax.config.update("jax_use_direct_linearize", False)
        try:
            _trace()
        finally:
            jax.config.update("jax_use_direct_linearize", previous)

    def test_forward_mode_raises(self):
        """A custom_vjp has no forward-mode rule: jax.jvp and jacfwd raise JAX's own error, as for any
        custom_vjp (forward over reverse, jax.hessian, works)."""
        objects, arrays, config, params = _scene()
        with pytest.raises(TypeError, match="custom_vjp"):
            jax.jvp(_loss(objects, arrays, config, _power()), (params,), (params,))

    def test_a_recorder_is_refused(self):
        """Kept after a one-string switch from reversible, it filled the recording every step (13.7x memory)."""
        recorder = fdtdx.Recorder(modules=[fdtdx.DtypeConversion(dtype=jnp.bfloat16)])
        with pytest.raises(Exception, match="drop it for method='reciprocity'"):
            GradientConfig(method="reciprocity", recorder=recorder)
