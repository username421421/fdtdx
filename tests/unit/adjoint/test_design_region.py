"""The design region needs no user setup: structure of the internal scene.

No solves here, only placement, so these run in CI in well under a second. The
gradient parity of the same code paths is covered in
``tests/simulation/adjoint/test_production.py``.
"""

import jax
import jax.numpy as jnp
import pytest

import fdtdx
from fdtdx.adjoint.scene import (
    DESIGN_DETECTOR_PREFIX,
    DESIGN_DETECTOR_SETTINGS,
    canonical_components,
    design_regions,
    internal_scene,
)
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.objects.detectors.phasor import PhasorDetector

_KEY = jax.random.PRNGKey(0)
_N = 12
_WL = (600e-9, 700e-9)


def _scene(*, devices=(("design", (4, 4, 4), (4, 4, 4)),), stock_detector=True, dtype=jnp.float64):
    config = SimulationConfig(time=10e-15, grid=UniformGrid(spacing=50e-9), backend="cpu", dtype=dtype)
    objs, cons = [], []
    vol = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
    objs.append(vol)
    for name, lo, span in devices:
        dev = fdtdx.Device(
            name=name,
            partial_grid_shape=span,
            partial_voxel_grid_shape=(1, 1, 1),
            materials={"air": fdtdx.Material(permittivity=1.0), "si": fdtdx.Material(permittivity=2.25)},
            param_transforms=[],
        )
        cons.append(dev.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=lo))
        objs.append(dev)
    wcs = [fdtdx.WaveCharacter(wavelength=w) for w in _WL]
    mon = PhasorDetector(
        name="mon", partial_grid_shape=(1, 1, 1), wave_characters=wcs, components=("Ez",), exact_interpolation=False
    )
    cons.append(mon.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(10, 6, 6)))
    objs.append(mon)
    if stock_detector:
        # PhasorDetector's stock defaults: six components, continuous, exact
        # interpolation, complex64 -- every one of them wrong for the kernel.
        des = PhasorDetector(name="des", partial_grid_shape=(4, 4, 4), wave_characters=wcs[:1])
        cons.append(des.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(4, 4, 4)))
        objs.append(des)
    objects, arrays, _, config, _ = fdtdx.place_objects(object_list=objs, config=config, constraints=cons, key=_KEY)
    return objects, arrays, config


def _internal(objects, arrays, config, design=None, keep=("mon",)):
    mon = next(d for d in objects.detectors if d.name == "mon")
    return internal_scene(
        objects,
        arrays,
        config,
        _KEY,
        keep_detectors=keep,
        design=design,
        wave_characters=mon.wave_characters,
    )


class TestDesignRegions:
    def test_default_is_every_device(self):
        objects, _, _ = _scene(devices=(("a", (2, 2, 2), (3, 3, 3)), ("b", (7, 7, 7), (3, 3, 3))))
        assert [r.name for r in design_regions(objects, None)] == ["a", "b"]

    def test_names_are_resolved_and_deduplicated(self):
        objects, _, _ = _scene()
        assert [r.name for r in design_regions(objects, ("design", "des", "design"))] == ["design", "des"]

    def test_no_device_and_no_name_raises(self):
        objects, _, _ = _scene(devices=())
        with pytest.raises(ValueError, match="no Device"):
            design_regions(objects, None)

    def test_unknown_name_raises(self):
        objects, _, _ = _scene()
        with pytest.raises(ValueError, match="No object named"):
            design_regions(objects, "nope")


class TestInternalScene:
    def test_design_detector_is_built_in_the_kernel_configuration(self):
        objects, arrays, config = _scene()
        io, ia, names = _internal(objects, arrays, config)
        assert names == (f"{DESIGN_DETECTOR_PREFIX}design",)
        det = next(d for d in io.detectors if d.name == names[0])
        assert type(det) is PhasorDetector
        for key, value in DESIGN_DETECTOR_SETTINGS.items():
            if key != "switch":
                assert getattr(det, key) == value, key
        assert det._num_time_steps_on == config.time_steps_total
        assert det._dft_stride == 1 and float(det._static_scale()) == 1.0
        assert det.dtype == jnp.complex128, "float64 run must record complex128 design phasors"
        device = next(d for d in objects.devices if d.name == "design")
        assert det.grid_slice_tuple == device.grid_slice_tuple
        # the objective's frequencies, not whatever a user's detector had
        assert len(det._angular_frequencies) == len(_WL)
        assert ia.detector_states[names[0]]["phasor"].shape == (1, len(_WL), 3, 4, 4, 4)

    def test_float32_run_records_complex64(self):
        objects, arrays, config = _scene(dtype=jnp.float32)
        io, _, names = _internal(objects, arrays, config)
        assert next(d for d in io.detectors if d.name == names[0]).dtype == jnp.complex64

    def test_unneeded_detectors_are_dropped_and_the_caller_is_untouched(self):
        objects, arrays, config = _scene()
        before = arrays.detector_states["des"]["phasor"].shape
        io, ia, names = _internal(objects, arrays, config)
        assert {d.name for d in io.detectors} == {"mon", *names}
        assert set(ia.detector_states) == {"mon", *names}
        assert io.volume is objects.volume
        # caller's containers are unchanged, so their own run_fdtd still records "des"
        assert "des" in {d.name for d in objects.detectors}
        assert arrays.detector_states["des"]["phasor"].shape == before == (1, 1, 6, 4, 4, 4)

    def test_adjoint_scene_keeps_no_objective(self):
        objects, arrays, config = _scene()
        io, ia, names = _internal(objects, arrays, config, keep=())
        assert {d.name for d in io.detectors} == set(names)
        assert set(ia.detector_states) == set(names)

    def test_one_detector_per_device(self):
        objects, arrays, config = _scene(devices=(("a", (2, 2, 2), (3, 3, 3)), ("b", (7, 7, 7), (3, 3, 3))))
        io, _, names = _internal(objects, arrays, config)
        assert names == (f"{DESIGN_DETECTOR_PREFIX}a", f"{DESIGN_DETECTOR_PREFIX}b")
        for name, dev in zip(names, objects.devices):
            assert next(d for d in io.detectors if d.name == name).grid_slice_tuple == dev.grid_slice_tuple

    def test_a_named_detector_region_uses_only_its_cells(self):
        objects, arrays, config = _scene()
        io, _, names = _internal(objects, arrays, config, design="des")
        assert names == (f"{DESIGN_DETECTOR_PREFIX}des",)
        des = next(d for d in objects.detectors if d.name == "des")
        assert next(d for d in io.detectors if d.name == names[0]).grid_slice_tuple == des.grid_slice_tuple


class TestRefusals:
    """Configurations measured to be silently wrong, refused at setup (no solve runs)."""

    def test_restricted_objective_switch_is_refused(self):
        """Measured before the guard: rel 1.8e-01 (recording from 20 fs) and 8.2e+02 (40 fs)."""
        from fdtdx.adjoint import reciprocity_phasor_fn

        objects, arrays, config = _scene()
        mon = next(d for d in objects.detectors if d.name == "mon")
        late = mon.aset("switch", fdtdx.OnOffSwitch(start_time=5e-15))
        late = late.aset("_grid_slice_tuple", ((-1, -1), (-1, -1), (-1, -1)))
        late = late.place_on_grid(mon.grid_slice_tuple, config, _KEY)
        assert late._num_time_steps_on < config.time_steps_total
        objects = objects.aset("object_list", [late if o is mon else o for o in objects.object_list])
        with pytest.raises(NotImplementedError, match="time steps"):
            reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")

    def test_dispersive_device_material_is_refused(self):
        """Measured before the guard: forward FoM off by 15%, gradient rel 0.82."""
        from fdtdx.adjoint import reciprocity_param_fn

        objects, arrays, config = _scene()
        device = objects.devices[0]
        pole = fdtdx.LorentzPole(resonance_frequency=4e15, damping=2e14, delta_epsilon=0.8)
        disp = fdtdx.Material(permittivity=2.0, dispersion=fdtdx.DispersionModel(poles=(pole,)))
        dispersive = device.aset("materials", {"air": device.materials["air"], "disp": disp})
        objects = objects.aset("object_list", [dispersive if o is device else o for o in objects.object_list])
        with pytest.raises(NotImplementedError, match="dispersive"):
            reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon")


class TestStoredComponentOrder:
    def test_phasor_detector_stacks_components_canonically(self):
        """The adjoint current follows the STORED order; pin that it is canonical.

        ``components=("Hz", "Ex")`` stores Ex at index 0. If PhasorDetector ever
        starts honouring the declared order, ``canonical_components`` must go,
        or every out-of-order monitor drives the wrong adjoint component.
        """
        config = SimulationConfig(time=5e-15, grid=UniformGrid(spacing=50e-9), backend="cpu", dtype=jnp.float64)
        det = PhasorDetector(
            wave_characters=[fdtdx.WaveCharacter(wavelength=600e-9)],
            components=("Hz", "Ex"),
            scaling_mode="pulse",
            exact_interpolation=False,
            dtype=jnp.complex128,
        ).place_on_grid(((0, 1), (0, 1), (0, 1)), config, _KEY)
        E = jnp.asarray([1.0, 2.0, 3.0]).reshape(3, 1, 1, 1)
        H = jnp.asarray([4.0, 5.0, 6.0]).reshape(3, 1, 1, 1)
        state = det.update(jnp.asarray(0), E, H, det.init_state(), jnp.ones((1, 1, 1, 1)), 1.0)
        stored = jnp.real(state["phasor"][0, 0, :, 0, 0, 0])
        assert tuple(float(x) for x in stored) == (1.0, 6.0)  # Ex, then Hz
        assert canonical_components(det) == ("Ex", "Hz")

    def test_static_scale_semantics(self):
        """The kernel divides _static_scale() back out: pin what it means."""
        config = SimulationConfig(time=5e-15, grid=UniformGrid(spacing=50e-9), backend="cpu", dtype=jnp.float64)
        kw = dict(wave_characters=[fdtdx.WaveCharacter(wavelength=600e-9)], exact_interpolation=False)
        cont = PhasorDetector(scaling_mode="continuous", **kw).place_on_grid(((0, 1),) * 3, config, _KEY)
        pulse = PhasorDetector(scaling_mode="pulse", **kw).place_on_grid(((0, 1),) * 3, config, _KEY)
        assert cont._window_sum == config.time_steps_total
        assert float(cont._static_scale()) == pytest.approx(2.0 / config.time_steps_total, rel=1e-12)
        assert pulse._static_scale() == pulse._dft_stride == 1
