"""reciprocity_param_fn takes the objects straight from place_objects.

``place_objects`` leaves every object whose projection overlaps a Device's
unapplied, so a plane source spanning the design's footprint, or a mode port
sharing a periodic span with it, had no incident field and crashed the solve
with ``TypeError: 'Null' object is not subscriptable``. ``apply_objects_once``
applies them at setup exactly as ``apply_params`` would; these tests pin that
the result is identical, leaf for leaf, to what ``apply_params`` returns. No
FDTD solves here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.adjoint import reciprocity_param_fn, reciprocity_phasor_fn
from fdtdx.adjoint.design import apply_objects_once
from fdtdx.config import SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.core.null import Null
from fdtdx.fdtd.initialization import apply_params

_KEY = jax.random.PRNGKey(5)
_WL = 1.2e-6


@pytest.fixture(autouse=True)
def _enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _wc(w=_WL):
    return fdtdx.WaveCharacter(wavelength=w)


def _pulse():
    return fdtdx.GaussianPulseProfile(center_wave=_wc(), spectral_width=fdtdx.WaveCharacter(frequency=0.2 * c0 / _WL))


def _scene(kind, source_inside=False):
    """Device plus a source (and, for "mode", a mode port) whose projection overlaps it."""
    config = SimulationConfig(time=10e-15, grid=UniformGrid(spacing=50e-9), backend="cpu", dtype=jnp.float64)
    air, core = fdtdx.Material(permittivity=1.0), fdtdx.Material(permittivity=4.0)
    if kind == "mode":
        shape = (40, 2, 28)
        boundary = fdtdx.BoundaryConfig(
            boundary_type_miny="periodic",
            boundary_type_maxy="periodic",
            thickness_grid_minx=6,
            thickness_grid_maxx=6,
            thickness_grid_miny=1,
            thickness_grid_maxy=1,
            thickness_grid_minz=6,
            thickness_grid_maxz=6,
        )
    else:
        shape = (16, 16, 20)
        boundary = fdtdx.BoundaryConfig.from_uniform_bound(thickness=4)
    vol = fdtdx.SimulationVolume(partial_grid_shape=shape, material=air)
    objs, cons = [vol], []
    bd, cl = fdtdx.boundary_objects_from_config(boundary, vol)
    objs.extend(bd.values())
    cons.extend(cl)

    def at(obj, lo):
        objs.append(obj)
        cons.append(obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=lo))

    if kind == "mode":
        at(fdtdx.UniformMaterialObject(name="guide", partial_grid_shape=(40, 2, 6), material=core), (0, 0, 11))
        at(
            fdtdx.Device(
                name="design",
                partial_grid_shape=(8, 2, 8),
                partial_voxel_grid_shape=(1, 2, 1),
                materials={"background": air, "core": core},
                param_transforms=[],
                placement_order=10,
            ),
            (16, 0, 10),
        )
        mode_kw = dict(mode_index=0, filter_pol="te", partial_grid_shape=(1, 2, 14))
        at(
            fdtdx.ModePlaneSource(
                name="src", direction="+", wave_character=_wc(), temporal_profile=_pulse(), **mode_kw
            ),
            (8, 0, 7),
        )
        at(fdtdx.ModeOverlapDetector(name="port", direction="+", wave_characters=(_wc(),), **mode_kw), (30, 0, 7))
    else:
        at(
            fdtdx.Device(
                name="design",
                partial_grid_shape=(6, 6, 3),
                partial_voxel_grid_shape=(1, 1, 1),
                materials={"air": air, "si": fdtdx.Material(permittivity=2.25)},
                param_transforms=[],
            ),
            (5, 5, 8),
        )
        at(
            fdtdx.UniformPlaneSource(
                name="src",
                partial_grid_shape=(16, 16, 1),
                direction="-",
                fixed_E_polarization_vector=(1, 0, 0),
                wave_character=_wc(),
                temporal_profile=_pulse(),
                normalize_by_energy=True,
            ),
            (0, 0, 9 if source_inside else 14),
        )
        at(fdtdx.PhasorDetector(name="mon", partial_grid_shape=(4, 4, 1), wave_characters=(_wc(),)), (6, 6, 5))
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objs, config=config, constraints=cons, key=_KEY
    )
    return objects, arrays, params, config


def _assert_same_leaves(a, b, name):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb), name
    for x, y in zip(la, lb):
        if isinstance(x, jax.Array | np.ndarray):
            assert np.array_equal(np.asarray(x), np.asarray(y)), f"{name}: applied state differs"
        else:
            assert x == y, name


@pytest.mark.parametrize("kind", ["plane", "mode"])
def test_placed_objects_come_back_unapplied(kind):
    """The hazard itself: place_objects skips every object sharing a Device's projection."""
    objects, *_ = _scene(kind)
    assert isinstance(objects["src"]._E, Null)
    if kind == "mode":
        assert isinstance(objects["port"]._mode_E, Null)


@pytest.mark.parametrize("kind", ["plane", "mode"])
def test_identical_to_apply_params(kind):
    objects, arrays, params, _ = _scene(kind)
    once = apply_objects_once(arrays, objects, _KEY)
    _, ref, _ = apply_params(arrays, objects, params, _KEY)
    names = ("src", "port") if kind == "mode" else ("src",)
    for name in names:
        assert not isinstance(once[name]._E if name == "src" else once[name]._mode_E, Null)
        _assert_same_leaves(once[name], ref[name], name)
    # the caller's container is untouched
    assert isinstance(objects["src"]._E, Null)


def test_param_fn_exposes_the_applied_scene():
    objects, arrays, params, config = _scene("mode")
    param_fn = reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="port")
    _, ref, _ = apply_params(arrays, objects, params, _KEY)
    _assert_same_leaves(param_fn.objects["port"], ref["port"], "port")
    _assert_same_leaves(param_fn.objects["src"], ref["src"], "src")


def test_param_fn_runs_under_jit():
    """The returned callable is an object, not a function: it must still jit (10 fs, value only)."""
    objects, arrays, params, config = _scene("plane")
    param_fn = reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon")
    eager = param_fn(params)
    jitted = jax.jit(param_fn)(params)
    assert eager.shape == jitted.shape
    assert float(jnp.linalg.norm(eager - jitted)) <= 1e-5 * float(jnp.linalg.norm(eager))


def test_phasor_fn_names_an_unapplied_source():
    """At the inv_permittivities level the objects are the caller's; say so instead of crashing."""
    objects, arrays, _, config = _scene("plane")
    with pytest.raises(ValueError, match="never applied"):
        reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")


def test_source_overlapping_a_device_is_refused():
    """Its incident field would depend on the design, so applying it once would freeze it."""
    objects, arrays, _, _ = _scene("plane", source_inside=True)
    with pytest.raises(NotImplementedError, match="overlap a Device"):
        apply_objects_once(arrays, objects, _KEY)
