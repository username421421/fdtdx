from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx


def test_ssp_intricate_grad_nan_bug():
    path = Path(__file__).parent.parent.parent / "data" / "example_device_params.npy"
    arr = jnp.load(path)

    config = fdtdx.SimulationConfig(
        time=200e-15,
        grid=fdtdx.UniformGrid(spacing=20e-9),
        dtype=jnp.float32,
        courant_factor=0.99,
    )
    material_config = {
        "sio2": fdtdx.Material(permittivity=3.9),
        "si": fdtdx.Material(permittivity=12.25),
    }
    height = 220e-9
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(7e-6, 6e-6, height),
        material=material_config["sio2"],
    )
    device = fdtdx.Device(
        name="Device",
        partial_real_shape=(7e-6, 6e-6, height),
        materials=material_config,
        param_transforms=[
            fdtdx.GaussianSmoothing2D(std_discrete=3),
            fdtdx.SubpixelSmoothedProjection(),
        ],
        partial_voxel_real_shape=(config.uniform_spacing(), config.uniform_spacing(), height),
    )
    key = jax.random.PRNGKey(42)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, device],
        config=config,
        constraints=[device.place_at_center(volume)],
        key=key,
    )
    params["Device"] = arr
    arrays, new_objects, _ = fdtdx.apply_params(arrays, objects, params, key, beta=5.0)

    def fn(p):
        cur_material_indices = new_objects["Device"](p[device.name], expand_to_sim_grid=False, beta=5.0)  # type: ignore
        return jnp.sum(cur_material_indices)

    value, grad = jax.value_and_grad(fn)(params)
    assert not jnp.isnan(value) and not jnp.isinf(value)
    assert not jnp.any(jnp.isnan(grad["Device"]))


def _block_scene(block, fill=1.0):
    """16^3 PML scene, a dipole beside ``block`` at cells 6..10; Device parameters set to ``fill``."""
    config = fdtdx.SimulationConfig(time=30e-15, grid=fdtdx.UniformGrid(spacing=50e-9), backend="cpu")
    volume = fdtdx.SimulationVolume(partial_grid_shape=(16, 16, 16))
    boundaries, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=4), volume
    )
    source = fdtdx.PointDipoleSource(
        name="src", partial_grid_shape=(1, 1, 1), wave_character=fdtdx.WaveCharacter(wavelength=600e-9), polarization=2
    )
    constraints += [
        block.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(6, 6, 6)),
        source.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(4, 8, 8)),
    ]
    key = jax.random.PRNGKey(0)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=[volume, *boundaries.values(), block, source], config=config, constraints=constraints, key=key
    )
    params = {name: jnp.full_like(p, fill) for name, p in params.items()}
    arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
    _, out = fdtdx.run_fdtd(arrays, objects, config, key, show_progress=False)
    return arrays, out.fields.E


def test_lossy_device_matches_static_block():
    """``apply_params`` writes a Device material's conductivity with the permittivity's weights."""
    lossy = fdtdx.Material(permittivity=2.25, electric_conductivity=1e5)

    def device(materials, transforms=()):
        return fdtdx.Device(
            name="block",
            partial_grid_shape=(4, 4, 4),
            partial_voxel_grid_shape=(1, 1, 1),
            materials=materials,
            param_transforms=list(transforms),
        )

    static_arrays, static_E = _block_scene(
        fdtdx.UniformMaterialObject(name="block", partial_grid_shape=(4, 4, 4), material=lossy)
    )
    lossy_arrays, lossy_E = _block_scene(device({"air": fdtdx.Material(), "lossy": lossy}))
    _, lossless_E = _block_scene(device({"air": fdtdx.Material(), "si": fdtdx.Material(permittivity=2.25)}))
    half_arrays, _ = _block_scene(device({"air": fdtdx.Material(), "lossy": lossy}), fill=0.5)
    snapped_arrays, _ = _block_scene(
        device({"air": fdtdx.Material(), "lossy": lossy}, [fdtdx.ClosestIndex()]), fill=0.8
    )

    block = (slice(None), slice(6, 10), slice(6, 10), slice(6, 10))
    assert static_arrays.electric_conductivity is not None and lossy_arrays.electric_conductivity is not None
    assert float(static_arrays.electric_conductivity[block].min()) > 0.0
    np.testing.assert_array_equal(lossy_arrays.electric_conductivity, static_arrays.electric_conductivity)
    np.testing.assert_array_equal(snapped_arrays.electric_conductivity, static_arrays.electric_conductivity)
    np.testing.assert_allclose(
        half_arrays.electric_conductivity[block], 0.5 * static_arrays.electric_conductivity[block], rtol=1e-6
    )
    np.testing.assert_allclose(lossy_E, static_E, rtol=1e-5, atol=1e-6 * float(jnp.abs(static_E).max()))
    # and the loss is real (it used to be dropped: the fields equalled the lossless ones)
    assert float(jnp.sum(lossy_E**2)) < 0.9 * float(jnp.sum(lossless_E**2))
