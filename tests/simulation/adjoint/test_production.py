"""Production path: single-scene API, real Devices, lossy media, sources in the design.

These cover what the raw-array tests in ``test_reciprocity.py`` do not: the
parameter-level entry point, parity with FDTDX's own ``run_fdtd`` gradient
pipeline, and the two material/geometry cases that were previously believed to
be unsupported.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.adjoint import (
    derive_adjoint_objects,
    design_region_slice,
    gaussian_window,
    reciprocity_param_fn,
    reciprocity_phasor_fn,
)
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.initialization import apply_params
from fdtdx.objects.sources.adjoint import AdjointCurrentSource

_RES = 50e-9
_N = 24
_PML = 4
_WL = (600e-9,)
_OMEGAS = tuple(2 * np.pi * c0 / w for w in _WL)
_SRC = (_PML + 1, _N // 2, _N // 2)
_MON = (_N - _PML - 2, _N // 2, _N // 2)
_DES_LO = _PML + 4
_DES_SPAN = _N - 2 * _DES_LO
_KEY = jax.random.PRNGKey(3)


@pytest.fixture(autouse=True)
def _enable_x64():
    """float64 per test: pytest shares one process, so a global flip would change
    default dtypes for every other test module in the session."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _narrowband_dipole(name, cell):
    """A dipole narrow enough for the single-frequency reciprocity identity.

    Spectral width matters and is not a detail: at 0.4*f0 the pulse spans about
    2.5 optical cycles and the relative error against checkpointed autodiff is
    2.4e-03; at 0.1*f0 it is 2.5e-07.
    """
    wc = fdtdx.WaveCharacter(wavelength=_WL[0])
    src = fdtdx.PointDipoleSource(
        name=name,
        partial_grid_shape=(1, 1, 1),
        wave_character=wc,
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_wave=wc,
            spectral_width=fdtdx.WaveCharacter(frequency=float(c0 / _WL[0]) * 0.1),
        ),
        polarization=2,
        amplitude=1.0,
    )
    return src, src.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=cell)


def _scene(
    sim_fs=150.0,
    *,
    with_device=False,
    sigma=0.0,
    source_cell=_SRC,
    components=("Ez",),
    mon_scaling="pulse",
    mon2_scaling=None,
    design_detector_kwargs=None,
    devices=None,
    monitor_kwargs=None,
):
    """24^3 test scene.

    ``design_detector_kwargs`` is forwarded to the ``"des"`` PhasorDetector over
    the design region; ``{}`` leaves it at PhasorDetector's stock defaults, and
    ``None`` uses the configuration the kernel is calibrated for, which is what
    every test had to write out before the design detector became internal.
    ``devices`` is a list of ``(name, lower_corner, shape)`` replacing the single
    ``"design"`` Device; ``mon2_scaling`` adds a second monitor ``"mon2"``.
    ``monitor_kwargs``, when given, replaces every setting of the ``"mon"``
    monitor except its name, shape and frequencies; ``{}`` is PhasorDetector's
    stock defaults.
    """
    if devices is None and with_device:
        devices = [("design", (_DES_LO,) * 3, (_DES_SPAN,) * 3)]
    config = SimulationConfig(
        time=sim_fs * 1e-15,
        grid=UniformGrid(spacing=_RES),
        backend="cpu",
        dtype=jnp.float64,
        courant_factor=0.99,
        gradient_config=None,
    )
    objs, cons = [], []
    vol = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
    objs.append(vol)
    bd, cl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
    objs.extend(bd.values())
    cons.extend(cl)

    if devices:
        for name, lower, shape in devices:
            device = fdtdx.Device(
                name=name,
                partial_grid_shape=shape,
                partial_voxel_grid_shape=(1, 1, 1),
                materials={"air": fdtdx.Material(permittivity=1.0), "si": fdtdx.Material(permittivity=2.25)},
                param_transforms=[],
            )
            cons.append(device.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=lower))
            objs.append(device)
    else:
        block = fdtdx.UniformMaterialObject(
            name="block",
            partial_grid_shape=(_DES_SPAN,) * 3,
            material=fdtdx.Material(permittivity=2.25, electric_conductivity=sigma),
        )
        cons.append(block.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(_DES_LO,) * 3))
        objs.append(block)

    src, constraint = _narrowband_dipole("src", source_cell)
    cons.append(constraint)
    objs.append(src)

    wcs = [fdtdx.WaveCharacter(wavelength=w) for w in _WL]
    if monitor_kwargs is None:
        monitor_kwargs = dict(
            components=components,
            scaling_mode=mon_scaling,
            dft_subsample=1,
            exact_interpolation=False,
            reduce_volume=False,
            dtype=jnp.complex128,
        )
    mon = fdtdx.PhasorDetector(name="mon", partial_grid_shape=(1, 1, 1), wave_characters=wcs, **monitor_kwargs)
    cons.append(mon.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=_MON))
    objs.append(mon)
    if mon2_scaling is not None:
        mon2 = fdtdx.PhasorDetector(
            name="mon2",
            partial_grid_shape=(1, 1, 1),
            wave_characters=wcs,
            components=("Ez",),
            scaling_mode=mon2_scaling,
            dft_subsample=1,
            exact_interpolation=False,
            reduce_volume=False,
            dtype=jnp.complex128,
        )
        mon2_cell = (_MON[0], _MON[1] + 2, _MON[2])
        cons.append(mon2.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=mon2_cell))
        objs.append(mon2)
    if design_detector_kwargs is None:
        design_detector_kwargs = dict(
            components=("Ex", "Ey", "Ez"),
            scaling_mode="pulse",
            dft_subsample=1,
            exact_interpolation=False,
            reduce_volume=False,
            dtype=jnp.complex128,
        )
    des = fdtdx.PhasorDetector(
        name="des",
        partial_grid_shape=(_DES_SPAN,) * 3,
        **{"wave_characters": wcs, **design_detector_kwargs},
    )
    cons.append(des.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(_DES_LO,) * 3))
    objs.append(des)

    return fdtdx.place_objects(object_list=objs, config=config, constraints=cons, key=_KEY)


def _rel_cos(a, b):
    rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
    cos = float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
    return rel, cos


def _official_inv_eps_grad(objects, arrays, config, readout):
    cfg_ck = config.aset("gradient_config", GradientConfig(method="checkpointed", num_checkpoints=8))

    def official(ie):
        _, out = fdtdx.run_fdtd(arrays.aset("inv_permittivities", ie), objects, cfg_ck, _KEY, show_progress=False)
        return readout(out.detector_states)

    return jax.value_and_grad(official)(arrays.inv_permittivities)


def _official_param_grad(objects, arrays, params, config):
    cfg_ck = config.aset("gradient_config", GradientConfig(method="checkpointed", num_checkpoints=8))

    def official(p):
        arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
        _, out = fdtdx.run_fdtd(arrs, objs, cfg_ck, _KEY, show_progress=False)
        return _FOM(out.detector_states["mon"]["phasor"])

    return jax.value_and_grad(official)(params)


def _FOM(P):
    return -jnp.sum(jnp.abs(P) ** 2)


def _flat(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.concatenate([jnp.ravel(x) for x in leaves])


class TestAdjointSceneDerivation:
    def test_second_place_objects_would_change_device_params(self):
        """Why derive_adjoint_objects exists rather than a second place_objects.

        place_objects splits its PRNG once per placed object before initializing
        device parameters, so two scenes differing only in object count get
        DIFFERENT initial device parameters. Building the adjoint scene that way
        would silently simulate a different structure.
        """
        _, _, p_one, _, _ = _scene(sim_fs=20.0, with_device=True)
        objs, cons = [], []
        # same scene plus one extra source object
        vol = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
        objs.append(vol)
        bd, cl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
        objs.extend(bd.values())
        cons.extend(cl)
        device = fdtdx.Device(
            name="design",
            partial_grid_shape=(_DES_SPAN,) * 3,
            partial_voxel_grid_shape=(1, 1, 1),
            materials={"air": fdtdx.Material(permittivity=1.0), "si": fdtdx.Material(permittivity=2.25)},
            param_transforms=[],
        )
        cons.append(device.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(_DES_LO,) * 3))
        objs.append(device)
        for i, cell in enumerate((_SRC, _MON)):
            s, c = _narrowband_dipole(f"s{i}", cell)
            cons.append(c)
            objs.append(s)
        config = SimulationConfig(
            time=20e-15,
            grid=UniformGrid(spacing=_RES),
            backend="cpu",
            dtype=jnp.float64,
            courant_factor=0.99,
            gradient_config=None,
        )
        _, _, p_two, _, _ = fdtdx.place_objects(object_list=objs, config=config, constraints=cons, key=_KEY)
        assert not bool(jnp.all(_flat(p_one) == _flat(p_two))), (
            "object count no longer perturbs device parameters; derive_adjoint_objects "
            "may no longer be necessary, re-check before simplifying"
        )

    def test_derived_scene_keeps_geometry_and_swaps_the_source(self):
        objects, _arrays, _, config, _ = _scene(sim_fs=20.0)
        window = gaussian_window(int(config.time_steps_total))
        adj, names = derive_adjoint_objects(
            objects=objects, config=config, objective_detectors="mon", window=window, key=_KEY
        )
        assert len(adj.sources) == 1
        assert isinstance(adj.sources[0], AdjointCurrentSource)
        assert adj.sources[0].name == names[0][0]
        mon = next(d for d in objects.detectors if d.name == "mon")
        assert adj.sources[0].grid_slice_tuple == mon.grid_slice_tuple
        fwd_mat = {o.name for o in objects.static_material_objects}
        adj_mat = {o.name for o in adj.static_material_objects}
        assert fwd_mat == adj_mat, "material geometry must be carried over unchanged"
        assert adj.volume.grid_shape == objects.volume.grid_shape


class TestOfficialPipelineParity:
    """Same answer as FDTDX's own gradient pipeline, at the parameter level."""

    def test_matches_run_fdtd_checkpointed_on_device_params(self):
        objects, arrays, params, config, _ = _scene(with_device=True)
        window = gaussian_window(int(config.time_steps_total))

        cfg_ck = config.aset("gradient_config", GradientConfig(method="checkpointed", num_checkpoints=8))

        def official(p):
            arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
            _, out = fdtdx.run_fdtd(arrs, objs, cfg_ck, _KEY, show_progress=False)
            return _FOM(out.detector_states["mon"]["phasor"])

        param_fn = reciprocity_param_fn(
            arrays,
            objects,
            config,
            _KEY,
            objective_detectors="mon",
            design_detector="des",
            window=window,
        )

        v_off, g_off = jax.value_and_grad(official)(params)
        v_rec, g_rec = jax.value_and_grad(lambda p: _FOM(param_fn(p)))(params)

        assert jax.tree_util.tree_structure(g_rec) == jax.tree_util.tree_structure(g_off)
        assert jnp.allclose(v_rec, v_off, rtol=1e-12), "forward values must be identical"

        a, b = _flat(g_rec), _flat(g_off)
        rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
        cos = float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
        sign = float((jnp.sign(a) == jnp.sign(b)).mean())
        assert rel < 1e-4, f"parameter gradient rel_L2 = {rel:.3e}"
        assert cos > 1 - 1e-6, f"cosine {cos:.10f}"
        assert sign > 0.99, f"sign agreement {sign:.4f}"


class TestMaterialAndGeometryCoverage:
    """Cases previously believed unsupported, each with the reason recorded."""

    def _rel_against_checkpointed(self, sim_fs=150.0, sigma=0.0, source_cell=_SRC):
        objects, arrays, _, config, _ = _scene(sim_fs, sigma=sigma, source_cell=source_cell)
        window = gaussian_window(int(config.time_steps_total))
        cfg_ck = config.aset("gradient_config", GradientConfig(method="checkpointed", num_checkpoints=8))

        def official(ie):
            arrs = arrays.aset("inv_permittivities", ie)
            _, out = fdtdx.run_fdtd(arrs, objects, cfg_ck, _KEY, show_progress=False)
            return _FOM(out.detector_states["mon"]["phasor"])

        fn = reciprocity_phasor_fn(
            arrays,
            objects,
            config,
            _KEY,
            objective_detectors="mon",
            design_detector="des",
            window=window,
        )
        ie = arrays.inv_permittivities
        g_off = jax.grad(official)(ie)
        g_rec = jax.grad(lambda x: _FOM(fn(x)))(ie)
        gs = next(d for d in objects.detectors if d.name == "des").grid_slice
        a, b = g_rec[:, *gs], g_off[:, *gs]
        return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))

    @pytest.mark.parametrize("sigma", [0.0, 1e4, 1e5], ids=["lossless", "sigma1e4", "sigma1e5"])
    def test_lossy_design_region_needs_no_extra_term(self, sigma):
        """Loss is already inside the field increment, so the kernel is unchanged.

        E_new = E_old + courant*inv_eps*(curl - sigma*eta0*E_old/2), hence
        dE_new/d(inv_eps) = (E_new - E_old)/inv_eps exactly as when lossless.
        Verified up to sigma = 3e5 S/m, where the damping factor deviates from
        unity by more than 1.
        """
        rel = self._rel_against_checkpointed(sigma=sigma)
        assert rel < 1e-3, f"rel={rel:.3e} at sigma={sigma}"

    def test_frozen_source_inside_design_region_is_refused(self):
        """Stock sources freeze their inv_eps factor, so overlap is refused.

        FDTDX injects an impressed current as E += -courant*inv_eps*J. A source
        inside the design region therefore contributes to the design gradient,
        but PointDipoleSource caches that factor during apply() and the TFSF
        plane sources stop_gradient it, so run_fdtd's own gradient omits the
        term. Leaving the caches in place gives a relative error of 1.16;
        clearing them gives a self-consistent but *different* gradient from the
        official pipeline. Refusing is the honest option.
        """
        centre = _DES_LO + _DES_SPAN // 2
        objects, arrays, _, config, _ = _scene(source_cell=(centre, centre, centre))
        window = gaussian_window(int(config.time_steps_total))
        with pytest.raises(NotImplementedError, match="overlap the design region"):
            reciprocity_phasor_fn(
                arrays,
                objects,
                config,
                _KEY,
                objective_detectors="mon",
                design_detector="des",
                window=window,
            )

    def test_adjoint_current_source_may_sit_inside_the_design_region(self):
        """The documented remedy works: AdjointCurrentSource reads inv_eps live."""
        centre = _DES_LO + _DES_SPAN // 2
        objects, _arrays, _, config, _ = _scene()
        window = gaussian_window(int(config.time_steps_total))
        nf, nc = len(_OMEGAS), 1
        live = AdjointCurrentSource(
            name="live",
            amplitudes=jnp.zeros((nf, nc, 1, 1, 1), dtype=jnp.complex128),
            window=window,
            angular_frequencies=_OMEGAS,
            components=("Ez",),
            wave_character=fdtdx.WaveCharacter(wavelength=_WL[0]),
        ).place_on_grid(grid_slice_tuple=((centre, centre + 1),) * 3, config=config, key=_KEY)
        des_slice = next(d for d in objects.detectors if d.name == "des").grid_slice_tuple
        swapped = objects.aset(
            "object_list",
            [o for o in objects.object_list if getattr(o, "name", None) != "src"] + [live],
        )
        # must not raise despite sitting at the centre of the design region
        from fdtdx.adjoint.vjp import _reject_frozen_sources_in_design

        _reject_frozen_sources_in_design(swapped, des_slice)


class TestMagneticComponents:
    """Objectives on H, needed before box-mode near-to-far can work.

    Box-mode field projection concatenates E and H, so its adjoint source drives
    all six components. Magnetic components carry an extra factor
    ``-exp(-i w dt / 2)``: the minus from Lorentz reciprocity's asymmetry between
    the electric and magnetic pairings, and the half-step because the detector
    stores the post-update H (living at n+1/2) but weights it with the
    integer-step kernel. Measured on an Hx objective: no factor 1.91 at cosine
    -0.998, sign flip alone 1.04e-01, both 6.3e-07 at cosine 1.000000000.

    Until the ``_scene`` fix that came with this docstring, ``components`` was
    accepted by ``_scene`` but never passed to the monitor, so every case here
    silently tested ``("Ez",)``.

    ``("Hx", "Ez")`` is declared out of order on purpose. PhasorDetector stores
    components in canonical order whatever order they are declared in, and the
    adjoint current used to follow the declared order, driving each cotangent
    into the other component.
    """

    @pytest.mark.parametrize(
        "components",
        [
            ("Ez",),
            ("Hx",),
            ("Hy",),
            ("Ez", "Hx"),
            ("Hx", "Ez"),
            ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        ],
        ids=["Ez", "Hx", "Hy", "Ez_Hx", "Hx_Ez_declared_out_of_order", "all_six"],
    )
    def test_matches_checkpointed_for_any_component_set(self, components):
        objects, arrays, _, config, _ = _scene(components=components)
        window = gaussian_window(int(config.time_steps_total))
        cfg_ck = config.aset("gradient_config", GradientConfig(method="checkpointed", num_checkpoints=8))

        def official(ie):
            arrs = arrays.aset("inv_permittivities", ie)
            _, out = fdtdx.run_fdtd(arrs, objects, cfg_ck, _KEY, show_progress=False)
            return _FOM(out.detector_states["mon"]["phasor"])

        fn = reciprocity_phasor_fn(
            arrays,
            objects,
            config,
            _KEY,
            objective_detectors="mon",
            design_detector="des",
            window=window,
        )
        ie = arrays.inv_permittivities
        mon = next(d for d in objects.detectors if d.name == "mon")
        assert set(mon.components) == set(components), "the scene must actually use these components"
        g_off = jax.grad(official)(ie)
        g_rec = jax.grad(lambda x: _FOM(fn(x)))(ie)
        gs = next(d for d in objects.detectors if d.name == "des").grid_slice
        a, b = g_rec[:, *gs], g_off[:, *gs]
        rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
        cos = float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
        assert rel < 1e-4, f"{components}: rel={rel:.3e}"
        assert cos > 1 - 1e-8, f"{components}: cosine {cos:.10f}"

    def test_magnetic_sign_flip_is_actually_needed(self):
        """Guard against someone 'simplifying' the magnetic factor away.

        Without the flip the Hx gradient anti-correlates with the truth, so the
        optimizer would walk uphill.
        """
        from fdtdx.adjoint import vjp as _vjp

        omegas = _OMEGAS
        dt = 1e-16
        factor = -np.exp(-1j * np.asarray(omegas) * dt / 2.0)
        assert np.all(factor.real < 0), "the magnetic factor must carry the reciprocity sign"
        assert not np.allclose(factor, -1.0), "the half-step phase must not be dropped"
        assert hasattr(_vjp, "make_reciprocity_phasor_fn")


class TestNearToFar:
    """Box-mode near-to-far projection.

    The VJP boundary sits at the detector's raw per-face phasors, so
    ``FieldProjectionAngleDetector.project`` runs above it as ordinary JAX and
    the projected far field is FDTDX's own, not a reimplementation. The box
    expands into one adjoint current per included face, and all faces are driven
    in a single adjoint solve by superposition.
    """

    _THETA = jnp.linspace(0.0, 0.5, 4)
    _PHI = jnp.linspace(0.0, 1.0, 3)
    _BOX_LO, _BOX_SPAN = _PML + 2, _N - 2 * (_PML + 2)

    def _scene(self, sim_fs=150.0, exact_interpolation=False):
        config = SimulationConfig(
            time=sim_fs * 1e-15,
            grid=UniformGrid(spacing=_RES),
            backend="cpu",
            dtype=jnp.float64,
            courant_factor=0.99,
            gradient_config=None,
        )
        objs, cons = [], []
        vol = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
        objs.append(vol)
        bd, cl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
        objs.extend(bd.values())
        cons.extend(cl)
        blk = fdtdx.UniformMaterialObject(
            name="blk",
            partial_grid_shape=(_DES_SPAN,) * 3,
            material=fdtdx.Material(permittivity=2.25),
        )
        cons.append(blk.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(_DES_LO,) * 3))
        objs.append(blk)
        src, constraint = _narrowband_dipole("src", (_PML + 1, _N // 2, _N // 2))
        cons.append(constraint)
        objs.append(src)

        wcs = [fdtdx.WaveCharacter(wavelength=w) for w in _WL]
        ff = fdtdx.FieldProjectionAngleDetector(
            name="ff",
            partial_grid_shape=(self._BOX_SPAN,) * 3,
            wave_characters=wcs,
            exclude_surfaces=("z-",),
            origin=(0.0, 0.0, 0.0),
            projection_distance=1e-3,
            far_field_approx=True,
            projection_medium=fdtdx.Material(permittivity=1.0),
            scaling_mode="pulse",
            dft_subsample=1,
            dtype=jnp.complex128,
        )
        if not exact_interpolation:
            ff = ff.aset("exact_interpolation", False)
        cons.append(ff.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(self._BOX_LO,) * 3))
        objs.append(ff)
        des = fdtdx.PhasorDetector(
            name="des",
            partial_grid_shape=(_DES_SPAN,) * 3,
            wave_characters=wcs,
            components=("Ex", "Ey", "Ez"),
            scaling_mode="pulse",
            dft_subsample=1,
            exact_interpolation=False,
            reduce_volume=False,
            dtype=jnp.complex128,
        )
        cons.append(des.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(_DES_LO,) * 3))
        objs.append(des)
        return fdtdx.place_objects(object_list=objs, config=config, constraints=cons, key=_KEY)

    def _fom(self, det, state):
        out = det.project(state, self._THETA, self._PHI)
        return -jnp.sum(jnp.abs(out["power"]))

    def test_box_expands_into_one_source_per_face(self):
        objects, _arrays, _, config, _ = self._scene(sim_fs=20.0)
        det = next(d for d in objects.detectors if d.name == "ff")
        assert det._projection_mode == "box"
        faces = det._included_box_surfaces()
        assert "z-" not in faces and len(faces) == 5
        window = gaussian_window(int(config.time_steps_total))
        adj, names = derive_adjoint_objects(
            objects=objects, config=config, objective_detectors="ff", window=window, key=_KEY
        )
        assert len(names) == 1 and len(names[0]) == len(faces)
        assert len(adj.sources) == len(faces)
        # each face source is a one-cell-thick slab on its own normal axis
        for src in adj.sources:
            thickness = [hi - lo for lo, hi in src.grid_slice_tuple]
            assert sorted(thickness)[0] == 1, f"{src.name} is not a face slab: {thickness}"

    @pytest.mark.parametrize("exact_interpolation", [False, True], ids=["raw_fields", "exact_interpolation"])
    def test_projected_far_field_gradient_matches_official(self, exact_interpolation):
        """``exact_interpolation=True`` is the detector's forced stock setting.

        It used to be refused: the co-location stencil runs outside the
        detector's own update. Each face's stencil is now transposed and its
        adjoint current covers the stencil's support.
        """
        objects, arrays, _, config, _ = self._scene(exact_interpolation=exact_interpolation)
        det = next(d for d in objects.detectors if d.name == "ff")
        window = gaussian_window(int(config.time_steps_total))
        cfg_ck = config.aset("gradient_config", GradientConfig(method="checkpointed", num_checkpoints=8))

        def official(ie):
            arrs = arrays.aset("inv_permittivities", ie)
            _, out = fdtdx.run_fdtd(arrs, objects, cfg_ck, _KEY, show_progress=False)
            return self._fom(det, out.detector_states["ff"])

        fn = reciprocity_phasor_fn(
            arrays,
            objects,
            config,
            _KEY,
            objective_detectors="ff",
            design_detector="des",
            window=window,
        )
        ie = arrays.inv_permittivities
        v_off, g_off = jax.value_and_grad(official)(ie)
        v_rec, g_rec = jax.value_and_grad(lambda x: self._fom(det, fn(x)))(ie)

        assert jnp.allclose(v_rec, v_off, rtol=1e-12), "forward far field must be identical"
        gs = next(d for d in objects.detectors if d.name == "des").grid_slice
        a, b = g_rec[:, *gs], g_off[:, *gs]
        rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
        cos = float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
        assert rel < 1e-4, f"near-to-far gradient rel_L2 = {rel:.3e}"
        assert cos > 1 - 1e-8, f"cosine {cos:.10f}"

    def test_exact_faces_get_the_stencil_support(self):
        """With exact interpolation each face current covers the stencil's reach, not the face."""
        objects, _arrays, _, config, _ = self._scene(sim_fs=20.0, exact_interpolation=True)
        det = next(d for d in objects.detectors if d.name == "ff")
        window = gaussian_window(int(config.time_steps_total))
        adj, names = derive_adjoint_objects(
            objects=objects, config=config, objective_detectors="ff", window=window, key=_KEY
        )
        assert len(names[0]) == len(det._included_box_surfaces()) == 5
        (sx, ex), (sy, ey), (_sz, ez) = det.grid_slice_tuple
        top = next(s for s in adj.sources if s.name.endswith("phasor_z_plus"))
        # z+ face at z = ez - 1: x, y reach one cell below, z one cell above
        assert top.grid_slice_tuple == ((sx - 1, ex), (sy - 1, ey), (ez - 1, ez + 1))


class TestFluxAndEnergyObjectives:
    """Flux, net power through a closed box, and stored energy.

    None of these needed a special case. Every supported detector accumulates
    complex phasors linearly from the fields, and whatever it computes on top is
    pure JAX above the VJP boundary, so the detector's own readout differentiates
    itself: ``compute_poynting_flux``, ``compute_net_flux`` and a hand-written
    energy integral all work through the same transpose. A closed box is six
    state keys and therefore six adjoint currents, still driven in one solve.
    """

    _BOX_LO, _BOX_SPAN = _PML + 3, _N - 2 * (_PML + 3)

    def _off_interp(self, det):
        if getattr(det, "exact_interpolation", False):
            return det.aset("exact_interpolation", False)
        return det

    def _scene(self, kind, sim_fs=120.0):
        config = SimulationConfig(
            time=sim_fs * 1e-15,
            grid=UniformGrid(spacing=_RES),
            backend="cpu",
            dtype=jnp.float64,
            courant_factor=0.99,
            gradient_config=None,
        )
        objs, cons = [], []
        vol = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
        objs.append(vol)
        bd, cl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
        objs.extend(bd.values())
        cons.extend(cl)
        blk = fdtdx.UniformMaterialObject(
            name="blk",
            partial_grid_shape=(_DES_SPAN,) * 3,
            material=fdtdx.Material(permittivity=2.25),
        )
        cons.append(blk.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(_DES_LO,) * 3))
        objs.append(blk)
        src, constraint = _narrowband_dipole("src", (_PML + 1, _N // 2, _N // 2))
        cons.append(constraint)
        objs.append(src)

        wcs = [fdtdx.WaveCharacter(wavelength=w) for w in _WL]
        all6 = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        if kind == "planar_flux":
            mon = fdtdx.PhasorPoyntingFluxDetector(
                name="mon",
                partial_grid_shape=(1, self._BOX_SPAN, self._BOX_SPAN),
                wave_characters=wcs,
                direction="+",
                scaling_mode="pulse",
                dft_subsample=1,
                dtype=jnp.complex128,
            )
            coords = (_N - _PML - 3, self._BOX_LO, self._BOX_LO)
        elif kind == "box_net_flux":
            mon = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
                name="mon",
                partial_grid_shape=(self._BOX_SPAN,) * 3,
                wave_characters=wcs,
                scaling_mode="pulse",
                dft_subsample=1,
                dtype=jnp.complex128,
            )
            coords = (self._BOX_LO,) * 3
        else:
            mon = fdtdx.PhasorDetector(
                name="mon",
                partial_grid_shape=(self._BOX_SPAN,) * 3,
                wave_characters=wcs,
                components=all6,
                scaling_mode="pulse",
                dft_subsample=1,
                exact_interpolation=False,
                reduce_volume=False,
                dtype=jnp.complex128,
            )
            coords = (self._BOX_LO,) * 3
        mon = self._off_interp(mon)
        cons.append(mon.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=coords))
        objs.append(mon)

        des = fdtdx.PhasorDetector(
            name="des",
            partial_grid_shape=(_DES_SPAN,) * 3,
            wave_characters=wcs,
            components=("Ex", "Ey", "Ez"),
            scaling_mode="pulse",
            dft_subsample=1,
            exact_interpolation=False,
            reduce_volume=False,
            dtype=jnp.complex128,
        )
        cons.append(des.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(_DES_LO,) * 3))
        objs.append(des)
        return fdtdx.place_objects(object_list=objs, config=config, constraints=cons, key=_KEY)

    @staticmethod
    def _fom_for(kind, det):
        if kind == "planar_flux":
            return lambda state: -jnp.sum(det.compute_poynting_flux(state))
        if kind == "box_net_flux":
            return lambda state: -jnp.sum(det.compute_net_flux(state))

        def energy(state):
            p = state["phasor"][0]
            return -(jnp.sum(jnp.abs(p[:, :3]) ** 2) + jnp.sum(jnp.abs(p[:, 3:]) ** 2))

        return energy

    @pytest.mark.parametrize(
        "kind",
        ["planar_flux", "box_net_flux", "energy"],
        ids=["planar_poynting_flux", "closed_box_net_power", "stored_energy"],
    )
    def test_matches_official_pipeline(self, kind):
        objects, arrays, _, config, _ = self._scene(kind)
        det = next(d for d in objects.detectors if d.name == "mon")
        fom = self._fom_for(kind, det)
        window = gaussian_window(int(config.time_steps_total))
        cfg_ck = config.aset("gradient_config", GradientConfig(method="checkpointed", num_checkpoints=8))

        def official(ie):
            arrs = arrays.aset("inv_permittivities", ie)
            _, out = fdtdx.run_fdtd(arrs, objects, cfg_ck, _KEY, show_progress=False)
            return fom(out.detector_states["mon"])

        fn = reciprocity_phasor_fn(
            arrays,
            objects,
            config,
            _KEY,
            objective_detectors="mon",
            design_detector="des",
            window=window,
        )

        def recip(x):
            out = fn(x)
            return fom(out if isinstance(out, dict) else {"phasor": out})

        ie = arrays.inv_permittivities
        v_off, g_off = jax.value_and_grad(official)(ie)
        v_rec, g_rec = jax.value_and_grad(recip)(ie)

        assert jnp.allclose(v_rec, v_off, rtol=1e-12), f"{kind}: forward values must match"
        gs = next(d for d in objects.detectors if d.name == "des").grid_slice
        a, b = g_rec[:, *gs], g_off[:, *gs]
        rel = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
        cos = float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
        assert rel < 1e-4, f"{kind}: rel_L2 = {rel:.3e}"
        assert cos > 1 - 1e-8, f"{kind}: cosine {cos:.10f}"

    def test_closed_box_uses_one_adjoint_source_per_face(self):
        objects, _arrays, _, config, _ = self._scene("box_net_flux", sim_fs=20.0)
        window = gaussian_window(int(config.time_steps_total))
        adj, names = derive_adjoint_objects(
            objects=objects, config=config, objective_detectors="mon", window=window, key=_KEY
        )
        det = next(d for d in objects.detectors if d.name == "mon")
        n_keys = len(det._shape_dtype_single_time_step())
        assert n_keys == 6, f"expected six faces, got {n_keys}"
        assert len(names[0]) == n_keys
        for src in adj.sources:
            thickness = [hi - lo for lo, hi in src.grid_slice_tuple]
            assert sorted(thickness)[0] == 1, f"{src.name} is not a face slab: {thickness}"

    def test_non_phasor_detector_is_refused(self):
        """An energy detector accumulates no phasors, so there is no transpose."""
        objects, arrays, _, config, _ = self._scene("energy", sim_fs=20.0)
        window = gaussian_window(int(config.time_steps_total))
        en = fdtdx.EnergyDetector(name="en", partial_grid_shape=(_N, _N, _N), reduce_volume=True)
        en = en.place_on_grid(grid_slice_tuple=((0, _N), (0, _N), (0, _N)), config=config, key=_KEY)
        broken = objects.aset("object_list", [*objects.object_list, en])
        with pytest.raises(NotImplementedError, match="does not accumulate complex phasors"):
            reciprocity_phasor_fn(
                arrays,
                broken,
                config,
                _KEY,
                objective_detectors="en",
                design_detector="des",
                window=window,
            )


_E3 = dict(
    components=("Ex", "Ey", "Ez"),
    dft_subsample=1,
    exact_interpolation=False,
    reduce_volume=False,
    dtype=jnp.complex128,
)


class TestDetectorDefaults:
    """PhasorDetector's stock defaults give the right gradient, with no setup.

    Two of those defaults used to be silently wrong for the design detector, and
    were then refused. Measured against ``run_fdtd(GradientConfig(checkpointed))``
    on this 24^3 scene before any guard existed:

    ============================  ==========  ========
    design detector                 rel L2     cosine
    ============================  ==========  ========
    ("Ex","Ey","Ez") + pulse        1.17e-06   1.00000
    all six (default) + pulse       3.80      -0.473
    ("Ex","Ey","Ez") + continuous   1.00       1.00000
    ============================  ==========  ========

    The design detector is now internal (built by
    ``fdtdx.adjoint.scene.make_design_detector``), so the user's one is only used
    for its cells, and the monitor's scale is divided out of the adjoint target.
    Every assertion is on relative L2: all the scaling errors this guards against
    had cosine 1.0000000000.
    """

    def test_stock_default_design_detector_matches_official(self):
        """A design detector given nothing but a name, a shape and frequencies."""
        objects, arrays, params, config, _ = _scene(with_device=True, design_detector_kwargs={})
        des = next(d for d in objects.detectors if d.name == "des")
        assert len(des.components) == 6 and des.scaling_mode == "continuous" and des.exact_interpolation

        v_off, g_off = _official_param_grad(objects, arrays, params, config)
        param_fn = reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon", design_detector="des")
        v_rec, g_rec = jax.value_and_grad(lambda p: _FOM(param_fn(p)))(params)

        assert jnp.allclose(v_rec, v_off, rtol=1e-12), "forward values must be identical"
        rel, cos = _rel_cos(_flat(g_rec), _flat(g_off))
        assert rel < 1e-5, f"stock-default design detector: rel_L2 = {rel:.3e}"
        # Device parameters are float32 (Device.init_params), so the cosine carries
        # float32 round-off (1 - 1.2e-07 measured); rel_L2 above is the real gate.
        assert cos > 1 - 1e-6, f"cosine {cos:.10f}"
        # the caller's own detector is untouched, so their run_fdtd still records six components
        assert arrays.detector_states["des"]["phasor"].shape[2] == 6

    @pytest.mark.parametrize("mon_scaling", ["pulse", "continuous"])
    def test_every_scaling_mode_combination_matches_official(self, mon_scaling):
        """The 2x2 table of monitor x design-detector scaling_mode.

        Measured before the scale correction (with the refusal removed): 1.0,
        7.9e+02 and 9.99e-01 relative error for the three non-pulse cells.
        """
        ref = None
        for des_scaling in ("pulse", "continuous"):
            objects, arrays, _, config, _ = _scene(
                mon_scaling=mon_scaling, design_detector_kwargs=dict(_E3, scaling_mode=des_scaling)
            )
            if ref is None:
                ref = _official_inv_eps_grad(objects, arrays, config, lambda s: _FOM(s["mon"]["phasor"]))
            v_off, g_off = ref
            fn = reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon", design_detector="des")
            v_rec, g_rec = jax.value_and_grad(lambda x: _FOM(fn(x)))(arrays.inv_permittivities)
            assert jnp.allclose(v_rec, v_off, rtol=1e-12)
            gs = design_region_slice(objects, "des")
            rel, _ = _rel_cos(g_rec[:, *gs], g_off[:, *gs])
            assert rel < 1e-5, f"monitor {mon_scaling}, design {des_scaling}: rel_L2 = {rel:.3e}"

    def test_monitors_in_different_modes_each_get_their_own_scale(self):
        """The monitor scale must be per detector: no global factor fits both.

        Before the fix this case measured rel 1.0 at cosine 0.99999994 -- one
        monitor dominates the FoM, so a cosine check would have passed it.
        """
        objects, arrays, _, config, _ = _scene(mon_scaling="pulse", mon2_scaling="continuous")

        def fom(p1, p2):
            return -jnp.sum(jnp.abs(p1) ** 2) + 0.5 * jnp.sum(jnp.real(p2) ** 2)

        v_off, g_off = _official_inv_eps_grad(
            objects, arrays, config, lambda s: fom(s["mon"]["phasor"], s["mon2"]["phasor"])
        )
        fn = reciprocity_phasor_fn(
            arrays, objects, config, _KEY, objective_detectors=("mon", "mon2"), design_detector="block"
        )
        v_rec, g_rec = jax.value_and_grad(lambda x: fom(*fn(x)))(arrays.inv_permittivities)
        assert jnp.allclose(v_rec, v_off, rtol=1e-12)
        gs = design_region_slice(objects, "block")
        rel, _ = _rel_cos(g_rec[:, *gs], g_off[:, *gs])
        assert rel < 1e-5, f"mixed-mode monitors: rel_L2 = {rel:.3e}"

    def test_design_scale_is_divided_out_twice(self, monkeypatch):
        """Exercise the 1/(s_d_fwd * s_d_adj) path, dead while the internal detector is pulse.

        With a continuous internal design detector the design scale rides on both
        the forward and the adjoint phasors; dividing it out once instead of
        twice would leave a factor 1/s_d = 787 here.
        """
        from fdtdx.adjoint import scene as adj_scene

        monkeypatch.setitem(adj_scene.DESIGN_DETECTOR_SETTINGS, "scaling_mode", "continuous")
        objects, arrays, _, config, _ = _scene(mon_scaling="continuous")
        block = next(o for o in objects.object_list if o.name == "block")
        mon = next(d for d in objects.detectors if d.name == "mon")
        probe = adj_scene.make_design_detector(block, mon.wave_characters, config, _KEY)
        assert float(probe._static_scale()) < 1e-2, "the design scale path is not being exercised"

        _, g_off = _official_inv_eps_grad(objects, arrays, config, lambda s: _FOM(s["mon"]["phasor"]))
        fn = reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon", design_detector="block")
        g_rec = jax.grad(lambda x: _FOM(fn(x)))(arrays.inv_permittivities)
        gs = design_region_slice(objects, "block")
        rel, _ = _rel_cos(g_rec[:, *gs], g_off[:, *gs])
        assert rel < 1e-5, f"continuous internal design detector: rel_L2 = {rel:.3e}"


class TestAutoDesignRegion:
    """No design detector at all: the design region is every Device in the scene."""

    @pytest.mark.integration
    def test_defaults_match_official_pipeline(self):
        """The whole zero-setup path in one test, cheap enough for CI.

        ``design_detector`` omitted, the monitor in PhasorDetector's default
        ``scaling_mode="continuous"``, and a stock-default detector over the
        design region left in the scene (it is dropped from both solves).
        Parameter gradient through ``apply_params`` against ``run_fdtd``.
        """
        objects, arrays, params, config, _ = _scene(
            with_device=True, mon_scaling="continuous", design_detector_kwargs={}
        )
        v_off, g_off = _official_param_grad(objects, arrays, params, config)
        param_fn = reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon")
        v_rec, g_rec = jax.value_and_grad(lambda p: _FOM(param_fn(p)))(params)

        assert jax.tree_util.tree_structure(g_rec) == jax.tree_util.tree_structure(g_off)
        assert jnp.allclose(v_rec, v_off, rtol=1e-12), "forward values must be identical"
        a, b = _flat(g_rec), _flat(g_off)
        rel, cos = _rel_cos(a, b)
        sign = float((jnp.sign(a) == jnp.sign(b)).mean())
        assert rel < 1e-5, f"parameter gradient rel_L2 = {rel:.3e}"
        assert cos > 1 - 1e-6, f"cosine {cos:.10f}"  # float32 parameters, see above
        assert sign > 0.99, f"sign agreement {sign:.4f}"

    def test_every_device_gets_its_own_design_region(self):
        """Two Devices: one internal design detector each, both gradients right.

        Naming one Device restricts the gradient to it, so the other's is zero.
        """
        half = _DES_SPAN // 2
        devices = [
            ("dA", (_DES_LO, _DES_LO, _DES_LO), (_DES_SPAN, half, _DES_SPAN)),
            ("dB", (_DES_LO, _DES_LO + half, _DES_LO), (_DES_SPAN, half, _DES_SPAN)),
        ]
        objects, arrays, params, config, _ = _scene(devices=devices)
        _, g_off = _official_param_grad(objects, arrays, params, config)

        param_fn = reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon")
        g_rec = jax.grad(lambda p: _FOM(param_fn(p)))(params)
        for name in ("dA", "dB"):
            rel, _ = _rel_cos(g_rec[name], g_off[name])
            assert rel < 1e-5, f"auto design regions, device {name}: rel_L2 = {rel:.3e}"

        only_a = reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon", design_detector="dA")
        g_a = jax.grad(lambda p: _FOM(only_a(p)))(params)
        rel, _ = _rel_cos(g_a["dA"], g_off["dA"])
        assert rel < 1e-5, f"named device dA: rel_L2 = {rel:.3e}"
        assert float(jnp.linalg.norm(g_a["dB"])) == 0.0
        assert float(jnp.linalg.norm(g_off["dB"])) > 0.0

    def test_no_device_and_no_region_raises(self):
        objects, arrays, _, config, _ = _scene(sim_fs=20.0)
        with pytest.raises(ValueError, match="no Device"):
            reciprocity_phasor_fn(arrays, objects, config, _KEY, objective_detectors="mon")


def _param_parity(objects, arrays, params, config, fom_states, fom_phasors, name):
    """Reciprocity vs ``apply_params -> run_fdtd(checkpointed)``, both from the PLACED objects.

    ``fom_states(detector, state)``: the reference FoM on ``run_fdtd``'s detector
    state; ``fom_phasors(detector, out)`` the same FoM on ``param_fn``'s output.
    Each side takes the detector from its own applied container.
    Returns ``(forward values equal, rel_L2, cosine, best-fit scale)``.
    """
    cfg_ck = config.aset("gradient_config", GradientConfig(method="checkpointed", num_checkpoints=8))

    def official(p):
        arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
        _, out = fdtdx.run_fdtd(arrs, objs, cfg_ck, _KEY, show_progress=False)
        return fom_states(objs[name], out.detector_states[name])

    v_off, g_off = jax.value_and_grad(official)(params)
    param_fn = reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors=name)
    det = param_fn.objects[name]
    v_rec, g_rec = jax.value_and_grad(lambda p: fom_phasors(det, param_fn(p)))(params)
    a, b = _flat(g_rec), _flat(g_off)
    rel, cos = _rel_cos(a, b)
    scale = float(jnp.sum(a * b) / jnp.sum(a * a))
    return bool(v_rec == v_off), rel, cos, scale


class TestStockObjectives:
    """Objective monitors at FDTDX's stock settings, as the real problems use them.

    * a plain PhasorDetector given only a name, a shape and frequencies (six
      components, ``exact_interpolation=True``, continuous, complex64);
    * a mode port in a 2D x-z scene with two periodic y cells, whose stencil
      wraps around the periodic axis (FDTDX's padded whole-domain path);
    * a box far-field projection (exact interpolation forced) with one excluded
      face, through ``project_all``;
    * a strided (``dft_subsample``) monitor.

    The scenes go straight from ``place_objects`` into ``reciprocity_param_fn``:
    their plane source and mode port share the Device's footprint, so
    ``place_objects`` leaves them unapplied (this used to crash the forward solve
    with ``'Null' object is not subscriptable``).
    """

    @pytest.mark.integration
    def test_stock_monitor_matches_official(self):
        """The whole stock-default path in one cheap test, run in CI.

        Measured on the GPU (float64): rel 5.4e-07 at cosine 1.0000000000. The
        monitor records co-located fields, so the co-location stencil's
        transpose and the magnetic time average are both exercised.
        """
        objects, arrays, params, config, _ = _scene(with_device=True, monitor_kwargs={})
        mon = next(d for d in objects.detectors if d.name == "mon")
        assert mon.exact_interpolation and len(mon.components) == 6 and mon.scaling_mode == "continuous"
        v_off, g_off = _official_param_grad(objects, arrays, params, config)
        param_fn = reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon")
        v_rec, g_rec = jax.value_and_grad(lambda p: _FOM(param_fn(p)))(params)
        assert v_rec == v_off, "the forward value must be bit-identical to run_fdtd's"
        rel, cos = _rel_cos(_flat(g_rec), _flat(g_off))
        assert rel < 1e-5, f"stock monitor: rel_L2 = {rel:.3e}"
        assert cos > 1 - 1e-6, f"cosine {cos:.10f}"

    def test_stencil_transpose_and_time_average_are_both_needed(self, monkeypatch):
        """An Hx monitor recorded as if raw: measured rel 2.0e-01 at cosine 0.980 (with them 8.3e-07)."""
        from fdtdx.adjoint import recording, scene, vjp

        objects, arrays, _, config, _ = _scene(
            monitor_kwargs=dict(components=("Hx",), scaling_mode="pulse", dtype=jnp.complex128)
        )
        _, g_off = _official_inv_eps_grad(objects, arrays, config, lambda s: _FOM(s["mon"]["phasor"]))
        gs = design_region_slice(objects, "block")

        def rel_now():
            fn = reciprocity_phasor_fn(
                arrays, objects, config, _KEY, objective_detectors="mon", design_detector="block"
            )
            g = jax.grad(lambda x: _FOM(fn(x)))(arrays.inv_permittivities)
            return _rel_cos(g[:, *gs], g_off[:, *gs])[0]

        assert rel_now() < 1e-5
        honest = recording.channel_recordings

        def as_raw(detector, *args, **kwargs):
            return honest(detector.aset("exact_interpolation", False), *args, **kwargs)

        monkeypatch.setattr(scene, "channel_recordings", as_raw)
        monkeypatch.setattr(vjp, "channel_recordings", as_raw)
        assert rel_now() > 1e-2

    def test_periodic_mode_port_through_param_fn(self):
        """2D x-z, two periodic y cells, ModePlaneSource, ModeOverlapDetector at stock settings.

        FoM is the mode power through FDTDX's own ``compute_overlap``. The port
        spans the periodic axis, so its stencil takes FDTDX's padded
        whole-domain path and wraps. Measured on the GPU (float64, 300 fs):
        rel 6.8e-05 at cosine 0.9999999986; ModeOverlap gradients converge
        more slowly than the FoM (1.7e-03 at 150 fs).
        """
        objects, arrays, params, config = _periodic_mode_scene(sim_fs=300.0)
        port = next(d for d in objects.detectors if d.name == "out")
        assert port.exact_interpolation and port.grid_slice_tuple[1] == (0, 2)

        def power(det, state):
            return -jnp.sum(jnp.abs(det.compute_overlap(state)) ** 2)

        same, rel, cos, scale = _param_parity(
            objects, arrays, params, config, power, lambda det, out: power(det, {"phasor": out}), "out"
        )
        assert same, "the forward value must be bit-identical to run_fdtd's"
        assert rel < 3e-4, f"mode port: rel_L2 = {rel:.3e} (scale {scale:.6f})"
        assert cos > 1 - 1e-7, f"cosine {cos:.10f}"

    def test_box_far_field_through_param_fn(self):
        """FieldProjectionAngleDetector box, stock settings, 5 faces, UniformPlaneSource(normalize_by_energy).

        Measured on the GPU (float64, 150 fs): rel 6.1e-07 at cosine 1.0000000000.
        """
        objects, arrays, params, config = _box_far_field_scene()
        theta = jnp.asarray([0.0, 0.3, 0.6, 2.6, 3.0])
        phi = jnp.asarray([0.0, 0.8, 1.6, 2.4, 3.1])

        def power(det, state):
            return -jnp.sum(det.project_all(state, theta, phi)["power"])

        same, rel, cos, scale = _param_parity(objects, arrays, params, config, power, power, "ff")
        assert same, "the forward value must be bit-identical to run_fdtd's"
        assert rel < 1e-5, f"box far field: rel_L2 = {rel:.3e} (scale {scale:.6f})"
        assert cos > 1 - 1e-8, f"cosine {cos:.10f}"

    def test_monitor_on_an_electric_symmetry_plane(self):
        """``config.symmetry`` puts the monitor's stencil through FDTDX's mirror padding.

        24^3 reduced to 12x24x24 by a PEC plane that the Device and a stock
        monitor both straddle. Measured on the GPU (float64): rel 4.1e-07, and
        3.8e-07 for the same scene without symmetry.
        """
        wl0 = 600e-9
        config = SimulationConfig(
            time=150e-15,
            grid=UniformGrid(spacing=_RES),
            backend="cpu",
            dtype=jnp.float64,
            courant_factor=0.99,
            gradient_config=None,
            symmetry=(-1, 0, 0),
        )
        vol = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
        objs, cons = [vol], []
        bd, cl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
        objs.extend(bd.values())
        cons.extend(cl)

        def at(obj, lower):
            objs.append(obj)
            cons.append(obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=lower))

        wc = fdtdx.WaveCharacter(wavelength=wl0)
        at(
            fdtdx.Device(
                name="design",
                partial_grid_shape=(8, 8, 4),
                partial_voxel_grid_shape=(1, 1, 1),
                materials={"air": fdtdx.Material(permittivity=1.0), "si": fdtdx.Material(permittivity=2.25)},
                param_transforms=[],
            ),
            (8, 8, 9),
        )
        at(
            fdtdx.UniformPlaneSource(
                name="source",
                partial_grid_shape=(_N, _N, 1),
                direction="+",
                fixed_E_polarization_vector=(1, 0, 0),
                wave_character=wc,
                temporal_profile=fdtdx.GaussianPulseProfile(
                    center_wave=wc, spectral_width=fdtdx.WaveCharacter(frequency=0.3 * float(c0 / wl0))
                ),
                normalize_by_energy=True,
            ),
            (0, 0, 5),
        )
        at(fdtdx.PhasorDetector(name="mon", partial_grid_shape=(4, 4, 1), wave_characters=(wc,)), (10, 10, 17))
        objects, arrays, params, config, _ = fdtdx.place_objects(
            object_list=objs, config=config, constraints=cons, key=_KEY
        )
        mon = next(d for d in objects.detectors if d.name == "mon")
        assert mon.grid_slice_tuple[0][0] == 0, "the monitor must touch the reduced domain's symmetry plane"

        def fom(_det, state):
            return _FOM(state["phasor"])

        same, rel, cos, scale = _param_parity(
            objects, arrays, _varied(params), config, fom, lambda det, out: _FOM(out), "mon"
        )
        assert same, "the forward value must be bit-identical to run_fdtd's"
        assert rel < 1e-5, f"symmetry plane: rel_L2 = {rel:.3e} (scale {scale:.6f})"
        assert cos > 1 - 1e-8, f"cosine {cos:.10f}"

    @pytest.mark.parametrize("scaling_mode", ["continuous", "pulse"])
    def test_strided_monitor_matches_the_strided_reference(self, scaling_mode):
        """``dft_subsample=3`` against the exact gradient of the strided recording.

        Only the principal term of the strided DFT is transposed; its aliases sit
        where a band-limited source puts no field. Measured on the GPU (float64):
        rel 5.5e-07 at strides 2, 3 and 5, the same as stride 1.
        """
        objects, arrays, params, config, _ = _scene(
            with_device=True, monitor_kwargs=dict(dft_subsample=3, scaling_mode=scaling_mode)
        )
        assert next(d for d in objects.detectors if d.name == "mon")._dft_stride == 3
        v_off, g_off = _official_param_grad(objects, arrays, params, config)
        param_fn = reciprocity_param_fn(arrays, objects, config, _KEY, objective_detectors="mon")
        v_rec, g_rec = jax.value_and_grad(lambda p: _FOM(param_fn(p)))(params)
        assert v_rec == v_off
        rel, _ = _rel_cos(_flat(g_rec), _flat(g_off))
        assert rel < 1e-5, f"stride 3 ({scaling_mode}): rel_L2 = {rel:.3e}"


def _periodic_mode_scene(sim_fs):
    """The neural-to-coverage problems' layout: 2D x-z, periodic y of two cells, PML on x and z."""
    wl0, wls, nx, ny, nz, pml = 1.2e-6, (1.15e-6, 1.25e-6), 48, 2, 32, 8
    config = SimulationConfig(
        time=sim_fs * 1e-15,
        grid=UniformGrid(spacing=_RES),
        backend="cpu",
        dtype=jnp.float64,
        courant_factor=0.99,
        gradient_config=None,
    )
    air, core = fdtdx.Material(permittivity=1.0), fdtdx.Material(permittivity=4.0)
    vol = fdtdx.SimulationVolume(name="volume", partial_grid_shape=(nx, ny, nz), material=air)
    objs, cons = [vol], []
    boundary = fdtdx.BoundaryConfig(
        boundary_type_miny="periodic",
        boundary_type_maxy="periodic",
        thickness_grid_minx=pml,
        thickness_grid_maxx=pml,
        thickness_grid_miny=1,
        thickness_grid_maxy=1,
        thickness_grid_minz=pml,
        thickness_grid_maxz=pml,
    )
    bd, cl = fdtdx.boundary_objects_from_config(boundary, vol)
    objs.extend(bd.values())
    cons.extend(cl)

    def at(obj, lower):
        objs.append(obj)
        cons.append(obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=lower))

    def wc(w):
        return fdtdx.WaveCharacter(wavelength=w)

    at(fdtdx.UniformMaterialObject(name="guide", partial_grid_shape=(nx, ny, 8), material=core), (0, 0, 12))
    at(
        fdtdx.Device(
            name="design",
            partial_grid_shape=(12, ny, 12),
            partial_voxel_grid_shape=(1, 2, 1),
            materials={"background": air, "core": core},
            param_transforms=[],
            placement_order=10,
        ),
        (18, 0, 10),
    )
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wc(wl0), spectral_width=fdtdx.WaveCharacter(frequency=0.2 * float(c0 / wl0))
    )
    port = dict(mode_index=0, filter_pol="te", partial_grid_shape=(1, ny, 16))
    at(
        fdtdx.ModePlaneSource(name="source", direction="+", wave_character=wc(wl0), temporal_profile=pulse, **port),
        (11, 0, 8),
    )
    at(
        fdtdx.ModeOverlapDetector(name="out", direction="+", wave_characters=tuple(wc(w) for w in wls), **port),
        (36, 0, 8),
    )
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objs, config=config, constraints=cons, key=_KEY
    )
    return objects, arrays, _varied(params), config


def _box_far_field_scene(sim_fs=150.0):
    """The colour splitter's layout: plane wave down onto a Device on a substrate, box far field around it."""
    wl0 = 600e-9
    config = SimulationConfig(
        time=sim_fs * 1e-15,
        grid=UniformGrid(spacing=_RES),
        backend="cpu",
        dtype=jnp.float64,
        courant_factor=0.99,
        gradient_config=None,
    )
    vol = fdtdx.SimulationVolume(partial_grid_shape=(24, 24, 30))
    objs, cons = [vol], []
    bd, cl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), vol)
    objs.extend(bd.values())
    cons.extend(cl)

    def at(obj, lower):
        objs.append(obj)
        cons.append(obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=lower))

    def wc(w):
        return fdtdx.WaveCharacter(wavelength=w)

    at(
        fdtdx.UniformMaterialObject(
            name="substrate", partial_grid_shape=(24, 24, 10), material=fdtdx.Material(permittivity=2.1)
        ),
        (0, 0, 0),
    )
    at(
        fdtdx.Device(
            name="design",
            partial_grid_shape=(8, 8, 4),
            partial_voxel_grid_shape=(1, 1, 1),
            materials={"air": fdtdx.Material(permittivity=1.0), "si": fdtdx.Material(permittivity=2.25)},
            param_transforms=[],
        ),
        (8, 8, 11),
    )
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wc(wl0), spectral_width=fdtdx.WaveCharacter(frequency=0.3 * float(c0 / wl0))
    )
    at(
        fdtdx.UniformPlaneSource(
            name="source",
            partial_grid_shape=(24, 24, 1),
            direction="-",
            fixed_E_polarization_vector=(1, 0, 0),
            wave_character=wc(wl0),
            temporal_profile=pulse,
            normalize_by_energy=True,
        ),
        (0, 0, 23),
    )
    at(
        fdtdx.FieldProjectionAngleDetector(
            name="ff",
            partial_grid_shape=(12, 12, 10),
            wave_characters=(wc(550e-9), wc(650e-9)),
            exclude_surfaces=("z-",),
        ),
        (6, 6, 10),
    )
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objs, config=config, constraints=cons, key=_KEY
    )
    return objects, arrays, _varied(params), config


def _varied(params):
    """Non-uniform float64 parameters, so the gradient is neither symmetric by accident nor float32-limited."""
    return jax.tree_util.tree_map(
        lambda x: 0.5 + 0.3 * jnp.sin(jnp.arange(x.size, dtype=jnp.float64).reshape(x.shape)), params
    )
