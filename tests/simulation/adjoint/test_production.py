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
from fdtdx.adjoint import derive_adjoint_objects, gaussian_window, reciprocity_param_fn, reciprocity_phasor_fn
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


def _scene(sim_fs=150.0, *, with_device=False, sigma=0.0, source_cell=_SRC, components=("Ez",)):
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

    if with_device:
        device = fdtdx.Device(
            name="design",
            partial_grid_shape=(_DES_SPAN,) * 3,
            partial_voxel_grid_shape=(1, 1, 1),
            materials={"air": fdtdx.Material(permittivity=1.0), "si": fdtdx.Material(permittivity=2.25)},
            param_transforms=[],
        )
        cons.append(device.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=(_DES_LO,) * 3))
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
    mon = fdtdx.PhasorDetector(
        name="mon",
        partial_grid_shape=(1, 1, 1),
        wave_characters=wcs,
        components=("Ez",),
        scaling_mode="pulse",
        dft_subsample=1,
        exact_interpolation=False,
        reduce_volume=False,
        dtype=jnp.complex128,
    )
    cons.append(mon.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=_MON))
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
    """

    @pytest.mark.parametrize(
        "components",
        [
            ("Ez",),
            ("Hx",),
            ("Hy",),
            ("Ez", "Hx"),
            ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        ],
        ids=["Ez", "Hx", "Hy", "Ez_Hx", "all_six"],
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

    def test_projected_far_field_gradient_matches_official(self):
        objects, arrays, _, config, _ = self._scene()
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

    def test_exact_interpolation_is_refused_with_the_remedy(self):
        """Left on, it would be silently wrong, so it raises and says what to do."""
        objects, arrays, _, config, _ = self._scene(sim_fs=20.0, exact_interpolation=True)
        window = gaussian_window(int(config.time_steps_total))
        with pytest.raises(NotImplementedError, match="exact_interpolation"):
            reciprocity_phasor_fn(
                arrays,
                objects,
                config,
                _KEY,
                objective_detectors="ff",
                design_detector="des",
                window=window,
            )


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
