"""``GradientConfig(method="reciprocity")`` against ``method="checkpointed"``, through ``apply_params -> run_fdtd``.

Every test is an inverse-design script with the method string changed: the forward value must be
``run_fdtd``'s bit for bit, and the Device-parameter gradient checkpointed's (relative L2 and
best-fit scale at 1e-5 unless noted). Refusals that need no solve are in
``tests/unit/adjoint/test_refusals.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.adjoint import validation
from fdtdx.adjoint.solve import AdjointSolve
from fdtdx.adjoint.source import AdjointCurrentSource
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import QuasiUniformGrid, UniformGrid
from fdtdx.fdtd.initialization import apply_params

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


def _config(sim_fs=150.0, grid=None, symmetry=(0, 0, 0)):
    return SimulationConfig(
        time=sim_fs * 1e-15,
        grid=grid or UniformGrid(spacing=_RES),
        backend="cpu",
        dtype=jnp.float64,
        courant_factor=0.99,
        gradient_config=None,
        symmetry=symmetry,
    )


def _narrowband_dipole(name):
    """A dipole narrow enough that the run outlasts it: at 0.1 f0 the reciprocity error is 2.5e-07,
    at 0.4 f0 (about 2.5 optical cycles) 2.4e-03."""
    wc = fdtdx.WaveCharacter(wavelength=_WL[0])
    return fdtdx.PointDipoleSource(
        name=name,
        partial_grid_shape=(1, 1, 1),
        wave_character=wc,
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_wave=wc, spectral_width=fdtdx.WaveCharacter(frequency=float(c0 / _WL[0]) * 0.1)
        ),
        polarization=2,
        amplitude=1.0,
    )


def _device(name, shape, material=None, etched=False):
    if etched:
        materials = {"etch": material or fdtdx.Material(permittivity=1.0)}
    else:
        materials = {"air": fdtdx.Material(permittivity=1.0), "si": material or fdtdx.Material(permittivity=2.25)}
    return fdtdx.Device(
        name=name,
        partial_grid_shape=shape,
        partial_voxel_grid_shape=(1, 1, 1),
        materials=materials,
        param_transforms=[],
        use_etching=etched,
    )


class _Scene:
    """A placement list: objects at index-space lower corners, or by physical margins on a non-uniform grid."""

    def __init__(self, config, shape=(_N, _N, _N), pml=_PML):
        self.config = config
        self.vol = fdtdx.SimulationVolume(partial_grid_shape=shape)
        self.objs, self.cons = [self.vol], []
        bd, cl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=pml), self.vol)
        self.objs.extend(bd.values())
        self.cons.extend(cl)
        grid = config.grid
        self.edges = None if isinstance(grid, UniformGrid) else [config.resolve_grid(shape).edges(a) for a in range(3)]

    def at(self, obj, lower):
        self.objs.append(obj)
        if self.edges is None:
            self.cons.append(obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=lower))
        else:  # index-space placement is refused on a non-uniform grid
            margins = tuple(float(e[i] - e[0]) for e, i in zip(self.edges, lower))
            self.cons.append(obj.place_relative_to(self.vol, (0, 1, 2), (-1,) * 3, (-1,) * 3, margins=margins))

    def place(self):
        return fdtdx.place_objects(object_list=self.objs, config=self.config, constraints=self.cons, key=_KEY)


_POINT = dict(dft_subsample=1, exact_interpolation=False, reduce_volume=False, dtype=jnp.complex128)


def _scene(
    sim_fs=150.0,
    *,
    devices=None,
    source_cell=_SRC,
    components=("Ez",),
    mon_scaling="pulse",
    mon2_scaling=None,
    monitor_kwargs=None,
    extra_blocks=(),
    monitor_shape=(1, 1, 1),
    monitor_cell=_MON,
    grid=None,
    device_material=None,
    lead_wavelengths=None,
    wavelengths=_WL,
    etched=False,
    stock_detector_over_design=False,
):
    """24^3 test scene: a dipole, a Device ``"design"`` over the middle, a one-cell monitor ``"mon"``.

    ``devices`` is a list of ``(name, lower_corner, shape)`` replacing ``"design"``;
    ``monitor_kwargs``, when given, replaces every setting of ``"mon"`` except its name, shape and
    frequencies (``{}`` is PhasorDetector's stock defaults); ``mon2_scaling`` adds a second monitor
    ``"mon2"``; ``lead_wavelengths`` adds a one-cell monitor ``"lead"`` at those wavelengths, listed
    before ``"mon"``. ``extra_blocks`` are static ``(name, lower, shape, material)`` blocks placed
    before the Devices, so a Device placed over one covers it. ``grid`` replaces the 50 nm cubes.
    ``device_material`` replaces the Devices' ``si`` (eps 2.25); ``etched`` makes them etch it (air
    by default) into what lies below. ``wavelengths`` are the monitors' (the source stays a 0.1 f0
    pulse at 600 nm). ``stock_detector_over_design`` adds an unread stock detector ``"des"``.
    """
    scene = _Scene(_config(sim_fs, grid))
    for name, lower, shape, material in extra_blocks:
        scene.at(fdtdx.UniformMaterialObject(name=name, partial_grid_shape=shape, material=material), lower)
    for name, lower, shape in devices or [("design", (_DES_LO,) * 3, (_DES_SPAN,) * 3)]:
        scene.at(_device(name, shape, device_material, etched), lower)
    scene.at(_narrowband_dipole("src"), source_cell)
    wcs = [fdtdx.WaveCharacter(wavelength=w) for w in wavelengths]
    if lead_wavelengths is not None:
        lead_wcs = [fdtdx.WaveCharacter(wavelength=w) for w in lead_wavelengths]
        scene.at(fdtdx.PhasorDetector(name="lead", partial_grid_shape=(1, 1, 1), wave_characters=lead_wcs), _MON)
    if monitor_kwargs is None:
        monitor_kwargs = dict(_POINT, components=components, scaling_mode=mon_scaling)
    scene.at(
        fdtdx.PhasorDetector(name="mon", partial_grid_shape=monitor_shape, wave_characters=wcs, **monitor_kwargs),
        monitor_cell,
    )
    if mon2_scaling is not None:
        mon2 = fdtdx.PhasorDetector(
            name="mon2", partial_grid_shape=(1, 1, 1), wave_characters=wcs, scaling_mode=mon2_scaling, **_POINT
        )
        scene.at(mon2, (_MON[0], _MON[1] + 2, _MON[2]))
    if stock_detector_over_design:
        scene.at(
            fdtdx.PhasorDetector(name="des", partial_grid_shape=(_DES_SPAN,) * 3, wave_characters=wcs), (_DES_LO,) * 3
        )
    return scene.place()


def _FOM(P):
    return -jnp.sum(jnp.abs(P) ** 2)


def _mon_fom(objs, states):
    return _FOM(states["mon"]["phasor"])


def _flat(tree):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree)])


def _varied(params):
    """Non-uniform float64 parameters, so the gradient is neither symmetric by accident nor float32-limited."""
    return jax.tree_util.tree_map(
        lambda x: 0.5 + 0.3 * jnp.sin(jnp.arange(x.size, dtype=jnp.float64).reshape(x.shape)), params
    )


def _loss(arrays, objects, config, method, fom=_mon_fom, tail_tolerance=1e-2):
    """An inverse-design script, ``apply_params -> run_fdtd -> fom(objects, detector_states)``, with the
    state as aux; the gradients compared differ only in ``method``."""
    gradient_config = GradientConfig(method=method, num_checkpoints=8, tail_tolerance=tail_tolerance)
    config = config.aset("gradient_config", gradient_config)

    def loss(p):
        arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
        state = fdtdx.run_fdtd(arrs, objs, config, _KEY, show_progress=False)
        return fom(objs, state[1].detector_states), state

    return loss


def _value_and_grad(arrays, objects, params, config, method, fom=_mon_fom, **kwargs):
    (value, _), grad = jax.value_and_grad(_loss(arrays, objects, config, method, fom, **kwargs), has_aux=True)(params)
    return value, grad


def _parity_metrics(g_ref, g_rec):
    """``(rel_L2, cosine, best-fit scale)`` of ``g_rec`` against ``g_ref``; the scale is
    ``sum(ref * rec) / sum(ref * ref)``, so a pure-scale error shows as cosine 1 with a scale away from 1."""
    a, b = _flat(g_ref), _flat(g_rec)
    rel = float(jnp.linalg.norm(b - a) / jnp.linalg.norm(a))
    cos = float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
    return rel, cos, float(jnp.sum(a * b) / jnp.sum(a * a))


def _parity(objects, arrays, params, config, fom=_mon_fom):
    """Reciprocity against checkpointed: ``(forward values equal, rel_L2, cosine, best-fit scale, both grads)``."""
    v_ck, g_ck = _value_and_grad(arrays, objects, params, config, "checkpointed", fom)
    v_rc, g_rc = _value_and_grad(arrays, objects, params, config, "reciprocity", fom)
    return (bool(v_rc == v_ck), *_parity_metrics(g_ck, g_rc), (g_ck, g_rc))


def _assert_parity(objects, arrays, params, config, fom=_mon_fom, tolerance=1e-5):
    same, rel, cos, scale, grads = _parity(objects, arrays, params, config, fom)
    assert same, "the forward value must be run_fdtd's, bit for bit"
    assert rel < tolerance and abs(scale - 1) < tolerance, f"rel {rel:.3e} cos {cos:.10f} scale {scale:.9f}"
    return grads


def _assert_device_parity(fom=_mon_fom, **scene_kwargs):
    objects, arrays, params, config, _ = _scene(**scene_kwargs)
    return _assert_parity(objects, arrays, params, config, fom)


def _adjoint_solve(objects, arrays, config, name):
    """The backward rule's setup for the objective monitor ``name``, without a solve."""
    return AdjointSolve(
        arrays, objects, config, _KEY, names=(name,), tolerance=None, forward_scales=[1.0] * len(objects.devices)
    )


def _bit_identical(a, b):
    leaves = zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    return jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b) and all(
        np.array_equal(x, y) for x, y in leaves
    )


class TestPipeline:
    """An existing script with one string changed."""

    def test_matches_checkpointed_on_device_params(self):
        objects, arrays, params, config, _ = _scene()
        g_ck, g_rc = _assert_parity(objects, arrays, params, config)
        assert jax.tree_util.tree_structure(g_rc) == jax.tree_util.tree_structure(g_ck)
        sign = float((jnp.sign(_flat(g_rc)) == jnp.sign(_flat(g_ck))).mean())
        assert sign > 0.99, f"sign agreement {sign:.4f}"

    @pytest.mark.integration
    def test_the_forward_is_run_fdtds(self):
        """With and without a gradient, every output (value, fields, detector states) is checkpointed's,
        bit for bit. Stock PhasorDetector; measured on the GPU (float64): gradient rel 4.2e-07."""
        objects, arrays, params, config, _ = _scene(monitor_kwargs={})
        (v_ck, s_ck), _ = jax.value_and_grad(_loss(arrays, objects, config, "checkpointed"), has_aux=True)(params)
        (v_rc, s_rc), _ = jax.value_and_grad(_loss(arrays, objects, config, "reciprocity"), has_aux=True)(params)
        assert v_rc == v_ck and _bit_identical(s_rc, s_ck), "the forward under jax.grad must be run_fdtd's"
        plain = [_loss(arrays, objects, config, m)(params)[1] for m in ("checkpointed", "reciprocity")]
        assert _bit_identical(*plain), "the forward without a gradient must be run_fdtd's"

    def test_second_place_objects_would_change_device_params(self):
        """Why the adjoint scene is derived from the placed one rather than placed again:
        ``place_objects`` splits its key once per object, so two scenes differing only in their
        object count get different initial Device parameters."""
        _, _, p_one, _, _ = _scene(sim_fs=20.0)
        scene = _Scene(_config(20.0))
        scene.at(_device("design", (_DES_SPAN,) * 3), (_DES_LO,) * 3)
        # the same Device and one object more than _scene's source and monitor
        for i, cell in enumerate((_SRC, _MON, (_SRC[0], _SRC[1] + 2, _SRC[2]))):
            scene.at(_narrowband_dipole(f"s{i}"), cell)
        _, _, p_two, _, _ = scene.place()
        assert not bool(jnp.all(_flat(p_one) == _flat(p_two))), (
            "object count no longer perturbs Device parameters; the derived adjoint scene may no longer be "
            "necessary, re-check before simplifying"
        )

    def test_the_adjoint_scene_keeps_geometry_and_swaps_the_source(self):
        objects, arrays, _, config, _ = _scene(sim_fs=20.0, extra_blocks=[("blk", (5, 16, 16), (2, 2, 2), _LOSSY)])
        solve = _adjoint_solve(objects, arrays, config, "mon")
        assert len(solve.objects.sources) == 1 and isinstance(solve.objects.sources[0], AdjointCurrentSource)
        assert solve.objects.sources[0].grid_slice_tuple == objects["mon"].grid_slice_tuple
        assert {o.name for o in solve.objects.static_material_objects} == {
            o.name for o in objects.static_material_objects
        }
        assert solve.objects.volume.grid_shape == objects.volume.grid_shape

    def test_every_device_gets_its_own_design_region(self):
        half = _DES_SPAN // 2
        devices = [
            ("dA", (_DES_LO, _DES_LO, _DES_LO), (_DES_SPAN, half, _DES_SPAN)),
            ("dB", (_DES_LO, _DES_LO + half, _DES_LO), (_DES_SPAN, half, _DES_SPAN)),
        ]
        g_ck, _ = _assert_device_parity(devices=devices)
        assert float(jnp.linalg.norm(g_ck["dB"])) > 0.0

    def test_overlapping_devices(self):
        """Both write the shared cells from the same fields: set, not added (adding was rel 9.7e-01 at
        best-fit scale 1.93, silently); ``apply_params`` keeps the later Device there."""
        devices = [
            ("dA", (_DES_LO,) * 3, (_DES_SPAN,) * 3),
            ("dB", (_DES_LO + 2,) * 3, (_DES_SPAN - 2,) * 3),
        ]
        _assert_device_parity(devices=devices)

    @pytest.mark.integration
    def test_unread_stock_detectors_and_a_continuous_monitor(self):
        """The whole stock path: a monitor in PhasorDetector's default continuous mode, and a stock detector
        over the Device the figure of merit does not read (dropped from both solves)."""
        _assert_device_parity(mon_scaling="continuous", stock_detector_over_design=True)

    def test_unread_monitors_get_no_adjoint_current_and_jit_sets_up_once(self, monkeypatch):
        """``lead``, ``mon2`` and ``des`` are recorded but not read: their cotangents are symbolic zeros, so
        the adjoint solve drives ``mon`` alone. Under ``jax.jit`` the setup runs at trace time, once. ``lead``
        records another frequency and is listed first: design phasors recorded at its frequency only were rel
        2.0 (cosine -0.71), the objective's row taken by position rel 0.67."""
        import fdtdx.adjoint.reciprocity as reciprocity

        built = []

        class Spy(reciprocity.AdjointSolve):
            def __init__(self, *args, names, **kwargs):
                built.append(names)
                super().__init__(*args, names=names, **kwargs)

        monkeypatch.setattr(reciprocity, "AdjointSolve", Spy)
        objects, arrays, params, config, _ = _scene(
            mon2_scaling="pulse", lead_wavelengths=(650e-9,), stock_detector_over_design=True
        )
        loss = _loss(arrays, objects, config, "reciprocity")
        grad = jax.jit(jax.grad(lambda p: loss(p)[0]))
        g_rc = grad(params)
        grad(jax.tree_util.tree_map(lambda x: 0.9 * x, params))
        assert built == [("mon",)]
        _, g_ck = _value_and_grad(arrays, objects, params, config, "checkpointed")
        rel, _, scale = _parity_metrics(g_ck, g_rc)
        assert rel < 1e-5 and abs(scale - 1) < 1e-5

    def test_progress_is_one_run(self):
        """``progress_callback`` sees one run from 0 to the end, as with checkpointed, though the solve runs in
        segments."""
        objects, arrays, params, config, _ = _scene(sim_fs=60.0)
        config = config.aset("gradient_config", GradientConfig(method="reciprocity"))
        seen = []

        def loss(p):
            arrs, objs, _ = apply_params(arrays, objects, p, _KEY)
            _, out = fdtdx.run_fdtd(
                arrs, objs, config, _KEY, show_progress=False, progress_callback=lambda s, n: seen.append((s, n))
            )
            return _FOM(out.detector_states["mon"]["phasor"])

        jax.grad(loss)(params)
        jax.effects_barrier()
        n = int(config.time_steps_total)
        steps = [s for s, _ in seen]
        assert {total for _, total in seen} == {n}
        assert steps == sorted(steps) and steps[-1] == n and steps.count(n) == 1


class TestMagneticComponents:
    """Objectives on H. Magnetic components carry an extra factor ``-exp(-i w dt)``: the minus from
    Lorentz reciprocity's asymmetry between the electric and magnetic pairings, and two half steps (the
    detector stores the post-update H, at n + 1/2, and weights it with the integer-step kernel; the
    current enters that update with its carrier at n). Measured on an Hx objective: no factor 1.91 at
    cosine -0.998, sign flip alone 1.04e-01, both 6.3e-07.

    ``("Hx", "Ez")`` is declared out of order on purpose: PhasorDetector stores components in canonical
    order, and the adjoint current once followed the declared order, driving each cotangent into the
    other component.
    """

    @pytest.mark.parametrize(
        "components",
        [
            ("Ez",),
            ("Hx",),
            ("Hy",),
            ("Ez", "Hx"),
            pytest.param(("Hx", "Ez"), marks=pytest.mark.integration),
            ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        ],
        ids=["Ez", "Hx", "Hy", "Ez_Hx", "Hx_Ez_declared_out_of_order", "all_six"],
    )
    def test_matches_checkpointed_for_any_component_set(self, components):
        objects, arrays, params, config, _ = _scene(components=components)
        assert set(objects["mon"].components) == set(components), "the scene must actually use these components"
        _assert_parity(objects, arrays, params, config)

    @pytest.mark.parametrize("components", [("Hx",), ("Ez", "Hx")], ids=["Hx", "Ez_Hx"])
    def test_two_close_frequencies(self, components):
        """594 and 606 nm: their windowed spectra overlap, so the amplitude solve couples them, and the H
        current's timing must be inside the solve, not a per-frequency factor on its target (rel 3.5e-03)."""
        _assert_device_parity(components=components, wavelengths=(594e-9, 606e-9))

    def test_magnetic_sign_flip_is_actually_needed(self):
        """Without the flip the Hx gradient anti-correlates with the truth, so the optimizer would walk uphill."""
        from fdtdx.adjoint.objective import target_factor

        objects, _, _, config, _ = _scene(sim_fs=20.0, components=("Ez", "Hx"))
        dt = float(config.time_step_duration)
        factor = target_factor(objects["mon"], exact=False, angular_frequencies=_OMEGAS, dt=dt)
        magnetic = factor[:, 1]
        np.testing.assert_allclose(factor[:, 0], 1.0)
        np.testing.assert_allclose(magnetic, -np.exp(-1j * np.asarray(_OMEGAS) * dt))
        assert np.all(magnetic.real < 0), "the magnetic factor must carry the reciprocity sign"
        assert not np.allclose(magnetic, -1.0), "the half-step phase must not be dropped"


def _box_scene(monitor, sim_fs=150.0):
    """The 24^3 scene with ``monitor`` (``(detector, lower_corner)``) as the objective, named ``"mon"``."""
    scene = _Scene(_config(sim_fs))
    scene.at(_device("design", (_DES_SPAN,) * 3), (_DES_LO,) * 3)
    scene.at(_narrowband_dipole("src"), _SRC)
    scene.at(*monitor)
    return scene.place()


class TestNearToFar:
    """Box-mode near-to-far projection. The gradient's boundary sits at the detector's raw per-face
    phasors, so ``FieldProjectionAngleDetector.project`` runs above it as ordinary JAX; the box expands
    into one adjoint current per included face, all driven in one adjoint solve."""

    _THETA = jnp.linspace(0.0, 0.5, 4)
    _PHI = jnp.linspace(0.0, 1.0, 3)

    def _scene(self, sim_fs=150.0, exact_interpolation=False):
        ff = fdtdx.FieldProjectionAngleDetector(
            name="mon",
            partial_grid_shape=(_N - 2 * (_PML + 2),) * 3,
            wave_characters=[fdtdx.WaveCharacter(wavelength=w) for w in _WL],
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
        return _box_scene((ff, (_PML + 2,) * 3), sim_fs)

    def _fom(self, objs, states):
        return -jnp.sum(jnp.abs(objs["mon"].project(states["mon"], self._THETA, self._PHI)["power"]))

    def test_box_expands_into_one_source_per_face(self):
        objects, arrays, _, config, _ = self._scene(sim_fs=20.0)
        faces = objects["mon"]._included_box_surfaces()
        assert objects["mon"]._projection_mode == "box" and "z-" not in faces and len(faces) == 5
        solve = _adjoint_solve(objects, arrays, config, "mon")
        assert len(solve.objects.sources) == len(faces)
        for src in solve.objects.sources:
            thickness = [hi - lo for lo, hi in src.grid_slice_tuple]
            assert sorted(thickness)[0] == 1, f"{src.name} is not a face slab: {thickness}"

    @pytest.mark.parametrize("exact_interpolation", [False, True], ids=["raw_fields", "exact_interpolation"])
    def test_projected_far_field_matches_checkpointed(self, exact_interpolation):
        """``exact_interpolation=True`` is the detector's forced stock setting: each face's co-location stencil
        is transposed, and its adjoint current covers the stencil's support."""
        objects, arrays, params, config, _ = self._scene(exact_interpolation=exact_interpolation)
        _assert_parity(objects, arrays, params, config, self._fom)

    def test_exact_faces_get_the_stencil_support(self):
        objects, arrays, _, config, _ = self._scene(sim_fs=20.0, exact_interpolation=True)
        solve = _adjoint_solve(objects, arrays, config, "mon")
        (sx, ex), (sy, ey), (_sz, ez) = objects["mon"].grid_slice_tuple
        top = next(s for s in solve.objects.sources if s.name.endswith("phasor_z_plus"))
        # z+ face at z = ez - 1: x, y reach one cell below, z one cell above
        assert top.grid_slice_tuple == ((sx - 1, ex), (sy - 1, ey), (ez - 1, ez + 1))


class TestFluxAndEnergyObjectives:
    """Flux, net power through a closed box, and stored energy: whatever a detector computes from its
    phasors is ordinary JAX above the gradient's boundary, so its own readout differentiates itself. A
    closed box is six state keys, six adjoint currents, one solve."""

    _BOX_LO, _BOX_SPAN = _PML + 3, _N - 2 * (_PML + 3)

    def _scene(self, kind, sim_fs=120.0):
        wcs = [fdtdx.WaveCharacter(wavelength=w) for w in _WL]
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
            lower = (_N - _PML - 3, self._BOX_LO, self._BOX_LO)
        elif kind == "box_net_flux":
            mon = fdtdx.ClosedSurfacePhasorPoyntingFluxDetector(
                name="mon",
                partial_grid_shape=(self._BOX_SPAN,) * 3,
                wave_characters=wcs,
                scaling_mode="pulse",
                dft_subsample=1,
                dtype=jnp.complex128,
            )
            lower = (self._BOX_LO,) * 3
        else:
            mon = fdtdx.PhasorDetector(
                name="mon",
                partial_grid_shape=(self._BOX_SPAN,) * 3,
                wave_characters=wcs,
                components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                scaling_mode="pulse",
                **_POINT,
            )
            lower = (self._BOX_LO,) * 3
        if getattr(mon, "exact_interpolation", False):
            mon = mon.aset("exact_interpolation", False)
        return _box_scene((mon, lower), sim_fs)

    @staticmethod
    def _fom(kind):
        if kind == "planar_flux":
            return lambda objs, states: -jnp.sum(objs["mon"].compute_poynting_flux(states["mon"]))
        if kind == "box_net_flux":
            return lambda objs, states: -jnp.sum(objs["mon"].compute_net_flux(states["mon"]))

        def energy(objs, states):
            p = states["mon"]["phasor"][0]
            return -(jnp.sum(jnp.abs(p[:, :3]) ** 2) + jnp.sum(jnp.abs(p[:, 3:]) ** 2))

        return energy

    @pytest.mark.parametrize(
        "kind",
        ["planar_flux", "box_net_flux", "energy"],
        ids=["planar_poynting_flux", "closed_box_net_power", "stored_energy"],
    )
    def test_matches_checkpointed(self, kind):
        objects, arrays, params, config, _ = self._scene(kind)
        _assert_parity(objects, arrays, params, config, self._fom(kind))

    def test_closed_box_uses_one_adjoint_source_per_face(self):
        objects, arrays, _, config, _ = self._scene("box_net_flux", sim_fs=20.0)
        assert len(objects["mon"]._shape_dtype_single_time_step()) == 6
        solve = _adjoint_solve(objects, arrays, config, "mon")
        assert len(solve.objects.sources) == 6
        for src in solve.objects.sources:
            assert sorted(hi - lo for lo, hi in src.grid_slice_tuple)[0] == 1, f"{src.name} is not a face slab"


class TestScalingModes:
    """The monitor's scale (``pulse``, or ``continuous``'s ``2 / sum(window)``) is divided out per detector.
    Every assertion is on relative L2 and best-fit scale: the scaling errors this guards against all had
    cosine 1.0000000000."""

    @pytest.mark.parametrize("mon_scaling", ["pulse", "continuous"])
    def test_either_monitor_scaling_mode(self, mon_scaling):
        _assert_device_parity(mon_scaling=mon_scaling)

    def test_monitors_in_different_modes_each_get_their_own_scale(self):
        """No global factor fits both: before the per-detector scale, rel 1.0 at cosine 0.99999994."""

        def fom(objs, s):
            return -jnp.sum(jnp.abs(s["mon"]["phasor"]) ** 2) + 0.5 * jnp.sum(jnp.real(s["mon2"]["phasor"]) ** 2)

        _assert_device_parity(fom, mon_scaling="pulse", mon2_scaling="continuous")

    @pytest.mark.integration
    def test_design_scale_is_divided_out_twice(self, monkeypatch):
        """The 1/(s_d_fwd * s_d_adj) path, dead while the internal design detector is pulse-scaled: with a
        continuous one, dividing it out once would leave a factor 1/s_d = 787 here."""
        from fdtdx.adjoint import design

        monkeypatch.setitem(design.DESIGN_DETECTOR_SETTINGS, "scaling_mode", "continuous")
        objects, arrays, params, config, _ = _scene(mon_scaling="continuous")
        probe = design.make_design_detector(objects.devices[0], objects["mon"].wave_characters, config, _KEY)
        assert float(probe._static_scale()) < 1e-2, "the design scale path is not being exercised"
        _assert_parity(objects, arrays, params, config)


class TestMonitorsAtDifferentFrequencies:
    """Objective monitors recording different frequencies: one adjoint solve over the union, each monitor's
    cotangent at its own rows. ``lead`` (stock settings, same cell as ``mon``) at 650 nm, or at 650 and 600
    nm, the second shared with ``mon`` and listed in the other order."""

    @staticmethod
    def _fom(objs, s):
        return _FOM(s["mon"]["phasor"]) - 3.0 * jnp.sum(jnp.abs(s["lead"]["phasor"]) ** 2)

    @pytest.mark.parametrize("lead", [(650e-9,), (650e-9, 600e-9)])
    def test_matches_checkpointed(self, lead):
        _assert_device_parity(self._fom, lead_wavelengths=lead)


class TestObjectiveCheck:
    """The objective phasors' own truncation, over the channels and frequencies the figure of merit reads
    with a share of its first-order change above 1e-6."""

    def test_a_weakly_read_channel_does_not_refuse(self):
        """A read monitor at 1200 nm carrying 1e-17 of the figure of merit, unconverged: an unweighted maximum
        refused a figure of merit converged to 1e-7. Compared with checkpointed at 100 fs (at 40 fs
        checkpointed itself is 8e-4 off)."""

        def fom(objs, s):
            return _FOM(s["mon"]["phasor"]) + _FOM(s["lead"]["phasor"])

        objects, arrays, params, config, _ = _scene(sim_fs=40.0, lead_wavelengths=(1200e-9,))
        long = _scene(sim_fs=100.0, lead_wavelengths=(1200e-9,))
        _, g_ck = _value_and_grad(long[1], long[0], params, long[3], "checkpointed", fom)
        _, g_rc = _value_and_grad(arrays, objects, params, config, "reciprocity", fom)
        rel, _, _ = _parity_metrics(g_ck, g_rc)
        assert rel < 1e-4, f"rel_L2 = {rel:.3e}"

    def test_a_penalty_near_its_target_is_refused(self):
        """A read row carrying 2% of the figure of merit's first-order change, 2% unconverged (a penalty term
        near its target): averaged by share it passed with the gradient 11% off."""
        objects, arrays, params, config, _ = _scene(sim_fs=40.0, wavelengths=(600e-9, 800e-9, 900e-9))
        params = jax.tree_util.tree_map(jnp.ones_like, params)  # eps 2.25 throughout the Device, as measured
        _, state = _loss(arrays, objects, config, "reciprocity")(params)
        target = 1.001 * float(jnp.sum(jnp.abs(state[1].detector_states["mon"]["phasor"][0, 2]) ** 2))

        def fom(objs, s):
            p = s["mon"]["phasor"]
            return -jnp.sum(jnp.abs(p[0, 0]) ** 2) + (jnp.sum(jnp.abs(p[0, 2]) ** 2) / target - 1.0) ** 2

        with pytest.raises(Exception, match="objective monitors"):
            jax.block_until_ready(_value_and_grad(arrays, objects, params, config, "reciprocity", fom))

    def test_an_unread_frequency_counts_nothing(self):
        """A monitor frequency the figure of merit does not read: its adjoint field is the solve's noise, never
        exactly 0, whose tail refused gradients exact to 3e-6."""

        def fom(objs, s):
            return -jnp.sum(jnp.abs(s["mon"]["phasor"][0, 0]) ** 2)

        _assert_device_parity(fom, wavelengths=(600e-9, 540e-9))

    def test_a_weak_row_folded_by_a_stride_is_refused(self):
        """A strided row at 1.5e-5 of the source peak, folded at 6e-2 of its own content: compared with the
        strongest row instead of its own, it passed with the gradient 11% off."""
        weak = float(c0 / 7.35235e14)
        objects, arrays, params, config, _ = _scene(
            wavelengths=(600e-9, weak),
            monitor_kwargs=dict(_POINT, components=("Ez",), scaling_mode="pulse", dft_subsample=7),
        )

        def read(row):
            return lambda objs, s: -jnp.sum(jnp.abs(s["mon"]["phasor"][0, row]) ** 2)

        with pytest.raises(Exception, match="dft_subsample"):
            jax.block_until_ready(_value_and_grad(arrays, objects, params, config, "reciprocity", read(1)))
        # the figure of merit reading the clean row only is not refused (exact to 1.9e-9)
        jax.block_until_ready(_value_and_grad(arrays, objects, params, config, "reciprocity", read(0)))


class TestConvergence:
    """The gradient is refused, on every call, when its estimated truncation error exceeds tail_tolerance.
    Measured in this scene (CPU, float64): 40 fs true rel 4.6e-07, estimate 1.1e-07."""

    _RINGING = fdtdx.Material(permittivity=12.0)

    def test_a_pulse_the_run_does_not_outlast_is_refused(self):
        # 0.1 f0 at 600 nm peaks at 19 fs and lasts to about 35 fs; refused when the gradient is traced
        objects, arrays, params, config, _ = _scene(sim_fs=25.0)
        loss = _loss(arrays, objects, config, "reciprocity")
        with pytest.raises(NotImplementedError, match="still injects"):
            jax.make_jaxpr(jax.grad(lambda p: loss(p)[0]))(params)

    @pytest.mark.integration
    def test_ringing_is_refused_and_really_wrong(self):
        objects, arrays, params, config, _ = _scene(sim_fs=60.0, device_material=self._RINGING)
        with pytest.raises(Exception, match="reciprocity gradient is refused"):
            jax.block_until_ready(_value_and_grad(arrays, objects, params, config, "reciprocity"))
        _, g_rc = _value_and_grad(arrays, objects, params, config, "reciprocity", tail_tolerance=None)
        _, g_ck = _value_and_grad(arrays, objects, params, config, "checkpointed")
        rel, cos, _ = _parity_metrics(g_ck, g_rc)
        assert rel > 1e-2, f"the 60 fs gradient was expected to be off, rel {rel:.3e} cos {cos:.6f}"


_LOSSY = fdtdx.Material(permittivity=2.25, electric_conductivity=1e5)
_ANISO = fdtdx.Material(permittivity=(2.0, 3.0, 4.0))
_LORENTZ = fdtdx.Material(
    permittivity=2.0,
    dispersion=fdtdx.DispersionModel(
        poles=(fdtdx.LorentzPole(resonance_frequency=1.3 * _OMEGAS[0], damping=0.1 * _OMEGAS[0], delta_epsilon=0.5),)
    ),
)
# A block around the monitor cell (18, 12, 12): x 17..19, y 10..14, z 10..14, clear of the Device
# (8..15), the source (5, 12, 12) and the PML (from 20).
_AROUND_MON = ((17, 10, 10), (3, 5, 5))
# The same block from y = 12 up, so it covers two of the three cells of a y-line monitor at (18, 11..13, 12).
_HALF_MON = ((17, 12, 10), (3, 3, 5))


class TestMaterials:
    """Loss, anisotropy, permeability and dispersion around the design and the monitors.

    Each catches an error every lossless isotropic scene passes, all silent (measured): FDTDX adds
    every source after the lossy division by ``1 + a``, so the adjoint current in a lossy monitor cell
    was ``1 + a`` too strong (rel 2.4e-01 at cosine 1.0, best-fit scale 1 + a exactly); the kernel
    summing the component axis of a three-row ``inv_eps``, a pure scale 3.0; the injection factor read
    from row 0, rel 2.0e-01; the H current ignoring ``inv_mu``, a pure scale 2.0. ``apply_params``
    writes no conductivity into a Device, as upstream, so the Device cells keep the loss placed there.
    """

    @pytest.mark.parametrize(
        "scene_kwargs",
        [
            dict(extra_blocks=[("loss", (_DES_LO,) * 3, (_DES_SPAN,) * 3, _LOSSY)]),
            dict(extra_blocks=[("loss", *_AROUND_MON, _LOSSY)]),
            dict(
                extra_blocks=[("mloss", *_AROUND_MON, fdtdx.Material(magnetic_conductivity=6e9))],
                components=("Hx",),
            ),
            pytest.param(
                dict(
                    extra_blocks=[
                        (
                            "loss",
                            *_HALF_MON,
                            fdtdx.Material(permittivity=2.25, electric_conductivity=1e5, magnetic_conductivity=6e9),
                        )
                    ],
                    monitor_kwargs=dict(components=("Ez", "Hx"), dtype=jnp.complex128),
                    monitor_shape=(1, 3, 1),
                    monitor_cell=(_MON[0], _MON[1] - 1, _MON[2]),
                ),
                marks=pytest.mark.integration,
            ),
            dict(monitor_cell=(_N // 2,) * 3),
            dict(extra_blocks=[("aniso", (_DES_LO,) * 3, (_DES_SPAN,) * 3, _ANISO)], components=("Ex", "Ey", "Ez")),
            dict(extra_blocks=[("aniso", *_AROUND_MON, _ANISO)], monitor_kwargs={}, monitor_shape=(1, 2, 1)),
            dict(
                extra_blocks=[
                    (
                        "aniso",
                        *_AROUND_MON,
                        fdtdx.Material(permittivity=(2.0, 3.0, 4.0), electric_conductivity=(1e5, 2e5, 3e5)),
                    )
                ],
                components=("Ex", "Ey", "Ez"),
            ),
            dict(device_material=fdtdx.Material(permittivity=(2.0, 3.0, 4.0)), components=("Ex", "Ey", "Ez")),
            dict(
                extra_blocks=[("mu", *_AROUND_MON, fdtdx.Material(permittivity=1.5, permeability=2.0))],
                components=("Hx",),
            ),
            dict(
                extra_blocks=[
                    (
                        "mu",
                        *_AROUND_MON,
                        fdtdx.Material(permeability=(1.0, 2.0, 3.0), magnetic_conductivity=(2e9, 4e9, 6e9)),
                    )
                ],
                components=("Hx", "Hy"),
            ),
            dict(extra_blocks=[("lorentz", (_DES_LO + _DES_SPAN, 10, 10), (2, 5, 5), _LORENTZ)]),
            pytest.param(
                dict(extra_blocks=[("lorentz", (_DES_LO,) * 3, (_DES_SPAN,) * 3, _LORENTZ)]),
                marks=pytest.mark.integration,
            ),
            dict(
                etched=True,
                extra_blocks=[
                    (
                        "core",
                        (_DES_LO,) * 3,
                        (_DES_SPAN,) * 3,
                        fdtdx.Material(permittivity=4.0, electric_conductivity=1e5),
                    )
                ],
            ),
        ],
        ids=[
            "lossy_design_cells",
            "electric_loss_around_the_monitor",
            "magnetic_loss_around_the_monitor",
            "partial_loss_under_a_stock_monitor",
            "monitor_inside_the_device",
            "eps_under_the_device",
            "eps_at_a_stock_monitor",
            "sigma_at_a_raw_monitor",
            "anisotropic_device_material",
            "mu_at_an_hx_monitor",
            "anisotropic_mu_and_sigma_h_at_an_h_monitor",
            "lorentz_block_on_the_path",
            "device_over_a_lorentz_block",
            "air_etched_into_a_lossy_core",
        ],
    )
    def test_matches_checkpointed(self, scene_kwargs):
        _assert_device_parity(**scene_kwargs)


def _param_parity(objects, arrays, params, config, fom):
    same, rel, cos, scale, _ = _parity(objects, arrays, params, config, fom)
    assert same, "the forward value must be run_fdtd's, bit for bit"
    return rel, cos, scale


def _periodic_mode_scene(sim_fs):
    """The neural-to-coverage problems' layout: 2D x-z, periodic y of two cells, PML on x and z."""
    wl0, wls, nx, ny, nz, pml = 1.2e-6, (1.15e-6, 1.25e-6), 48, 2, 32, 8
    config = _config(sim_fs)
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


def _box_far_field_scene(sim_fs=150.0, pml=_PML):
    """The colour splitter's layout: plane wave down onto a Device on a substrate, box far field around it."""
    wl0 = 600e-9

    def wc(w):
        return fdtdx.WaveCharacter(wavelength=w)

    scene = _Scene(_config(sim_fs), shape=(24, 24, 30), pml=pml)
    scene.at(
        fdtdx.UniformMaterialObject(
            name="substrate", partial_grid_shape=(24, 24, 10), material=fdtdx.Material(permittivity=2.1)
        ),
        (0, 0, 0),
    )
    scene.at(_device("design", (8, 8, 4)), (8, 8, 11))
    pulse = fdtdx.GaussianPulseProfile(
        center_wave=wc(wl0), spectral_width=fdtdx.WaveCharacter(frequency=0.3 * float(c0 / wl0))
    )
    scene.at(
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
    scene.at(
        fdtdx.FieldProjectionAngleDetector(
            name="ff",
            partial_grid_shape=(12, 12, 10),
            wave_characters=(wc(550e-9), wc(650e-9)),
            exclude_surfaces=("z-",),
        ),
        (6, 6, 10),
    )
    objects, arrays, params, config, _ = scene.place()
    return objects, arrays, _varied(params), config


_THETA = jnp.asarray([0.0, 0.3, 0.6, 2.6, 3.0])
_PHI = jnp.asarray([0.0, 0.8, 1.6, 2.4, 3.1])


def _far_field_power(objs, states):
    return -jnp.sum(objs["ff"].project_all(states["ff"], _THETA, _PHI)["power"])


class TestStockObjectives:
    """Objective monitors at FDTDX's stock settings, as the real problems use them: their plane sources
    and mode ports share the Device's footprint, so ``apply_params`` applies them on every call."""

    @pytest.mark.integration
    def test_stock_monitor(self):
        """Six components, ``exact_interpolation=True``, continuous, complex64: the co-location stencil's
        transpose and the magnetic time average are both exercised. GPU, float64: rel 5.4e-07."""
        objects, arrays, params, config, _ = _scene(monitor_kwargs={})
        mon = objects["mon"]
        assert mon.exact_interpolation and len(mon.components) == 6 and mon.scaling_mode == "continuous"
        _assert_parity(objects, arrays, params, config)

    def test_stencil_transpose_and_time_average_are_both_needed(self, monkeypatch):
        """An Hx monitor recorded as if raw: rel 2.0e-01 at cosine 0.980 (with them 8.3e-07)."""
        from fdtdx.adjoint import objective, solve

        objects, arrays, params, config, _ = _scene(
            monitor_kwargs=dict(components=("Hx",), scaling_mode="pulse", dtype=jnp.complex128)
        )
        _, g_ck = _value_and_grad(arrays, objects, params, config, "checkpointed")

        def rel_now():
            _, g_rc = _value_and_grad(arrays, objects, params, config, "reciprocity")
            return _parity_metrics(g_ck, g_rc)[0]

        assert rel_now() < 1e-5
        honest = objective.channel_recordings

        def as_raw(detector, *args, **kwargs):
            return honest(detector.aset("exact_interpolation", False), *args, **kwargs)

        monkeypatch.setattr(solve, "channel_recordings", as_raw)
        assert rel_now() > 1e-2

    def test_periodic_mode_port(self):
        """2D x-z, two periodic y cells, ModePlaneSource, ModeOverlapDetector at stock settings; the figure of
        merit is the mode power through FDTDX's own ``compute_overlap``. The port spans the periodic axis, so
        its stencil takes FDTDX's padded whole-domain path and wraps. CPU, float64: rel 2.3e-08."""
        objects, arrays, params, config = _periodic_mode_scene(sim_fs=150.0)
        port = objects["out"]
        assert port.exact_interpolation and port.grid_slice_tuple[1] == (0, 2)

        def power(objs, states):
            return -jnp.sum(jnp.abs(objs["out"].compute_overlap(states["out"])) ** 2)

        rel, cos, scale = _param_parity(objects, arrays, params, config, power)
        assert rel < 1e-6, f"mode port: rel_L2 = {rel:.3e} (scale {scale:.6f})"
        assert cos > 1 - 1e-10, f"cosine {cos:.10f}"

    def test_box_far_field(self):
        """FieldProjectionAngleDetector box, stock settings, 5 faces, UniformPlaneSource(normalize_by_energy),
        read through ``project_all``. GPU, float64: rel 6.1e-07."""
        objects, arrays, params, config = _box_far_field_scene()
        rel, cos, scale = _param_parity(objects, arrays, params, config, _far_field_power)
        assert rel < 1e-5 and abs(scale - 1) < 1e-5, f"rel {rel:.3e} cos {cos:.10f} scale {scale:.9f}"

    @pytest.mark.parametrize("plane, polarization", [(-1, (1, 0, 0)), (1, (0, 1, 0))], ids=["pec", "pmc"])
    def test_monitor_on_a_symmetry_plane(self, plane, polarization):
        """``config.symmetry`` puts the monitor's stencil through FDTDX's mirror padding: 24^3 reduced by a
        PEC (E normal to it) or PMC (E along it) plane that the Device and a stock monitor both straddle. GPU,
        float64, PEC: rel 4.1e-07, and 3.8e-07 for the same scene without symmetry."""
        wc = fdtdx.WaveCharacter(wavelength=600e-9)
        scene = _Scene(_config(symmetry=(plane, 0, 0)))
        scene.at(_device("design", (8, 8, 4)), (8, 8, 9))
        scene.at(
            fdtdx.UniformPlaneSource(
                name="source",
                partial_grid_shape=(_N, _N, 1),
                direction="+",
                fixed_E_polarization_vector=polarization,
                wave_character=wc,
                temporal_profile=fdtdx.GaussianPulseProfile(
                    center_wave=wc, spectral_width=fdtdx.WaveCharacter(frequency=0.3 * float(c0 / 600e-9))
                ),
                normalize_by_energy=True,
            ),
            (0, 0, 5),
        )
        scene.at(fdtdx.PhasorDetector(name="mon", partial_grid_shape=(4, 4, 1), wave_characters=(wc,)), (10, 10, 17))
        objects, arrays, params, config, _ = scene.place()
        assert objects["mon"].grid_slice_tuple[0][0] == 0, "the monitor must touch the reduced domain's symmetry plane"
        rel, cos, scale = _param_parity(objects, arrays, _varied(params), config, _mon_fom)
        assert rel < 1e-5 and abs(scale - 1) < 1e-5, f"rel {rel:.3e} cos {cos:.10f} scale {scale:.9f}"

    @pytest.mark.parametrize("scaling_mode", [pytest.param("continuous", marks=pytest.mark.integration), "pulse"])
    def test_strided_monitor(self, scaling_mode):
        """``dft_subsample=3``: only the principal term of the strided DFT is transposed; its aliases sit where
        a band-limited source puts no field. GPU, float64: rel 5.5e-07 at strides 2, 3 and 5."""
        objects, arrays, params, config, _ = _scene(monitor_kwargs=dict(dft_subsample=3, scaling_mode=scaling_mode))
        assert objects["mon"]._dft_stride == 3
        _assert_parity(objects, arrays, params, config)


class TestObjectiveInThePml:
    """An objective whose adjoint current reaches the lossy part of a PML is refused (GPU, float64, 150 fs).
    Box faces at cells 6 and 17 of 24. PML 6: the stencil enters the PML's zero-loss first cell only, rel
    5.6e-07. PML 7, one cell deeper: rel 5.4e-04, silently. PML 8: rel 1.0e-02."""

    @pytest.mark.parametrize("pml", [7, 8], ids=["one-cell-deeper", "two-cells-deeper"])
    def test_currents_in_the_lossy_pml_are_refused(self, pml):
        objects, arrays, params, config = _box_far_field_scene(pml=pml)
        loss = _loss(arrays, objects, config, "reciprocity", _far_field_power)
        with pytest.raises(NotImplementedError, match="inside a PML"):
            jax.make_jaxpr(jax.grad(lambda p: loss(p)[0]))(params)

    def test_the_first_pml_cell_is_harmless(self):
        objects, arrays, params, config = _box_far_field_scene(pml=6)
        rel, _, _ = _param_parity(objects, arrays, params, config, _far_field_power)
        assert rel < 1e-5, f"rel {rel:.3e}"


class TestGeometry:
    def test_one_cell_width_per_axis(self):
        """dz = 40 nm against 50 nm in x and y; widths varying along an axis are refused (test_refusals.py)."""
        _assert_device_parity(grid=QuasiUniformGrid(dx=_RES, dy=_RES, dz=40e-9))

    def test_an_adjoint_current_source_may_sit_inside_a_device(self):
        """It reads ``inv_eps`` live, unlike the stock sources, which are refused there."""
        objects, _, _, config, _ = _scene(sim_fs=20.0)
        centre = _DES_LO + _DES_SPAN // 2
        live = AdjointCurrentSource(
            name="live",
            amplitudes=jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex128),
            window=jnp.ones(int(config.time_steps_total)),
            angular_frequencies=_OMEGAS,
            components=("Ez",),
            wave_character=fdtdx.WaveCharacter(wavelength=_WL[0]),
        ).place_on_grid(grid_slice_tuple=((centre, centre + 1),) * 3, config=config, key=_KEY)
        swapped = objects.aset("object_list", [o for o in objects.object_list if o.name != "src"] + [live])
        validation.check_sources_outside(swapped)
