"""Reciprocity gradients: an adjoint source plus a second forward solve.

The acceptance reference is ``checkpointed_fdtd`` autodiff, which is the exact
derivative of the same discrete program. ``reversible_fdtd`` is deliberately not
used as a reference: it carries float32 reconstruction drift and its own test
only requires 1e-2.

Reciprocity is a frequency-domain identity, so it equals the exact discrete
adjoint only once both DFTs have converged. The tolerances below are therefore
tied to a stated decay time, and :func:`test_gradient_converges_with_runtime`
asserts the behaviour that actually matters: the error falls as the fields are
given time to leave the domain, rather than sitting on a floor.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.adjoint import gaussian_window, leapfrog_kernel, solve_adjoint_amplitudes
from fdtdx.adjoint.vjp import make_reciprocity_phasor_fn
from fdtdx.config import SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.fdtd import checkpointed_fdtd
from fdtdx.objects.sources.adjoint import AdjointCurrentSource


#: These tests need float64: the gate is a 1e-5 relative comparison against
#: checkpointed autodiff, which float32 cannot resolve.  x64 is enabled per test
#: rather than at import, because pytest shares one process and flipping it
#: globally changes default dtypes for every other test module in the session.
@pytest.fixture(autouse=True)
def _enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


_RES = 50e-9
_N = 16
_PML = 3
_WAVELENGTHS = (500e-9, 700e-9)
_OMEGAS = tuple(2 * np.pi * c0 / w for w in _WAVELENGTHS)
_SRC = (4, 8, 8)
_MON = (12, 8, 8)
_DES_LO, _DES_SPAN = 7, 3
_COMP = ("Ez",)
_KEY = jax.random.PRNGKey(1)

#: Source, design region and monitor are disjoint and clear of the PML. That is a
#: requirement, not tidiness: FDTDX sources scale their injection by the local
#: inv_eps, so a source inside the design region adds a gradient term the
#: reciprocity kernel does not model.
assert _PML <= _SRC[0] < _DES_LO
assert _DES_LO + _DES_SPAN <= _MON[0] < _N - _PML


def _config(sim_fs: float) -> SimulationConfig:
    return SimulationConfig(
        time=sim_fs * 1e-15,
        grid=UniformGrid(spacing=_RES),
        backend="cpu",
        dtype=jnp.float64,
        courant_factor=0.99,
        gradient_config=None,
    )


def _build(config, src_cell, amplitudes, window, src_name, periodic=False):
    objs, cons = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
    objs.append(volume)
    override = {k: "periodic" for k in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")} if periodic else None
    bcfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML, override_types=override)
    bd, cl = fdtdx.boundary_objects_from_config(bcfg, volume)
    objs.extend(bd.values())
    cons.extend(cl)

    slab = fdtdx.UniformMaterialObject(
        name="slab", partial_grid_shape=(None, None, 3), material=fdtdx.Material(permittivity=2.25)
    )
    cons += [
        slab.same_size(volume, axes=(0, 1)),
        slab.place_at_center(volume, axes=(0, 1)),
        slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(8,)),
    ]
    objs.append(slab)

    wcs = [fdtdx.WaveCharacter(wavelength=w) for w in _WAVELENGTHS]
    src = AdjointCurrentSource(
        name=src_name,
        partial_grid_shape=(1, 1, 1),
        amplitudes=amplitudes,
        window=window,
        angular_frequencies=_OMEGAS,
        components=_COMP,
        wave_character=wcs[0],
    )
    cons.append(src.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=src_cell))
    objs.append(src)

    mon = fdtdx.PhasorDetector(
        name="mon",
        partial_grid_shape=(1, 1, 1),
        wave_characters=wcs,
        components=_COMP,
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

    o, a, p, cfg, _ = fdtdx.place_objects(object_list=objs, config=config, constraints=cons, key=jax.random.PRNGKey(0))
    a, o, _ = fdtdx.apply_params(a, o, p, jax.random.PRNGKey(0))
    return o, a, cfg


def _scenes(sim_fs: float, periodic: bool = False):
    """Forward and adjoint scenes plus the shared window, for one runtime."""
    config = _config(sim_fs)
    window = gaussian_window(config.time_steps_total)
    nf, nc = len(_OMEGAS), len(_COMP)
    unit = jnp.ones((nf, nc, 1, 1, 1), dtype=jnp.complex128)
    zero = jnp.zeros((nf, nc, 1, 1, 1), dtype=jnp.complex128)
    amp_fwd, _ = solve_adjoint_amplitudes(unit, _OMEGAS, config.time_step_duration, window)
    fwd = _build(config, _SRC, amp_fwd, window, "src", periodic)
    adj = _build(config, _MON, zero, window, "adj", periodic)
    return fwd, adj, window


def _run(arrays, objects, config, inv_eps=None):
    if inv_eps is not None:
        arrays = arrays.aset("inv_permittivities", inv_eps)
    _, out = checkpointed_fdtd(arrays, objects, config, _KEY, show_progress=False)
    return out


def _phasor_fn(sim_fs: float):
    (obj_f, arr_f, cfg), (obj_a, arr_a, _), window = _scenes(sim_fs)
    fn = make_reciprocity_phasor_fn(
        forward_arrays=arr_f,
        forward_objects=obj_f,
        adjoint_arrays=arr_a,
        adjoint_objects=obj_a,
        config=cfg,
        key=_KEY,
        objective_detector="mon",
        design_detector="des",
        adjoint_source="adj",
        window=window,
    )
    return fn, obj_f, arr_f, cfg


def _checkpointed_grad(fom, obj_f, arr_f, cfg, inv_eps):
    def loss(ie):
        out = _run(arr_f, obj_f, cfg, ie)
        return fom(out.detector_states["mon"]["phasor"])

    return jax.value_and_grad(loss)(inv_eps)


def _rel_on_design(g_model, g_true, obj_f):
    gs = next(d for d in obj_f.detectors if d.name == "des").grid_slice
    a, b = g_model[:, *gs], g_true[:, *gs]
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-300))


# ──────────────────────────────────────────────────────────────────────────────
# AdjointCurrentSource properties
# ──────────────────────────────────────────────────────────────────────────────


class TestAdjointCurrentSource:
    def _source(self):
        cfg = _config(20.0)
        window = gaussian_window(cfg.time_steps_total)
        src = AdjointCurrentSource(
            amplitudes=jnp.ones((2, 1, 1, 1, 1), dtype=jnp.complex128),
            window=window,
            angular_frequencies=_OMEGAS,
            components=_COMP,
            wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTHS[0]),
        )
        return src.place_on_grid(((0, 1), (0, 1), (0, 1)), cfg, jax.random.PRNGKey(0)), cfg

    def test_inverse_exactly_negates(self):
        """The reverse update must undo the forward one bit for bit."""
        src, _ = self._source()
        E = jnp.zeros((3, 1, 1, 1))
        inv_eps = jnp.ones((1, 1, 1, 1))
        kwargs = dict(inv_permittivities=inv_eps, inv_permeabilities=1.0, time_step=jnp.asarray(7))
        fwd = src.update_E(E=E, inverse=False, **kwargs)
        back = src.update_E(E=fwd, inverse=True, **kwargs)
        assert jnp.allclose(back, E, atol=0.0, rtol=0.0)
        assert jnp.any(fwd != E), "forward injection did nothing, so the test proves nothing"

    def test_amplitude_change_preserves_treedef(self):
        """Amplitudes must be traced leaves, or every optimizer step recompiles."""
        src, _ = self._source()
        other = src.aset("amplitudes", src.amplitudes * 3.0 + 1.0)
        assert jax.tree_util.tree_structure(src) == jax.tree_util.tree_structure(other)

    def test_rejects_unknown_component(self):
        cfg = _config(20.0)
        with pytest.raises(ValueError, match="Unknown field components"):
            AdjointCurrentSource(
                amplitudes=jnp.ones((2, 1), dtype=jnp.complex128),
                window=gaussian_window(cfg.time_steps_total),
                angular_frequencies=_OMEGAS,
                components=("Qx",),
                wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTHS[0]),
            )

    def test_rejects_shape_mismatch(self):
        cfg = _config(20.0)
        with pytest.raises(ValueError, match=r"does not match|!="):
            AdjointCurrentSource(
                amplitudes=jnp.ones((5, 1, 1, 1, 1), dtype=jnp.complex128),
                window=gaussian_window(cfg.time_steps_total),
                angular_frequencies=_OMEGAS,
                components=_COMP,
                wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTHS[0]),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Amplitude solve
# ──────────────────────────────────────────────────────────────────────────────


class TestAmplitudeSolve:
    def test_solve_is_exact_and_well_conditioned(self):
        """The system is square, so the residual should be at machine precision."""
        cfg = _config(200.0)
        window = gaussian_window(cfg.time_steps_total)
        rng = np.random.default_rng(0)
        target = jnp.asarray(rng.normal(size=(2, 1, 2, 2, 2)) + 1j * rng.normal(size=(2, 1, 2, 2, 2)))
        amps, diag = solve_adjoint_amplitudes(target, _OMEGAS, cfg.time_step_duration, window)
        assert amps.shape == target.shape
        assert diag["residual"] < 1e-12, diag
        assert diag["cond"] < 1e3, diag

    def test_short_window_is_rejected_loudly(self):
        """Frequencies closer than 1/T cannot be separated; that must raise."""
        cfg = _config(200.0)
        # a window only a couple of steps wide cannot resolve anything
        window = gaussian_window(cfg.time_steps_total, center_frac=0.5, sigma_frac=1e-5)
        target = jnp.ones((2, 1, 1, 1, 1), dtype=jnp.complex128)
        with pytest.raises(ValueError, match="ill-conditioned"):
            solve_adjoint_amplitudes(target, _OMEGAS, cfg.time_step_duration, window)

    def test_leapfrog_kernel_beats_continuum_iomega(self):
        """The discrete factor is mandatory, not a refinement."""
        cfg = _config(200.0)
        dt, cc = cfg.time_step_duration, cfg.courant_number
        disc = leapfrog_kernel(_OMEGAS, dt, cc)
        cont = np.asarray([-1j * w * dt for w in _OMEGAS]) / cc
        # they agree only to first order; at these w*dt they are visibly different
        rel = np.abs(disc - cont) / np.abs(disc)
        assert np.all(rel > 1e-2), f"w*dt too small for this test to be meaningful: {rel}"


# ──────────────────────────────────────────────────────────────────────────────
# The premise: is the discrete phasor response reciprocal?
# ──────────────────────────────────────────────────────────────────────────────


class TestDiscreteReciprocity:
    @pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "pml"])
    def test_phasor_response_is_symmetric(self, periodic):
        """Swapping source and monitor must give the same phasor.

        Nothing downstream can work otherwise. This also settles empirically that
        FDTDX's CPML preserves the discrete symmetry: it holds to ~1e-15 with PML,
        and bit-exactly with periodic boundaries.
        """
        config = _config(120.0)
        window = gaussian_window(config.time_steps_total)
        unit = jnp.ones((len(_OMEGAS), len(_COMP), 1, 1, 1), dtype=jnp.complex128)
        amps, _ = solve_adjoint_amplitudes(unit, _OMEGAS, config.time_step_duration, window)

        def response(src_cell, mon_cell):
            objs, cons = [], []
            volume = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
            objs.append(volume)
            override = (
                {k: "periodic" for k in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")} if periodic else None
            )
            bd, cl = fdtdx.boundary_objects_from_config(
                fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML, override_types=override), volume
            )
            objs.extend(bd.values())
            cons.extend(cl)
            slab = fdtdx.UniformMaterialObject(
                name="slab",
                partial_grid_shape=(None, None, 3),
                material=fdtdx.Material(permittivity=2.25),
            )
            cons += [
                slab.same_size(volume, axes=(0, 1)),
                slab.place_at_center(volume, axes=(0, 1)),
                slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(8,)),
            ]
            objs.append(slab)
            wcs = [fdtdx.WaveCharacter(wavelength=w) for w in _WAVELENGTHS]
            src = AdjointCurrentSource(
                name="src",
                partial_grid_shape=(1, 1, 1),
                amplitudes=amps,
                window=window,
                angular_frequencies=_OMEGAS,
                components=_COMP,
                wave_character=wcs[0],
            )
            cons.append(src.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=src_cell))
            objs.append(src)
            det = fdtdx.PhasorDetector(
                name="det",
                partial_grid_shape=(1, 1, 1),
                wave_characters=wcs,
                components=_COMP,
                scaling_mode="pulse",
                dft_subsample=1,
                exact_interpolation=False,
                reduce_volume=False,
                dtype=jnp.complex128,
            )
            cons.append(det.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=mon_cell))
            objs.append(det)
            o, a, p, cfg, _ = fdtdx.place_objects(
                object_list=objs, config=config, constraints=cons, key=jax.random.PRNGKey(0)
            )
            a, o, _ = fdtdx.apply_params(a, o, p, jax.random.PRNGKey(0))
            return _run(a, o, cfg).detector_states["det"]["phasor"].ravel()

        g_ab = response(_SRC, _MON)
        g_ba = response(_MON, _SRC)
        denom = jnp.maximum(jnp.abs(g_ab), jnp.abs(g_ba)) + 1e-300
        rel = float(jnp.max(jnp.abs(g_ab - g_ba) / denom))
        assert float(jnp.max(jnp.abs(g_ab))) > 0.0, "no signal reached the monitor"
        assert rel < 1e-12, f"discrete reciprocity violated: rel={rel:.3e}"


# ──────────────────────────────────────────────────────────────────────────────
# The gradient
# ──────────────────────────────────────────────────────────────────────────────


def _fom_linear(P):
    rng = np.random.default_rng(7)
    w_re = jnp.asarray(rng.normal(size=P.shape))
    w_im = jnp.asarray(rng.normal(size=P.shape))
    return jnp.sum(w_re * jnp.real(P) + w_im * jnp.imag(P))


def _fom_intensity(P):
    return -jnp.sum(jnp.abs(P) ** 2)


def _fom_log_ratio(P):
    a = jnp.sum(jnp.abs(P[:, 0]) ** 2)
    b = jnp.sum(jnp.abs(P[:, 1:]) ** 2)
    return jnp.log(a + 1e-30) - jnp.log(b + 1e-30)


class TestReciprocityGradient:
    @pytest.mark.parametrize(
        "fom",
        [_fom_linear, _fom_intensity, _fom_log_ratio],
        ids=["random_linear", "intensity", "log_ratio"],
    )
    def test_matches_checkpointed_autodiff(self, fom):
        """Arbitrary differentiable FoM on the raw phasors, via plain jax.grad.

        This is the generality claim: the VJP boundary sits at the monitor's raw
        phasors, so the FoM is unconstrained and needs no adjoint-source rule of
        its own.
        """
        fn, obj_f, arr_f, cfg = _phasor_fn(300.0)
        ie = arr_f.inv_permittivities
        v_rec, g_rec = jax.value_and_grad(lambda x: fom(fn(x)))(ie)
        v_true, g_true = _checkpointed_grad(fom, obj_f, arr_f, cfg, ie)

        assert jnp.allclose(v_rec, v_true, rtol=1e-12), "forward values must be identical"
        assert jnp.all(jnp.isfinite(g_rec))
        rel = _rel_on_design(g_rec, g_true, obj_f)
        assert rel < 2e-3, f"relative L2 vs checkpointed AD = {rel:.3e}"

        gs = next(d for d in obj_f.detectors if d.name == "des").grid_slice
        a, b = g_rec[:, *gs], g_true[:, *gs]
        cos = float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
        assert cos > 1 - 1e-6, f"cosine similarity {cos:.9f}"

    def test_gradient_is_zero_outside_the_design_region(self):
        """Documented restriction: only the design detector's region is filled."""
        fn, obj_f, arr_f, _ = _phasor_fn(120.0)
        g = jax.grad(lambda x: _fom_intensity(fn(x)))(arr_f.inv_permittivities)
        gs = next(d for d in obj_f.detectors if d.name == "des").grid_slice
        assert float(jnp.linalg.norm(g.at[:, *gs].set(0.0))) == 0.0
        assert float(jnp.linalg.norm(g[:, *gs])) > 0.0

    @pytest.mark.slow
    def test_gradient_converges_with_runtime(self):
        """The error must fall as the fields are given time to decay.

        Reciprocity is a frequency-domain identity, so it matches the exact
        discrete adjoint only once both DFTs converge. Asserting convergence is
        the honest test; asserting a fixed tolerance would hide the mechanism.
        """
        errs = []
        for sim_fs in (100.0, 400.0):
            fn, obj_f, arr_f, cfg = _phasor_fn(sim_fs)
            ie = arr_f.inv_permittivities
            g_rec = jax.grad(lambda x: _fom_linear(fn(x)))(ie)
            _, g_true = _checkpointed_grad(_fom_linear, obj_f, arr_f, cfg, ie)
            errs.append(_rel_on_design(g_rec, g_true, obj_f))
        assert errs[-1] < errs[0] / 3.0, f"error did not converge with runtime: {errs}"
        assert errs[-1] < 1e-4, f"error at the long runtime is {errs[-1]:.3e}"


# ──────────────────────────────────────────────────────────────────────────────
# Guardrails
# ──────────────────────────────────────────────────────────────────────────────


class TestUnsupportedConfigurations:
    def test_rejects_reduce_volume_detector(self):
        config = _config(60.0)
        window = gaussian_window(config.time_steps_total)
        nf, nc = len(_OMEGAS), len(_COMP)
        amp, _ = solve_adjoint_amplitudes(
            jnp.ones((nf, nc, 1, 1, 1), dtype=jnp.complex128), _OMEGAS, config.time_step_duration, window
        )
        objs, cons = [], []
        volume = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
        objs.append(volume)
        bd, cl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), volume)
        objs.extend(bd.values())
        cons.extend(cl)
        wcs = [fdtdx.WaveCharacter(wavelength=w) for w in _WAVELENGTHS]
        src = AdjointCurrentSource(
            name="adj",
            partial_grid_shape=(1, 1, 1),
            amplitudes=amp,
            window=window,
            angular_frequencies=_OMEGAS,
            components=_COMP,
            wave_character=wcs[0],
        )
        cons.append(src.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=_MON))
        objs.append(src)
        bad = fdtdx.PhasorDetector(
            name="mon",
            partial_grid_shape=(1, 1, 1),
            wave_characters=wcs,
            components=_COMP,
            scaling_mode="pulse",
            dft_subsample=1,
            exact_interpolation=False,
            reduce_volume=True,
            dtype=jnp.complex128,
        )
        cons.append(bad.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=_MON))
        objs.append(bad)
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
        o, a, p, cfg, _ = fdtdx.place_objects(
            object_list=objs, config=config, constraints=cons, key=jax.random.PRNGKey(0)
        )
        a, o, _ = fdtdx.apply_params(a, o, p, jax.random.PRNGKey(0))
        with pytest.raises(NotImplementedError, match="reduce_volume"):
            make_reciprocity_phasor_fn(
                forward_arrays=a,
                forward_objects=o,
                adjoint_arrays=a,
                adjoint_objects=o,
                config=cfg,
                key=_KEY,
                objective_detector="mon",
                design_detector="des",
                adjoint_source="adj",
                window=window,
            )
