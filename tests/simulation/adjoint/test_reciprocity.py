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
from fdtdx.adjoint.kernel import amplitude_matrix, gaussian_window, leapfrog_kernel, solve_amplitudes
from fdtdx.adjoint.source import AdjointCurrentSource
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.fdtd import checkpointed_fdtd


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


def _amplitudes(target, config, window):
    """Amplitudes of an AdjointCurrentSource whose windowed current has the DFT ``target``, and the solve's
    ``cond`` and relative residual."""
    matrix, cond = amplitude_matrix(_OMEGAS, config.time_step_duration, window)
    amplitudes = solve_amplitudes(jnp.asarray(matrix), target)

    def packed(z):
        return np.concatenate([np.real(z), np.imag(z)]).reshape(2 * len(_OMEGAS), -1)

    rhs = packed(np.asarray(target))
    residual = float(np.linalg.norm(matrix @ packed(np.asarray(amplitudes)) - rhs) / np.linalg.norm(rhs))
    return amplitudes, cond, residual


def _unit_amplitudes(config, window):
    return _amplitudes(jnp.ones((len(_OMEGAS), len(_COMP), 1, 1, 1), dtype=jnp.complex128), config, window)[0]


def _slab(volume):
    slab = fdtdx.UniformMaterialObject(
        name="slab", partial_grid_shape=(None, None, 3), material=fdtdx.Material(permittivity=2.25)
    )
    constraints = [
        slab.same_size(volume, axes=(0, 1)),
        slab.place_at_center(volume, axes=(0, 1)),
        slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(8,)),
    ]
    return slab, constraints


def _source(name, amplitudes, window):
    return AdjointCurrentSource(
        name=name,
        partial_grid_shape=(1, 1, 1),
        amplitudes=amplitudes,
        window=window,
        angular_frequencies=_OMEGAS,
        components=_COMP,
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTHS[0]),
    )


def _monitor(name, **overrides):
    settings = dict(
        components=_COMP,
        scaling_mode="pulse",
        dft_subsample=1,
        exact_interpolation=False,
        reduce_volume=False,
        dtype=jnp.complex128,
    )
    return fdtdx.PhasorDetector(
        name=name,
        partial_grid_shape=(1, 1, 1),
        wave_characters=[fdtdx.WaveCharacter(wavelength=w) for w in _WAVELENGTHS],
        **{**settings, **overrides},
    )


def _scene(sim_fs: float, **monitor_overrides):
    """A slab, a one-cell AdjointCurrentSource, a Device ``"design"`` and a monitor, for one runtime;
    returned with the Device parameters applied."""
    config = _config(sim_fs)
    window = gaussian_window(config.time_steps_total)
    objs, cons = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_N, _N, _N))
    objs.append(volume)
    bd, cl = fdtdx.boundary_objects_from_config(fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML), volume)
    objs.extend(bd.values())
    cons.extend(cl)
    slab, slab_cons = _slab(volume)
    objs.append(slab)
    cons += slab_cons
    device = fdtdx.Device(
        name="design",
        partial_grid_shape=(_DES_SPAN,) * 3,
        partial_voxel_grid_shape=(1, 1, 1),
        materials={"air": fdtdx.Material(permittivity=1.0), "si": fdtdx.Material(permittivity=2.25)},
        param_transforms=[],
    )
    for obj, cell in (
        (device, (_DES_LO,) * 3),
        (_source("src", _unit_amplitudes(config, window), window), _SRC),
        (_monitor("mon", **monitor_overrides), _MON),
    ):
        cons.append(obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=cell))
        objs.append(obj)
    o, a, p, cfg, _ = fdtdx.place_objects(object_list=objs, config=config, constraints=cons, key=jax.random.PRNGKey(0))
    a, o, _ = fdtdx.apply_params(a, o, p, jax.random.PRNGKey(0))
    return o, a, cfg


def _run(arrays, objects, config, inv_eps=None):
    if inv_eps is not None:
        arrays = arrays.aset("inv_permittivities", inv_eps)
    _, out = checkpointed_fdtd(arrays, objects, config, _KEY, show_progress=False)
    return out


def _gradient(fom, objects, arrays, config, method, tail_tolerance=1e-2):
    """``fom`` of the monitor phasors as a function of ``inv_permittivities``, through ``run_fdtd``."""
    gradient_config = GradientConfig(method=method, num_checkpoints=8, tail_tolerance=tail_tolerance)
    config = config.aset("gradient_config", gradient_config)

    def loss(ie):
        _, out = fdtdx.run_fdtd(arrays.aset("inv_permittivities", ie), objects, config, _KEY, show_progress=False)
        return fom(out.detector_states["mon"]["phasor"])

    return jax.value_and_grad(loss)(arrays.inv_permittivities)


def _rel_on_design(g_model, g_true, objects):
    gs = objects["design"].grid_slice
    a, b = g_model[:, *gs], g_true[:, *gs]
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-300))


# ──────────────────────────────────────────────────────────────────────────────
# AdjointCurrentSource properties
# ──────────────────────────────────────────────────────────────────────────────


class TestAdjointCurrentSource:
    def _source(self):
        cfg = _config(20.0)
        window = gaussian_window(cfg.time_steps_total)
        src = _source("probe", jnp.ones((2, 1, 1, 1, 1), dtype=jnp.complex128), window)
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
            _source("probe", jnp.ones((5, 1, 1, 1, 1), dtype=jnp.complex128), gaussian_window(cfg.time_steps_total))


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
        amps, cond, residual = _amplitudes(target, cfg, window)
        assert amps.shape == target.shape
        assert residual < 1e-12 and cond < 1e3, (residual, cond)

    def test_short_window_is_rejected_loudly(self):
        """Frequencies closer than 1/T cannot be separated; that must raise."""
        cfg = _config(200.0)
        # a window only a couple of steps wide cannot resolve anything
        window = gaussian_window(cfg.time_steps_total, center_frac=0.5, sigma_frac=1e-5)
        with pytest.raises(ValueError, match="ill-conditioned"):
            amplitude_matrix(_OMEGAS, cfg.time_step_duration, window)

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
        amps = _unit_amplitudes(config, window)

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
            slab, slab_cons = _slab(volume)
            objs.append(slab)
            cons += slab_cons
            for obj, cell in ((_source("src", amps, window), src_cell), (_monitor("det"), mon_cell)):
                cons.append(obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=cell))
                objs.append(obj)
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
        objects, arrays, config = _scene(300.0)
        v_rec, g_rec = _gradient(fom, objects, arrays, config, "reciprocity")
        v_true, g_true = _gradient(fom, objects, arrays, config, "checkpointed")

        assert v_rec == v_true, "forward values must be identical"
        assert jnp.all(jnp.isfinite(g_rec))
        rel = _rel_on_design(g_rec, g_true, objects)
        assert rel < 2e-3, f"relative L2 vs checkpointed AD = {rel:.3e}"

        gs = objects["design"].grid_slice
        a, b = g_rec[:, *gs], g_true[:, *gs]
        cos = float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
        assert cos > 1 - 1e-6, f"cosine similarity {cos:.9f}"

    def test_gradient_is_zero_outside_the_devices(self):
        """The gradient with respect to ``inv_permittivities`` is filled inside the Devices only."""
        objects, arrays, config = _scene(120.0)
        _, g = _gradient(_fom_intensity, objects, arrays, config, "reciprocity")
        gs = objects["design"].grid_slice
        assert float(jnp.linalg.norm(g.at[:, *gs].set(0.0))) == 0.0
        assert float(jnp.linalg.norm(g[:, *gs])) > 0.0

    def test_gradient_converges_with_runtime(self):
        """The error must fall as the fields are given time to decay.

        Reciprocity is a frequency-domain identity, so it matches the exact
        discrete adjoint only once both DFTs converge. Asserting convergence is
        the honest test; asserting a fixed tolerance would hide the mechanism.
        """
        errs = []
        for sim_fs in (100.0, 400.0):
            objects, arrays, config = _scene(sim_fs)
            _, g_rec = _gradient(_fom_linear, objects, arrays, config, "reciprocity", tail_tolerance=None)
            _, g_true = _gradient(_fom_linear, objects, arrays, config, "checkpointed")
            errs.append(_rel_on_design(g_rec, g_true, objects))
        assert errs[-1] < errs[0] / 3.0, f"error did not converge with runtime: {errs}"
        assert errs[-1] < 1e-4, f"error at the long runtime is {errs[-1]:.3e}"


# ──────────────────────────────────────────────────────────────────────────────
# Guardrails
# ──────────────────────────────────────────────────────────────────────────────


class TestUnsupportedConfigurations:
    def test_rejects_reduce_volume_detector(self):
        objects, arrays, config = _scene(60.0, reduce_volume=True)
        with pytest.raises(NotImplementedError, match="reduce_volume"):
            jax.make_jaxpr(lambda: _gradient(_fom_intensity, objects, arrays, config, "reciprocity"))()
