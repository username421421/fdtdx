"""Pin the FDTDX internals the reciprocity gradient depends on.

The reciprocity transpose has to know things FDTDX does not expose publicly:
which frequencies a detector accumulates, which state keys it writes and what
cells each key reads, and the resolved DFT stride. Those are private attributes,
so an upstream change can move them.

Without this file such a change surfaces as a *wrong gradient* or an
``AttributeError`` inside someone's optimizer. With it, the contract breaks here,
in CI, naming the attribute and what the reciprocity code needs from it.

Two of the dependencies are deliberate. ``_surface_state_key`` and
``_surface_axis_direction`` are imported rather than reimplemented precisely so a
rename fails at import time; a local copy of the naming convention would keep
working and quietly generate keys that no longer match, which puts adjoint
currents in the wrong cells.

If a test here fails, do not relax it. Fix the adjoint code against the new
internals, or move to a public accessor if upstream added one.
"""

import jax
import jax.numpy as jnp
import pytest

import fdtdx
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.objects.detectors.field_projection import (
    FieldProjectionDetectorBase,
    _surface_axis_direction,
    _surface_state_key,
)
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.sources.source import Source

_WL = (600e-9, 800e-9)
_KEY = jax.random.PRNGKey(0)


@pytest.fixture
def config():
    return SimulationConfig(
        time=20e-15,
        grid=UniformGrid(spacing=50e-9),
        backend="cpu",
        dtype=jnp.float64,
    )


def _placed_phasor(config, **kwargs):
    det = PhasorDetector(
        wave_characters=[fdtdx.WaveCharacter(wavelength=w) for w in _WL],
        scaling_mode="pulse",
        dft_subsample=1,
        exact_interpolation=False,
        reduce_volume=False,
        dtype=jnp.complex128,
        **kwargs,
    )
    return det.place_on_grid(((0, 4), (0, 3), (0, 2)), config, _KEY)


class TestDetectorContract:
    def test_angular_frequencies_matches_public_wave_characters(self, config):
        """We read _angular_frequencies; wave_characters is the public source of truth.

        If these ever disagree, the adjoint amplitude solve would be built for
        different frequencies than the detector accumulates, which produces a
        confidently wrong gradient rather than an error.
        """
        det = _placed_phasor(config)
        private = tuple(float(w) for w in det._angular_frequencies)
        public = tuple(2.0 * jnp.pi * wc.get_frequency() for wc in det.wave_characters)
        assert len(private) == len(public)
        # 1e-6, not exact: the detector stores its frequencies at the simulation
        # dtype, so a float32 run round-trips them with about 1.6e-08 relative
        # error (0 in float64). The production check in fdtdx/adjoint/validation.py uses
        # the same tolerance for the same reason.
        for a, b in zip(private, public):
            assert abs(a - float(b)) / abs(float(b)) < 1e-6

    def test_plain_detector_state_is_a_single_phasor_key(self, config):
        """detector_channels treats a lone "phasor" key as one whole-slice channel."""
        det = _placed_phasor(config)
        shapes = det._shape_dtype_single_time_step()
        assert sorted(shapes) == ["phasor"]
        spec = shapes["phasor"]
        assert spec.shape == (len(_WL), len(det.components), 4, 3, 2), spec.shape

    def test_dft_stride_resolves_to_an_int(self, config):
        """We gate on _dft_stride == 1; it must stay a concrete int after placement."""
        det = _placed_phasor(config)
        assert isinstance(det._dft_stride, int)
        assert det._dft_stride == 1

    def test_placement_sets_config_and_slice(self, config):
        """AdjointCurrentSource reads _config for dt and grid_slice for placement."""
        det = _placed_phasor(config)
        assert det._config is not None
        assert det.grid_slice_tuple == ((0, 4), (0, 3), (0, 2))
        assert det.grid_shape == (4, 3, 2)


class TestProjectionContract:
    def _placed_box(self, config, span=7):
        det = fdtdx.FieldProjectionAngleDetector(
            partial_grid_shape=(span,) * 3,
            wave_characters=[fdtdx.WaveCharacter(wavelength=w) for w in _WL],
            exclude_surfaces=("z-",),
            origin=(0.0, 0.0, 0.0),
            projection_distance=1e-3,
            far_field_approx=True,
            projection_medium=fdtdx.Material(permittivity=1.0),
            scaling_mode="pulse",
            dft_subsample=1,
            dtype=jnp.complex128,
        ).aset("exact_interpolation", False)
        return det.place_on_grid(((0, span), (0, span), (0, span)), config, _KEY)

    def test_box_mode_and_faces(self, config):
        det = self._placed_box(config)
        assert isinstance(det, FieldProjectionDetectorBase)
        assert det._projection_mode == "box"
        faces = det._included_box_surfaces()
        assert "z-" not in faces, "exclude_surfaces must be honoured"
        assert len(faces) == 5

    def test_surface_helpers_still_exist_and_agree(self, config):
        """Imported rather than reimplemented so a rename fails loudly, here.

        A local copy of the naming convention would keep returning keys that no
        longer match the detector's state, silently misplacing adjoint currents.
        """
        det = self._placed_box(config)
        keys = set(det._shape_dtype_single_time_step())
        for surface in det._included_box_surfaces():
            assert _surface_state_key(surface) in keys, surface
            axis, direction = _surface_axis_direction(surface)
            assert axis in (0, 1, 2)
            assert direction in ("+", "-")

    def test_box_state_is_one_key_per_face_with_a_singleton_normal(self, config):
        span = 7
        det = self._placed_box(config, span=span)
        shapes = det._shape_dtype_single_time_step()
        assert len(shapes) == 5
        for surface in det._included_box_surfaces():
            axis, _ = _surface_axis_direction(surface)
            shape = shapes[_surface_state_key(surface)].shape
            spatial = shape[2:]
            assert spatial[axis] == 1, f"{surface}: normal axis not singleton, got {spatial}"
            assert len(shape) == 5

    def test_projection_detector_records_all_six_components(self, config):
        """Box mode concatenates E and H, so the adjoint source must drive six."""
        det = self._placed_box(config)
        assert tuple(det.components) == ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

    def test_project_is_a_pure_function_of_state(self, config):
        """The whole design rests on this: the projection sits above the VJP boundary."""
        det = self._placed_box(config)
        state = {k: jnp.zeros(v.shape, v.dtype)[None, ...] for k, v in det._shape_dtype_single_time_step().items()}
        out = det.project(state, jnp.linspace(0.0, 0.4, 3), jnp.linspace(0.0, 1.0, 2))
        assert "power" in out
        assert jnp.all(jnp.isfinite(out["power"]))


class TestSourceContract:
    def test_update_h_is_called_on_the_half_step(self):
        """The magnetic factor's exp(-i w dt / 2) assumes update_H sees n + 0.5.

        If FDTDX ever passes an integer time_step to update_H, that phase becomes
        wrong and the H part of every gradient tilts, with no error raised.
        """
        import inspect

        from fdtdx.fdtd import update as upd

        src = inspect.getsource(upd.update_H)
        assert "time_step + 0.5" in src, (
            "update_H no longer offsets the time step by half; the magnetic adjoint "
            "factor exp(-i w dt / 2) in fdtdx/adjoint/vjp.py assumes it does"
        )

    def test_source_injection_scales_with_local_inverse_permittivity(self):
        """The gradient kernel's (E_new - E_old)/inv_eps factoring assumes this."""
        import inspect

        from fdtdx.objects.sources import dipole

        src = inspect.getsource(dipole.PointDipoleSource.update_E)
        assert "inv_eps_oriented" in src and "courant" in inspect.getsource(dipole)
        # and that it caches, which is why an overlapping stock source is refused
        assert "_inv_eps_oriented" in inspect.getsource(dipole.PointDipoleSource.apply)

    def test_source_base_still_exposes_update_e_and_update_h(self):
        assert hasattr(Source, "update_E")
        assert hasattr(Source, "update_H")


class TestUpdateEquationContract:
    def test_lossy_update_keeps_inv_eps_outside_the_bracket(self):
        """Loss needs no extra gradient term only because of this factoring.

        E_new = E_old + courant * inv_eps * (curl - sigma*eta0*E_old/2) means
        dE_new/d(inv_eps) = (E_new - E_old)/inv_eps exactly as when lossless. If
        the lossy branch is ever rewritten so inv_eps no longer factors out, the
        kernel needs a conductivity term and lossy gradients go wrong silently.
        """
        import inspect

        from fdtdx.fdtd import update as upd

        src = inspect.getsource(upd.update_E)
        assert "factor = 1 - c * sigma_E * eta0 * inv_eps / 2" in src, (
            "the lossy E update changed shape; re-derive the gradient kernel in "
            "fdtdx/adjoint/kernel.py and re-run the sigma sweep"
        )
        assert "factor * arrays.fields.E + c * curl * inv_eps" in src
