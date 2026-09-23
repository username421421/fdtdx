"""The recording transpose against FDTDX's own detector update, no solves.

``fdtdx.adjoint.objective`` claims that for every channel of an objective
detector, ``sum(R(x) * y) == sum over blocks of sum(x[block] * R^T(y))`` with
``R`` the map from raw Yee fields to what the detector stores. Here ``R`` is not
re-derived: it is one call of FDTDX's ``update_detector_states`` at step 0,
which is exactly what the forward run records per step. Both branches of that
function are covered (the interior co-location block and the padded
whole-domain path), including a periodic axis, where the stencil wraps to the
opposite face and the support splits into two blocks.

The pairing is the unconjugated one JAX's cotangents use, checked on complex,
non-symmetric random data. Real-valued data could not tell it apart from the
conjugated pairing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.adjoint.objective import canonical_components, channel_recordings, is_interior, stored_fields
from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.update import update_detector_states

_KEY = jax.random.PRNGKey(0)
_WCS = (fdtdx.WaveCharacter(wavelength=600e-9), fdtdx.WaveCharacter(wavelength=700e-9))


@pytest.fixture(autouse=True)
def _enable_x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _place(shape, boundary, detectors):
    """Place ``detectors`` (``[(detector, lower_corner)]``) in an empty volume."""
    config = SimulationConfig(time=5e-15, grid=UniformGrid(spacing=50e-9), backend="cpu", dtype=jnp.float64)
    vol = fdtdx.SimulationVolume(partial_grid_shape=shape)
    objs, cons = [vol], []
    bd, cl = fdtdx.boundary_objects_from_config(boundary, vol)
    objs.extend(bd.values())
    cons.extend(cl)
    for det, corner in detectors:
        objs.append(det)
        cons.append(det.set_grid_coordinates(axes=(0, 1, 2), sides=("-",) * 3, coordinates=corner))
    objects, arrays, _, config, _ = fdtdx.place_objects(object_list=objs, config=config, constraints=cons, key=_KEY)
    return objects, arrays, config


def _pml(thickness=2):
    return fdtdx.BoundaryConfig.from_uniform_bound(thickness=thickness)


def _periodic_y(pml=2):
    return fdtdx.BoundaryConfig(
        boundary_type_miny="periodic",
        boundary_type_maxy="periodic",
        thickness_grid_minx=pml,
        thickness_grid_maxx=pml,
        thickness_grid_miny=1,
        thickness_grid_maxy=1,
        thickness_grid_minz=pml,
        thickness_grid_maxz=pml,
    )


def _phasor(name, shape, **kw):
    return fdtdx.PhasorDetector(name=name, partial_grid_shape=shape, wave_characters=_WCS, dtype=jnp.complex128, **kw)


def _box(name, shape):
    return fdtdx.FieldProjectionAngleDetector(
        name=name,
        partial_grid_shape=shape,
        wave_characters=_WCS,
        exclude_surfaces=("z-",),
        projection_medium=fdtdx.Material(permittivity=1.0),
        dtype=jnp.complex128,
    )


def _fdtdx_record(objects, arrays, config, name, E, H):
    """What FDTDX's own update stores for ``(E, H)`` in one step, scale divided out.

    Step 0 has phase 1 at every frequency, and ``H_prev = H`` makes the exact
    path's time average the identity, leaving the purely spatial map.
    """
    det = next(d for d in objects.detectors if d.name == name)

    @jax.jit
    def record(E, H):
        arrs = arrays.aset("fields", arrays.fields.aset("E", E).aset("H", H))
        return update_detector_states(jnp.asarray(0), arrs, objects, config, H_prev=H, inverse=False).detector_states

    scale = float(det._static_scale())
    return {k: v[0] / scale for k, v in record(E, H)[name].items()}


def _random_complex(key, shape):
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)


def _check_pairing(objects, arrays, config, name):
    """Assert the unconjugated (and conjugated) pairing on every channel; return the recordings."""
    det = next(d for d in objects.detectors if d.name == name)
    comps = canonical_components(det)
    recordings = channel_recordings(det, objects, config)
    grid = tuple(int(n) for n in objects.volume.grid_shape)
    kE, kH, ky = jax.random.split(jax.random.PRNGKey(7), 3)
    E = _random_complex(kE, (3, *grid))
    H = _random_complex(kH, (3, *grid))
    # R is real-linear, so R(x) for complex x is assembled from two real records
    re = _fdtdx_record(objects, arrays, config, name, jnp.real(E), jnp.real(H))
    im = _fdtdx_record(objects, arrays, config, name, jnp.imag(E), jnp.imag(H))
    x_stored = stored_fields(E, H, comps, tuple((0, n) for n in grid))
    for i, rec in enumerate(recordings):
        Rx = re[rec.state_key] + 1j * im[rec.state_key]  # (nf, nc, *channel)
        y = _random_complex(jax.random.fold_in(ky, i), Rx.shape)
        lhs = jnp.sum(Rx * y)
        lhs_conj = jnp.sum(jnp.conj(Rx) * y)
        blocks_T = rec.transpose(y)
        assert len(blocks_T) == len(rec.blocks)
        rhs = 0.0
        rhs_conj = 0.0
        for block, T in zip(rec.blocks, blocks_T):
            assert T.shape == (len(_WCS), len(comps), *(hi - lo for lo, hi in block))
            xb = x_stored[:, *(slice(lo, hi) for lo, hi in block)][None]
            rhs = rhs + jnp.sum(xb * T)
            rhs_conj = rhs_conj + jnp.sum(jnp.conj(xb) * T)
        scale = float(jnp.abs(lhs))
        assert abs(complex(lhs - rhs)) < 1e-11 * scale, f"{name}/{rec.state_key}: {lhs} vs {rhs}"
        assert abs(complex(lhs_conj - rhs_conj)) < 1e-11 * scale, f"{name}/{rec.state_key} conjugated"
    return recordings


class TestPairingAgainstFdtdxUpdate:
    def test_interior_block_all_six(self):
        objects, arrays, config = _place((12, 12, 12), _pml(), [(_phasor("mon", (2, 3, 2)), (5, 4, 6))])
        det = objects.detectors[0]
        assert det.exact_interpolation and is_interior(det.grid_slice_tuple, objects.volume.grid_shape)
        (rec,) = _check_pairing(objects, arrays, config, "mon")
        # detector ((5, 7), (4, 7), (6, 8)): backward averages reach one cell below
        # in x and y, the forward one in z one cell above
        assert rec.exact and rec.blocks == (((4, 7), (3, 7), (6, 9)),)

    def test_components_declared_out_of_order(self):
        det = _phasor("mon", (1, 1, 1), components=("Hz", "Ex"))
        objects, arrays, config = _place((10, 10, 10), _pml(), [(det, (5, 5, 5))])
        _check_pairing(objects, arrays, config, "mon")

    def test_non_interior_edge_uses_the_zero_halo(self):
        """A detector on the domain edge (non-periodic): FDTDX's padded path, clipped support."""
        objects, arrays, config = _place((10, 10, 10), _pml(), [(_phasor("mon", (2, 2, 2)), (0, 4, 4))])
        det = objects.detectors[0]
        assert not is_interior(det.grid_slice_tuple, objects.volume.grid_shape)
        (rec,) = _check_pairing(objects, arrays, config, "mon")
        (block,) = rec.blocks
        assert block[0] == (0, 2), "the x support is clipped at the domain edge"

    def test_periodic_axis_spanning_the_whole_period(self):
        """The neural-to-coverage port: 2D x-z, two periodic y cells, plane spans y."""
        objects, arrays, config = _place((12, 2, 12), _periodic_y(), [(_phasor("port", (1, 2, 5)), (6, 0, 3))])
        (rec,) = _check_pairing(objects, arrays, config, "port")
        (block,) = rec.blocks
        assert block[1] == (0, 2)

    def test_periodic_wrap_splits_the_support(self):
        """A detector at y = 0 reads y = -1, which on a periodic axis is the far face."""
        objects, arrays, config = _place((8, 8, 8), _periodic_y(), [(_phasor("mon", (2, 2, 2)), (3, 0, 3))])
        (rec,) = _check_pairing(objects, arrays, config, "mon")
        assert sorted({b[1] for b in rec.blocks}) == [(0, 2), (7, 8)]
        assert len(rec.blocks) == 2

    def test_box_projection_interior_faces(self):
        objects, arrays, config = _place((14, 14, 14), _pml(), [(_box("ff", (6, 6, 6)), (4, 4, 4))])
        recs = _check_pairing(objects, arrays, config, "ff")
        assert len(recs) == 5 and "phasor_z_minus" not in {r.state_key for r in recs}
        assert all(len(r.blocks) == 1 and r.exact for r in recs)

    def test_box_projection_touching_a_periodic_face(self):
        """A box spanning the periodic axis: every face through the padded path."""
        objects, arrays, config = _place((10, 6, 10), _periodic_y(), [(_box("ff", (4, 6, 4)), (3, 0, 3))])
        _check_pairing(objects, arrays, config, "ff")


class TestRawFieldsAreTheIdentity:
    def test_non_exact_detector_has_no_stencil(self):
        det = _phasor("mon", (2, 2, 2), exact_interpolation=False)
        objects, arrays, config = _place((10, 10, 10), _pml(), [(det, (4, 4, 4))])
        (rec,) = _check_pairing(objects, arrays, config, "mon")
        assert not rec.exact
        assert rec.blocks == (objects.detectors[0].grid_slice_tuple,) == (((4, 6), (4, 6), (4, 6)),)
        y = jnp.ones((2, 6, 2, 2, 2), dtype=jnp.complex128)
        (T,) = rec.transpose(y)
        assert T is y

    def test_the_pairing_check_has_teeth(self):
        """Pretending an exact detector records raw fields breaks the pairing."""
        objects, arrays, config = _place((12, 12, 12), _pml(), [(_phasor("mon", (2, 2, 2)), (5, 5, 5))])
        det = objects.detectors[0]
        raw = det.aset("exact_interpolation", False)
        (wrong,) = channel_recordings(raw, objects, config)
        grid = tuple(int(n) for n in objects.volume.grid_shape)
        E = _random_complex(jax.random.PRNGKey(1), (3, *grid))
        H = _random_complex(jax.random.PRNGKey(2), (3, *grid))
        re = _fdtdx_record(objects, arrays, config, "mon", jnp.real(E), jnp.real(H))["phasor"]
        im = _fdtdx_record(objects, arrays, config, "mon", jnp.imag(E), jnp.imag(H))["phasor"]
        y = _random_complex(jax.random.PRNGKey(3), re.shape)
        lhs = jnp.sum((re + 1j * im) * y)
        (block,) = wrong.blocks
        x = jnp.concatenate([E, H])[:, *(slice(lo, hi) for lo, hi in block)][None]
        rhs = jnp.sum(x * wrong.transpose(y)[0])
        assert abs(complex(lhs - rhs)) > 1e-2 * float(jnp.abs(lhs))


def test_support_blocks_cover_only_cells_the_stencil_reads():
    """The block is the stencil's reach, inside the (s - 1, e + 1) block FDTDX slices."""
    objects, _arrays, config = _place((12, 12, 12), _pml(), [(_phasor("mon", (3, 3, 3)), (4, 4, 4))])
    det = objects.detectors[0]
    (rec,) = channel_recordings(det, objects, config)
    (block,) = rec.blocks
    (sx, ex), (sy, ey), (sz, ez) = det.grid_slice_tuple
    assert block == ((sx - 1, ex), (sy - 1, ey), (sz, ez + 1))
    assert np.all(np.asarray([hi - lo for lo, hi in block]) == 4)
