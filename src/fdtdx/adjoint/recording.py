"""Transpose of what an objective detector records, and where its adjoint current goes.

A phasor detector's state is a DFT of the fields ``update_detector_states`` hands
its ``update``. With ``exact_interpolation=False`` those are the raw Yee fields on
the detector's cells. With ``exact_interpolation=True`` (every detector's default)
they are first co-located onto the E_z Yee point by
:func:`~fdtdx.core.physics.curl.interpolate_fields`, a one-cell stencil read

* from the domain itself, on the block ``(s - 1, e + 1)`` per axis, when that
  block stays inside the domain (the detector is *interior*); or
* from FDTDX's padded whole-domain fields
  (:func:`~fdtdx.fdtd.update.pad_fields_with_symmetry_mirror`: periodic wrap,
  zero halo, symmetry mirror) when it does not, e.g. for every detector spanning
  a periodic axis.

That spatial map ``R`` is linear and time-invariant, so it commutes with the DFT,

    P = scale * R(DFT(E), (1 + e^{+i w dt}) / 2 * DFT(H)),

the H factor being the DFT of the time average ``(H_prev + H) / 2`` the exact
path records. The cotangent the adjoint current has to reproduce is therefore
``R^T(ct)`` on the raw fields. It is built here with ``jax.linear_transpose`` of
FDTDX's *own* ``interpolate_fields`` and padding, never with a hand-derived
stencil, and its nonzero cells -- found by transposing a random cotangent once at
setup -- are where the adjoint currents are placed: one
:class:`~fdtdx.objects.sources.adjoint.AdjointCurrentSource` per contiguous
block, so a support that wraps around a periodic axis becomes two blocks.

The stencil coefficients are real, so ``R^T`` satisfies the unconjugated pairing
``sum(R(x) * y) == sum(x * R^T(y))`` that JAX's cotangent convention needs, and
the conjugated one as well.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.config import SimulationConfig
from fdtdx.core.physics.curl import interpolate_fields
from fdtdx.fdtd.container import ObjectContainer
from fdtdx.fdtd.update import pad_fields_with_symmetry_mirror
from fdtdx.typing import SliceTuple3D

#: Field family (0 = E, 1 = H) and component axis of every detector component.
FAMILY_AXIS: dict[str, tuple[int, int]] = {
    "Ex": (0, 0),
    "Ey": (0, 1),
    "Ez": (0, 2),
    "Hx": (1, 0),
    "Hy": (1, 1),
    "Hz": (1, 2),
}


def is_interior(slice_tuple: SliceTuple3D, grid_shape: Sequence[int]) -> bool:
    """FDTDX's own test (``update_detector_states``) for the local co-location block.

    The stencil reads domain cells ``s - 1 .. e`` on every axis; when that stays in
    bounds FDTDX interpolates the block ``(s - 1, e + 1)`` directly, otherwise it
    uses the padded whole-domain fields.
    """
    return all(s >= 1 and e <= grid_shape[a] - 1 for a, (s, e) in enumerate(slice_tuple))


@dataclass(frozen=True)
class ChannelRecording:
    """How one stored phasor array reads the fields, and its transpose.

    Attributes:
        state_key: the detector-state key this channel is stored under.
        channel_slice: absolute cells the channel's phasors are stored on.
        blocks: absolute cells the adjoint currents for this channel cover, one
            ``AdjointCurrentSource`` each.
        transpose: ``ct -> tuple of amplitudes``, mapping a cotangent of shape
            ``(nf, nc, *channel_shape)`` to one array ``(nf, nc, *block_shape)``
            per block. The identity for a raw-field detector.
        exact: whether the channel records co-located fields (and so H time-averaged).
    """

    state_key: str
    channel_slice: SliceTuple3D
    blocks: tuple[SliceTuple3D, ...]
    transpose: Callable[[jax.Array], tuple[jax.Array, ...]]
    exact: bool


def _shape(slice_tuple: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(hi - lo for lo, hi in slice_tuple)


def _shifted(block: SliceTuple3D, origin: Sequence[tuple[int, int]]) -> SliceTuple3D:
    """``block`` (local to a sub-block starting at ``origin``) in absolute cells."""
    (x0, x1), (y0, y1), (z0, z1) = block
    ox, oy, oz = (lo for lo, _ in origin)
    return ((x0 + ox, x1 + ox), (y0 + oy, y1 + oy), (z0 + oz, z1 + oz))


def _select(E: jax.Array, H: jax.Array, components: Sequence[str]) -> jax.Array:
    return jnp.stack([(E if FAMILY_AXIS[c][0] == 0 else H)[FAMILY_AXIS[c][1]] for c in components], axis=0)


def _transpose_of(
    record: Callable[[jax.Array, jax.Array], jax.Array],
    input_shape: tuple[int, ...],
    components: Sequence[str],
) -> Callable[[jax.Array], jax.Array]:
    """``ct (nf, nc, *out) -> (nf, nc, *input_shape)``, the transpose of ``record``.

    ``record(E, H)`` maps fields of shape ``(3, *input_shape)`` to the stored
    components ``(nc, *out)``. The primal prototypes take the cotangent's dtype:
    ``jax.linear_transpose`` refuses a cotangent whose dtype differs from them.
    """

    def one_frequency(ct: jax.Array) -> jax.Array:
        proto = jnp.zeros((3, *input_shape), dtype=ct.dtype)
        ct_E, ct_H = jax.linear_transpose(record, proto, proto)(ct)
        return _select(ct_E, ct_H, components)

    return jax.vmap(one_frequency)


def _segments(indices: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive integers, as half-open ``(lo, hi)``."""
    runs: list[tuple[int, int]] = []
    for i in indices.tolist():
        if runs and runs[-1][1] == i:
            runs[-1] = (runs[-1][0], i + 1)
        else:
            runs.append((i, i + 1))
    return runs


def _support_blocks(
    transpose: Callable[[jax.Array], jax.Array],
    channel_shape: tuple[int, ...],
    num_components: int,
    dtype: jnp.dtype,
) -> list[SliceTuple3D]:
    """Contiguous blocks (local to the transpose's input) covering its nonzero cells.

    Found numerically, by transposing two random complex cotangents, so the
    placement follows whatever FDTDX's stencil and padding actually read rather
    than a copy of them: the wrapped halo of a periodic axis and the mirror
    partner of a symmetry plane included. Random values make an exact
    cancellation a measure-zero event; two draws make it a non-event.
    """
    mask = None
    for seed in (0, 1):
        k_re, k_im = jax.random.split(jax.random.PRNGKey(seed))
        shape = (1, num_components, *channel_shape)
        real_dtype = jnp.float64 if dtype == jnp.complex128 else jnp.float32
        probe = (jax.random.normal(k_re, shape, real_dtype) + 1j * jax.random.normal(k_im, shape, real_dtype)).astype(
            dtype
        )
        out = np.abs(np.asarray(jax.device_get(transpose(probe)))).sum(axis=(0, 1)) > 0
        mask = out if mask is None else (mask | out)
    assert mask is not None
    if not mask.any():  # pragma: no cover - a detector always reads some cell
        raise RuntimeError("the detector's recording transpose is identically zero")
    per_axis = []
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        per_axis.append(_segments(np.flatnonzero(mask.any(axis=other))))
    return [(bx, by, bz) for bx, by, bz in itertools.product(*per_axis)]


def channel_recordings(
    detector,
    channels: Sequence[tuple[str, SliceTuple3D]],
    components: Sequence[str],
    objects: ObjectContainer,
    config: SimulationConfig,
) -> list[ChannelRecording]:
    """Recording maps, transposes and adjoint-current blocks for a detector's channels.

    Args:
        detector: a placed phasor detector.
        channels: ``detector_channels(detector, name)``, i.e. its state keys and
            the absolute cells each one stores.
        components: the components in the order the phasors store them.
        objects: the placed container the detector records in; its boundaries
            and volume define the padding for a non-interior detector.
        config: the resolved config.

    Returns:
        One :class:`ChannelRecording` per channel, in the given order.
    """
    comps = tuple(components)
    if not detector.exact_interpolation:
        return [
            ChannelRecording(state_key=key, channel_slice=sl, blocks=(sl,), transpose=lambda ct: (ct,), exact=False)
            for key, sl in channels
        ]

    grid_shape = tuple(int(n) for n in objects.volume.grid_shape)
    # Decided per DETECTOR, as update_detector_states does: a box whose halo touches
    # an edge interpolates every face through the padded domain.
    interior = is_interior(detector.grid_slice_tuple, grid_shape)
    dtype = jnp.complex128 if config.dtype == jnp.float64 else jnp.complex64

    out: list[ChannelRecording] = []
    for key, channel_slice in channels:
        if interior:
            # The block FDTDX interpolates. Interpolating a face over its own
            # block equals slicing the face out of the box's interpolation,
            # because the stencil reaches one cell and the weights are indexed
            # by absolute position (region_slice).
            input_slice = tuple((s - 1, e + 1) for s, e in channel_slice)

            def record(E, H, region=channel_slice):
                E_i, H_i = interpolate_fields(E, H, config=config, region_slice=region)
                return _select(E_i, H_i, comps)

        else:
            input_slice = tuple((0, n) for n in grid_shape)
            region_idx = tuple(slice(lo, hi) for lo, hi in channel_slice)

            def record(E, H, region_idx=region_idx):
                E_i, H_i = interpolate_fields(
                    E_pad=pad_fields_with_symmetry_mirror(E, objects, config, "E"),
                    H_pad=pad_fields_with_symmetry_mirror(H, objects, config, "H"),
                    config=config,
                )
                return _select(E_i, H_i, comps)[:, *region_idx]

        # jitted: it runs eagerly twice at setup (the support probe) and is one
        # small fused kernel inside the backward rule
        full_transpose = jax.jit(_transpose_of(record, _shape(input_slice), comps))
        local_blocks = _support_blocks(full_transpose, _shape(channel_slice), len(comps), dtype)
        blocks = tuple(_shifted(b, input_slice) for b in local_blocks)
        index = tuple((slice(None), slice(None), *(slice(lo, hi) for lo, hi in b)) for b in local_blocks)

        def transpose(ct, full_transpose=full_transpose, index=index):
            full = full_transpose(ct)
            return tuple(full[i] for i in index)

        out.append(
            ChannelRecording(
                state_key=key,
                channel_slice=channel_slice,
                blocks=blocks,
                transpose=transpose,
                exact=True,
            )
        )
    return out
