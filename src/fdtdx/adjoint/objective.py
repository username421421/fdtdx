"""Objective side: what a monitor stores, the transpose of how it records, where its adjoint current goes.

A phasor detector stores one or more *channels* (state keys), each a linear, time-invariant
spatial map ``R`` of the fields, DFT'd:

    P = scale * R(DFT(E), (1 + exp(+i w dt)) / 2 * DFT(H))

With ``exact_interpolation=False`` ``R`` selects the raw Yee fields on the detector's cells.
With ``exact_interpolation=True`` (the default) it is FDTDX's co-location stencil
:func:`~fdtdx.core.physics.curl.interpolate_fields`, read on the block ``(s - 1, e + 1)`` for
an interior detector and on the padded whole domain
(:func:`~fdtdx.fdtd.update.pad_fields_with_symmetry_mirror`: periodic wrap, zero halo,
mirror) otherwise, exactly as ``update_detector_states`` does; the H half is the time
average ``(H_prev + H) / 2``. ``R^T`` is ``jax.linear_transpose`` of FDTDX's own code, never
a hand-written stencil. Its nonzero cells, found by transposing random cotangents at setup,
are where the adjoint currents go: one :class:`AdjointCurrentSource` per contiguous block,
so a stencil wrapping around a periodic axis gets two.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.config import SimulationConfig
from fdtdx.constants import eta0
from fdtdx.core.physics.curl import interpolate_fields
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.fdtd.update import pad_fields_with_symmetry_mirror
from fdtdx.objects.detectors.field_projection import (
    FieldProjectionDetectorBase,
    _surface_axis_direction,
    _surface_state_key,
)
from fdtdx.objects.sources.adjoint import COMPONENT_MAP, AdjointCurrentSource
from fdtdx.typing import SliceTuple3D

#: Name prefix of the adjoint currents.
ADJOINT_SOURCE_PREFIX = "__adjoint__"


def complex_dtype(config: SimulationConfig):
    """The complex dtype matching the simulation precision."""
    return jnp.complex128 if config.dtype == jnp.float64 else jnp.complex64


def canonical_components(detector) -> tuple[str, ...]:
    """The detector's components in the order its phasors store them (always Ex..Hz, whatever was declared)."""
    declared = set(detector.components)
    return tuple(c for c in COMPONENT_MAP if c in declared)


def is_box_projection(detector) -> bool:
    """Is this a field projection detector storing one phasor array per box face?"""
    return isinstance(detector, FieldProjectionDetectorBase) and detector._projection_mode == "box"


def _face(slice_tuple, axis: int, side: str):
    """``slice_tuple`` reduced to its one-cell face on ``axis`` (``"-"``/``"min"`` first, else last)."""
    bounds = list(slice_tuple)
    lo, hi = bounds[axis]
    bounds[axis] = (lo, lo + 1) if side in ("-", "min") else (hi - 1, hi)
    return tuple(bounds)


def detector_channels(detector) -> list[tuple[str, SliceTuple3D]]:
    """``[(state_key, absolute cells), ...]`` for each phasor array the detector stores.

    Known layouts: one ``"phasor"`` key (:class:`PhasorDetector` and subclasses that only add a
    readout, e.g. ``PhasorPoyntingFluxDetector``, ``ModeOverlapDetector``);
    ``phasor_{axis}_{minus,plus}`` of box-mode field projection; ``phasor_axis{a}_{min,max}`` of
    ``ClosedSurfacePhasorPoyntingFluxDetector``.

    Raises:
        NotImplementedError: for any other layout, whose cells cannot be known.
    """
    keys = sorted(detector._shape_dtype_single_time_step().keys())
    if keys == ["phasor"]:
        return [("phasor", detector.grid_slice_tuple)]
    if is_box_projection(detector):
        return [
            (_surface_state_key(surface), _face(detector.grid_slice_tuple, *_surface_axis_direction(surface)))
            for surface in detector._included_box_surfaces()
        ]
    if all(k.startswith("phasor_axis") for k in keys):
        channels = []
        for key in keys:
            axis_text, _, side = key[len("phasor_axis") :].partition("_")
            if side not in ("min", "max") or not axis_text.isdigit():
                raise NotImplementedError(f"detector {detector.name!r}: cannot parse state key {key!r}")
            channels.append((key, _face(detector.grid_slice_tuple, int(axis_text), side)))
        return channels
    raise NotImplementedError(
        f"Detector {detector.name!r} ({type(detector).__name__}) stores phasors under unrecognised keys {keys}, "
        "so the cells each key reads are unknown. Record a PhasorDetector and post-process its phasors in JAX."
    )


def is_interior(slice_tuple: SliceTuple3D, grid_shape: Sequence[int]) -> bool:
    """``update_detector_states``' test for interpolating the local block ``(s - 1, e + 1)``."""
    return all(s >= 1 and e <= grid_shape[a] - 1 for a, (s, e) in enumerate(slice_tuple))


@dataclass(frozen=True)
class ChannelRecording:
    """How one stored phasor array reads the fields, and its transpose.

    Attributes:
        state_key: the detector-state key of the channel.
        channel_slice: absolute cells the channel stores.
        blocks: absolute cells of its adjoint currents, one ``AdjointCurrentSource`` each.
        transpose: ``ct (nf, nc, *channel) -> tuple of (nf, nc, *block)``, one per block.
        exact: whether the channel records co-located (and so time-averaged H) fields.
    """

    state_key: str
    channel_slice: SliceTuple3D
    blocks: tuple[SliceTuple3D, ...]
    transpose: Callable[[jax.Array], tuple[jax.Array, ...]]
    exact: bool


def _shape(slice_tuple: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(hi - lo for lo, hi in slice_tuple)


def _select(E: jax.Array, H: jax.Array, components: Sequence[str]) -> jax.Array:
    return jnp.stack([(E if COMPONENT_MAP[c][0] == "E" else H)[COMPONENT_MAP[c][1]] for c in components], axis=0)


def stored_fields(E: jax.Array, H: jax.Array, components: Sequence[str], cells: SliceTuple3D) -> jax.Array:
    """The raw fields ``(nc, *cells)`` of ``components`` on ``cells``."""
    index = tuple(slice(int(lo), int(hi)) for lo, hi in cells)
    return jnp.stack([(E if COMPONENT_MAP[c][0] == "E" else H)[COMPONENT_MAP[c][1]][index] for c in components])


def _transpose_of(record, input_shape: tuple[int, ...], components: Sequence[str]):
    """``ct (nf, nc, *out) -> (nf, nc, *input_shape)``, the transpose of ``record(E, H)``.

    The primal prototypes take the cotangent's dtype, which ``jax.linear_transpose`` requires.
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


def _support_blocks(transpose, channel_shape: tuple[int, ...], num_components: int, dtype) -> list[SliceTuple3D]:
    """Contiguous blocks, local to the transpose's input, covering its nonzero cells.

    Found by transposing two random complex cotangents, so the placement follows what
    FDTDX's stencil and padding read (wrapped halo and mirror partner included); two draws
    make an exact cancellation a non-event.
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
    assert mask is not None and mask.any(), "the detector's recording transpose is identically zero"
    per_axis = [_segments(np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))) for axis in range(3)]
    return [(bx, by, bz) for bx, by, bz in itertools.product(*per_axis)]


def channel_recordings(detector, objects: ObjectContainer, config: SimulationConfig) -> list[ChannelRecording]:
    """One :class:`ChannelRecording` per channel of a placed phasor detector, in :func:`detector_channels` order.

    ``objects`` is the container the detector records in; its boundaries and volume define
    the padding of a non-interior detector.
    """
    channels = detector_channels(detector)
    comps = canonical_components(detector)
    if not detector.exact_interpolation:
        return [
            ChannelRecording(state_key=key, channel_slice=sl, blocks=(sl,), transpose=lambda ct: (ct,), exact=False)
            for key, sl in channels
        ]

    grid_shape = tuple(int(n) for n in objects.volume.grid_shape)
    # decided per detector, as update_detector_states does: a box whose halo touches an
    # edge interpolates every face through the padded domain
    interior = is_interior(detector.grid_slice_tuple, grid_shape)
    dtype = complex_dtype(config)
    out: list[ChannelRecording] = []
    for key, channel_slice in channels:
        if interior:
            # interpolating a face over its own block equals slicing it out of the box's
            # interpolation: the stencil reaches one cell and its weights are position-indexed
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

        # jitted: it runs twice at setup (the support probe) and once per backward pass
        full_transpose = jax.jit(_transpose_of(record, _shape(input_slice), comps))
        local_blocks = _support_blocks(full_transpose, _shape(channel_slice), len(comps), dtype)
        (ox, _), (oy, _), (oz, _) = input_slice
        blocks = tuple(
            ((x0 + ox, x1 + ox), (y0 + oy, y1 + oy), (z0 + oz, z1 + oz))
            for (x0, x1), (y0, y1), (z0, z1) in local_blocks
        )
        index = tuple((slice(None), slice(None), *(slice(lo, hi) for lo, hi in b)) for b in local_blocks)

        def transpose(ct, full_transpose=full_transpose, index=index):
            full = full_transpose(ct)
            return tuple(full[i] for i in index)

        out.append(ChannelRecording(key, channel_slice, blocks, transpose, exact=True))
    return out


def adjoint_sources(
    name: str,
    detector,
    recordings: Sequence[ChannelRecording],
    config: SimulationConfig,
    window: jax.Array,
    key: jax.Array,
) -> list[AdjointCurrentSource]:
    """Zero-amplitude adjoint currents, placed on every block of every channel, in channel then block order.

    The backward rule fills in the amplitudes, which are traced leaves.
    """
    omegas = tuple(float(w) for w in detector._angular_frequencies)
    components = canonical_components(detector)
    sources = []
    for rec in recordings:
        for b_i, block in enumerate(rec.blocks):
            suffix = "" if len(rec.blocks) == 1 else f"__block{b_i}"
            source = AdjointCurrentSource(
                name=f"{ADJOINT_SOURCE_PREFIX}{name}__{rec.state_key}{suffix}",
                amplitudes=jnp.zeros((len(omegas), len(components), *_shape(block)), dtype=complex_dtype(config)),
                window=window,
                angular_frequencies=omegas,
                components=components,
                wave_character=detector.wave_characters[0],
            )
            sources.append(source.place_on_grid(grid_slice_tuple=block, config=config, key=key))
    return sources


def raw_scale(detector) -> float:
    """The stored phasor's scale over the raw every-step DFT: ``_static_scale() / stride``.

    Pulse mode's scale is the stride itself; continuous mode's ``2 / sum(window)`` counts kept steps.
    """
    return float(detector._static_scale()) / int(detector._dft_stride)


def target_factor(detector, exact: bool, angular_frequencies: Sequence[float], dt: float) -> np.ndarray:
    """``(nf, nc)`` factor taking a stored channel's cotangent to its adjoint current's target DFT.

    ``raw_scale`` on every component, since the kernel pairs raw phasors. On a magnetic
    component also ``-exp(-i w dt / 2)``: the minus sign of Lorentz reciprocity's magnetic
    pairing, and the half step between the stored post-update H and the injection at
    ``time_step + 0.5``; with exact interpolation times ``(1 + exp(+i w dt)) / 2``, the DFT
    of the time average ``(H_prev + H) / 2``.
    """
    w = np.asarray(angular_frequencies, dtype=np.float64)
    half_step = -np.exp(-1j * w * dt / 2.0)
    time_average = (1.0 + np.exp(1j * w * dt)) / 2.0
    magnetic_factor = half_step * time_average if exact else half_step
    is_magnetic = np.asarray([c.startswith("H") for c in canonical_components(detector)])
    return np.where(is_magnetic[None, :], magnetic_factor[:, None], 1.0 + 0.0j) * raw_scale(detector)


def _component_row(arr: jax.Array, axis: int) -> jax.Array:
    """Row ``axis`` of a ``(1 | 3, ...)`` material array: the one row when isotropic."""
    return arr[axis] if arr.shape[0] > 1 else arr[0]


@dataclass(frozen=True)
class LossyInjection:
    """FDTDX's lossy-update divisor on the cells of one adjoint current.

    ``update_E`` divides the curl update by ``1 + a``, ``a = courant * sigma_E * eta0 * inv_eps / 2``,
    and only then adds the sources, so a current in a lossy cell enters ``1 + a`` times stronger
    than the reciprocal partner of the field the monitor reads there; the target is divided by it
    per cell and component (``1 + b``, ``b = courant * sigma_H * inv_mu / (2 eta0)``, on H). In a
    lossy design cell the factor appears on both sides of the pairing and cancels.

    Attributes:
        index: the block's cells.
        axes: per stored component, its axis, i.e. the row of a ``(3, ...)`` material array it uses.
        electric: per stored component, ``courant * eta0 / 2`` on E and 0 on H.
        magnetic: ``b`` per component and cell (0 on E); ``sigma_H`` and ``inv_mu`` are static.
    """

    index: tuple[slice, ...]
    axes: tuple[int, ...]
    electric: np.ndarray
    magnetic: np.ndarray

    def divisor(self, inv_eps: jax.Array, sigma_e: jax.Array | None) -> jax.Array:
        """``1 + a`` (E) or ``1 + b`` (H), shape ``(nc, *block)``, at the live ``inv_eps`` and ``sigma_E``."""
        magnetic = jnp.asarray(self.magnetic, dtype=inv_eps.dtype)
        if sigma_e is None:
            return 1.0 + magnetic

        def rows(arr):
            local = arr[:, *self.index]
            return jnp.stack([_component_row(local, axis) for axis in self.axes], axis=0)

        electric = jnp.asarray(self.electric, dtype=inv_eps.dtype).reshape((-1,) + (1,) * (inv_eps.ndim - 1))
        return 1.0 + electric * rows(sigma_e) * rows(inv_eps) + magnetic


def lossy_injection(
    arrays: ArrayContainer,
    courant: float,
    block: Sequence[tuple[int, int]],
    components: Sequence[str],
) -> LossyInjection | None:
    """The lossy-update divisor on ``block``, or ``None`` in a scene without conductivity.

    Built whenever the scene has an ``electric_conductivity``, which may be design-dependent
    (a lossy Device); a lossless block then divides by exactly one.
    """
    sigma_e = arrays.electric_conductivity
    sigma_h = arrays.magnetic_conductivity
    if sigma_e is None and sigma_h is None:
        return None
    index = tuple(slice(int(lo), int(hi)) for lo, hi in block)
    shape = tuple(int(hi) - int(lo) for lo, hi in block)
    axes = tuple(COMPONENT_MAP[c][1] for c in components)
    electric = np.asarray([courant * float(eta0) / 2.0 if c.startswith("E") else 0.0 for c in components])
    magnetic = np.zeros((len(components), *shape))
    inv_mu = arrays.inv_permeabilities
    for k, comp in enumerate(components):
        if comp.startswith("H") and sigma_h is not None:
            sigma = np.asarray(jax.device_get(_component_row(sigma_h, axes[k])[index]), dtype=np.float64)
            if isinstance(inv_mu, jax.Array) and inv_mu.ndim > 0:
                mu = np.asarray(jax.device_get(_component_row(inv_mu, axes[k])[index]), dtype=np.float64)
            else:
                mu = float(inv_mu)
            magnetic[k] = courant * sigma * mu / (2.0 * float(eta0))
    return LossyInjection(index=index, axes=axes, electric=electric, magnetic=magnetic)
