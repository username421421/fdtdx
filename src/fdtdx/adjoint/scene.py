"""Derive the adjoint scene from an already-placed forward scene.

Why this exists. The obvious approach is to call :func:`fdtdx.place_objects`
twice, once with the real source and once with an
:class:`~fdtdx.objects.sources.adjoint.AdjointCurrentSource`. That is a
correctness bug as soon as the scene contains a :class:`~fdtdx.objects.device.Device`:
``place_objects`` splits its PRNG key once per placed object before initializing
device parameters, so two scenes with different object counts get **different
initial device parameters**. Measured on a 16^3 scene with one 4^3 device, the
only difference being one extra source object:

    1 source  -> device parameter sum 32.3474464417
    2 sources -> device parameter sum 27.7113399506

The forward and adjoint runs would then be simulating different structures, and
nothing downstream would notice. So the adjoint container is derived from the
placed forward container instead: same grid, same materials, same device, every
``Source`` swapped for adjoint current sources placed on the objective monitors'
own cells.

A box-mode field projection detector expands into one source per included face,
because it stores one phasor array per face and each face needs its own adjoint
current.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.default_key import default_key
from fdtdx.fdtd.container import ObjectContainer
from fdtdx.objects.detectors.detector import Detector
from fdtdx.objects.detectors.field_projection import (
    FieldProjectionDetectorBase,
    _surface_axis_direction,
    _surface_state_key,
)
from fdtdx.objects.sources.adjoint import AdjointCurrentSource
from fdtdx.objects.sources.source import Source


def find_object(objects: ObjectContainer, name: str):
    """Look up a placed object by name, with a useful error if it is missing."""
    for obj in objects.object_list:
        if getattr(obj, "name", None) == name:
            return obj
    available = [getattr(o, "name", None) for o in objects.object_list]
    raise ValueError(f"No object named {name!r}. Placed objects: {available}")


def is_box_projection(detector) -> bool:
    """Is this a field projection detector recording one phasor array per box face?"""
    return isinstance(detector, FieldProjectionDetectorBase) and detector._projection_mode == "box"


def face_slice_tuple(detector, surface: str):
    """Absolute grid slice of one box face, keeping its singleton normal axis.

    Mirrors the face slicing inside ``FieldProjectionDetectorBase.update``: the
    ``"-"`` face is the first cell along the normal axis and ``"+"`` is the last.
    """
    axis, direction = _surface_axis_direction(surface)
    bounds = list(detector.grid_slice_tuple)
    lo, hi = bounds[axis]
    bounds[axis] = (lo, lo + 1) if direction == "-" else (hi - 1, hi)
    return tuple(bounds)


def detector_channels(detector, name: str) -> list[tuple[str, tuple]]:
    """State keys and absolute face slices this detector needs adjoint currents for.

    A plain phasor detector has one channel over its whole slice; a box
    projection detector has one per included face.
    """
    if is_box_projection(detector):
        return [
            (_surface_state_key(surface), face_slice_tuple(detector, surface))
            for surface in detector._included_box_surfaces()
        ]
    return [("phasor", detector.grid_slice_tuple)]


def derive_adjoint_objects(
    objects: ObjectContainer,
    config: SimulationConfig,
    objective_detectors: str | Sequence[str],
    window: jax.Array,
    key: jax.Array | None = None,
    name_prefix: str = "__adjoint__",
) -> tuple[ObjectContainer, tuple[tuple[str, ...], ...]]:
    """Return an adjoint container derived from a placed forward container.

    Every :class:`Source` is dropped and replaced with adjoint current sources:
    one per plain objective monitor, or one per included face for a box-mode
    field projection detector. Each carries zero amplitudes; the reciprocity VJP
    fills them in its backward rule, which is cheap because they are traced
    leaves.

    The sources are placed with ``place_on_grid`` rather than through
    ``place_objects``, so the resolved grid, materials and device parameters are
    shared with the forward scene by construction.

    Args:
        objects: a **placed** forward container.
        config: the resolved config from the same ``place_objects`` call.
        objective_detectors: monitor name, or a sequence of them.
        window: adjoint excitation envelope, length ``config.time_steps_total``.
        key: PRNG key for ``place_on_grid``; deterministic if omitted.
        name_prefix: prefix for the derived source names.

    Returns:
        ``(adjoint_objects, source_names)`` where ``source_names[i]`` lists the
        sources for ``objective_detectors[i]``, ordered to match
        :func:`detector_channels`.

    Raises:
        ValueError: if a named detector is missing or is not a Detector, or the
            forward container has no volume.
    """
    key = default_key(key)
    names = (objective_detectors,) if isinstance(objective_detectors, str) else tuple(objective_detectors)
    if not names:
        raise ValueError("at least one objective detector is required")

    sources = []
    per_detector: list[tuple[str, ...]] = []
    for det_name in names:
        detector = find_object(objects, det_name)
        if not isinstance(detector, Detector):
            raise ValueError(f"{det_name!r} is a {type(detector).__name__}, not a Detector")
        omegas = tuple(float(w) for w in detector._angular_frequencies)
        components = tuple(detector.components)
        made: list[str] = []
        for state_key, slice_tuple in detector_channels(detector, det_name):
            shape = tuple(hi - lo for lo, hi in slice_tuple)
            src_name = f"{name_prefix}{det_name}__{state_key}"
            source = AdjointCurrentSource(
                name=src_name,
                amplitudes=jax.numpy.zeros((len(omegas), len(components), *shape), dtype=jax.numpy.complex128),
                window=window,
                angular_frequencies=omegas,
                components=components,
                wave_character=detector.wave_characters[0],
            )
            sources.append(source.place_on_grid(grid_slice_tuple=slice_tuple, config=config, key=key))
            made.append(src_name)
        per_detector.append(tuple(made))

    kept = [o for o in objects.object_list if not isinstance(o, Source)]
    volume = objects.volume
    new_list = [*kept, *sources]
    try:
        volume_idx = new_list.index(volume)
    except ValueError as exc:  # pragma: no cover - a container always has a volume
        raise ValueError("forward container has no volume object to carry over") from exc

    return ObjectContainer(object_list=new_list, volume_idx=volume_idx), tuple(per_detector)
