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
``Source`` swapped for one adjoint current source placed on the objective
monitor's own cells.
"""

from __future__ import annotations

import jax

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.default_key import default_key
from fdtdx.fdtd.container import ObjectContainer
from fdtdx.objects.detectors.detector import Detector
from fdtdx.objects.sources.adjoint import AdjointCurrentSource
from fdtdx.objects.sources.source import Source

#: Name given to the derived adjoint source.
ADJOINT_SOURCE_NAME = "__adjoint_source__"


def find_object(objects: ObjectContainer, name: str):
    """Look up a placed object by name, with a useful error if it is missing."""
    for obj in objects.object_list:
        if getattr(obj, "name", None) == name:
            return obj
    available = [getattr(o, "name", None) for o in objects.object_list]
    raise ValueError(f"No object named {name!r}. Placed objects: {available}")


def derive_adjoint_objects(
    objects: ObjectContainer,
    config: SimulationConfig,
    objective_detector: str,
    window: jax.Array,
    key: jax.Array | None = None,
    adjoint_source_name: str = ADJOINT_SOURCE_NAME,
) -> tuple[ObjectContainer, str]:
    """Return an adjoint container derived from a placed forward container.

    Every :class:`Source` is dropped and replaced with a single
    :class:`AdjointCurrentSource` occupying the objective detector's grid slice,
    carrying zero amplitudes. The reciprocity VJP fills those amplitudes in its
    backward rule, which is cheap because they are a traced leaf.

    The new source is placed with ``place_on_grid`` rather than through
    ``place_objects``, so the resolved grid, the materials and any device
    parameters are shared with the forward scene by construction.

    Args:
        objects: a **placed** forward container, as returned by ``place_objects``.
        config: the resolved config from the same ``place_objects`` call.
        objective_detector: name of the monitor the figure of merit reads.
        window: adjoint excitation envelope, length ``config.time_steps_total``.
        key: PRNG key for ``place_on_grid``; a deterministic one is used if omitted.
        adjoint_source_name: name for the derived source.

    Returns:
        ``(adjoint_objects, adjoint_source_name)``.

    Raises:
        ValueError: if the objective detector is missing, or the forward
            container holds no volume.
    """
    key = default_key(key)
    detector = find_object(objects, objective_detector)
    if not isinstance(detector, Detector):
        raise ValueError(f"{objective_detector!r} is a {type(detector).__name__}, not a Detector")

    omegas = tuple(float(w) for w in detector._angular_frequencies)
    components = tuple(detector.components)
    shape = (len(omegas), len(components), *detector.grid_shape)

    adjoint_source = AdjointCurrentSource(
        name=adjoint_source_name,
        amplitudes=jax.numpy.zeros(shape, dtype=jax.numpy.complex128),
        window=window,
        angular_frequencies=omegas,
        components=components,
        wave_character=detector.wave_characters[0],
    )
    adjoint_source = adjoint_source.place_on_grid(grid_slice_tuple=detector.grid_slice_tuple, config=config, key=key)

    kept = [o for o in objects.object_list if not isinstance(o, Source)]
    volume = objects.volume
    new_list = [*kept, adjoint_source]
    try:
        volume_idx = new_list.index(volume)
    except ValueError as exc:  # pragma: no cover - a container always has a volume
        raise ValueError("forward container has no volume object to carry over") from exc

    return ObjectContainer(object_list=new_list, volume_idx=volume_idx), adjoint_source_name
