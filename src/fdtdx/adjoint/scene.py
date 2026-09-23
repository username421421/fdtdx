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

The design region needs no setup either. The detector that records the fields
there is internal scratch -- its phasors never leave the VJP -- so
:func:`internal_scene` builds it from scratch over each design region, in the
one configuration the gradient kernel is calibrated for, instead of asking the
user to configure one correctly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.default_key import default_key
from fdtdx.core.switch import OnOffSwitch
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.objects.detectors.detector import Detector
from fdtdx.objects.detectors.field_projection import (
    FieldProjectionDetectorBase,
    _surface_axis_direction,
    _surface_state_key,
)
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.object import INVALID_SLICE_TUPLE_3D, SimulationObject
from fdtdx.objects.sources.adjoint import AdjointCurrentSource
from fdtdx.objects.sources.source import Source
from fdtdx.typing import ProjectionSurface

#: The order in which :meth:`PhasorDetector.update` stacks components, whatever
#: order ``components`` lists them in. Box-mode projection concatenates ``(E, H)``,
#: which is the same order.
CANONICAL_COMPONENTS: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

#: Name prefix of the design-region detectors the reciprocity path builds itself.
DESIGN_DETECTOR_PREFIX = "__adjoint_design__"

#: The one configuration the gradient kernel is calibrated for. None of it is the
#: user's to get right, because the design phasors never leave the VJP. Each entry
#: was a user-facing setting whose wrong value was silent, measured against
#: ``run_fdtd(GradientConfig("checkpointed"))`` before this was internal:
#:
#: * components other than exactly (Ex, Ey, Ez): rel 3.80 at cosine -0.473, because
#:   the kernel contracts the component axis against a scalar permittivity;
#: * ``scaling_mode="continuous"``: a pure scale error at cosine 1.0 (the kernel now
#:   also divides the scale back out, so this is belt and braces);
#: * ``exact_interpolation=True``: co-located E, where the kernel needs raw Yee E;
#: * ``inverse=True``: never updated in a forward run, gradient exactly zero;
#: * a restricted ``switch`` or an ``apodization``: rel 7.9e-05 and 9.2e-01, both
#:   at cosine ~1.
#:
#: ``dtype`` is not listed: it follows the simulation precision (complex64 cost 12x
#: in accuracy on a float64 run).
DESIGN_DETECTOR_SETTINGS: dict[str, Any] = {
    "components": ("Ex", "Ey", "Ez"),
    "scaling_mode": "pulse",
    "exact_interpolation": False,
    "reduce_volume": False,
    "dft_subsample": 1,
    "apodization": None,
    "inverse": False,
    "switch": OnOffSwitch(),
    "plot": False,
}


def canonical_components(detector) -> tuple[str, ...]:
    """The detector's components in the order its phasor array actually stores them.

    ``PhasorDetector.update`` appends Ex, Ey, Ez, Hx, Hy, Hz in that fixed order
    whatever order ``components`` was given in, so ``components=("Hx", "Ez")``
    stores Ez at index 0. The adjoint current has to follow the stored order, not
    the declared one, or each cotangent drives the wrong field component.
    """
    declared = set(detector.components)
    return tuple(c for c in CANONICAL_COMPONENTS if c in declared)


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


def _narrow(slice_tuple, axis: int, side: str):
    """Return ``slice_tuple`` reduced to a one-cell face on ``axis``."""
    bounds = list(slice_tuple)
    lo, hi = bounds[axis]
    bounds[axis] = (lo, lo + 1) if side in ("-", "min") else (hi - 1, hi)
    return tuple(bounds)


def face_slice_tuple(detector, surface: ProjectionSurface):
    """Absolute grid slice of one box face, keeping its singleton normal axis.

    Mirrors the face slicing inside ``FieldProjectionDetectorBase.update``: the
    ``"-"`` face is the first cell along the normal axis and ``"+"`` the last.
    """
    axis, direction = _surface_axis_direction(surface)
    return _narrow(detector.grid_slice_tuple, axis, direction)


def detector_channels(detector, name: str) -> list[tuple[str, tuple]]:
    """State keys and absolute grid slices this detector needs adjoint currents for.

    Every supported objective detector stores complex phasors keyed either by a
    single ``"phasor"`` entry over its whole slice, or by one entry per face of a
    closed surface. Each entry needs its own adjoint current, because each is an
    independent linear functional of the fields.

    Recognised layouts:

    * one ``"phasor"`` key, covering :class:`PhasorDetector` itself and
      subclasses that only add a pure readout on top, such as
      ``PhasorPoyntingFluxDetector.compute_poynting_flux`` and
      ``ModeOverlapDetector``;
    * ``phasor_{axis}_{minus,plus}``, the box-mode field projection detectors;
    * ``phasor_axis{a}_{min,max}``, ``ClosedSurfacePhasorPoyntingFluxDetector``.

    Args:
        detector: a **placed** detector.
        name: its name, for error messages.

    Returns:
        ``[(state_key, absolute_grid_slice_tuple), ...]``.

    Raises:
        NotImplementedError: if the state layout is not one of the above, since
            guessing a face slice from an unknown key would silently put the
            adjoint current in the wrong place.
    """
    keys = sorted(detector._shape_dtype_single_time_step().keys())

    if keys == ["phasor"]:
        return [("phasor", detector.grid_slice_tuple)]

    if is_box_projection(detector):
        return [
            (_surface_state_key(surface), face_slice_tuple(detector, surface))
            for surface in detector._included_box_surfaces()
        ]

    # ClosedSurfacePhasorPoyntingFluxDetector: phasor_axis{a}_{min,max}
    if all(k.startswith("phasor_axis") for k in keys):
        channels = []
        for key in keys:
            body = key[len("phasor_axis") :]
            axis_text, _, side = body.partition("_")
            if side not in ("min", "max") or not axis_text.isdigit():
                raise NotImplementedError(f"detector {name!r}: cannot parse state key {key!r}")
            channels.append((key, _narrow(detector.grid_slice_tuple, int(axis_text), side)))
        return channels

    raise NotImplementedError(
        f"Detector {name!r} ({type(detector).__name__}) stores phasors under keys {keys}, which "
        "this transpose does not recognise, so there is no way to know which cells each key "
        "reads and where its adjoint current belongs. Record raw phasors with a PhasorDetector "
        "and do the post-processing in JAX on top of the returned phasors instead."
    )


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
        # Check this before touching _angular_frequencies, which only phasor
        # detectors have; otherwise a time-domain detector fails with an opaque
        # AttributeError instead of the explanation below.
        if not isinstance(detector, PhasorDetector):
            raise NotImplementedError(
                f"Detector {det_name!r} is a {type(detector).__name__}, which does not accumulate "
                "complex phasors, so there is no linear transpose to take and no adjoint current "
                "to place. Record phasors with a PhasorDetector and compute the quantity you want "
                "in JAX on top of them."
            )
        omegas = tuple(float(w) for w in detector._angular_frequencies)
        components = canonical_components(detector)
        made: list[str] = []
        for state_key, slice_tuple in detector_channels(detector, det_name):
            shape = tuple(hi - lo for lo, hi in slice_tuple)
            src_name = f"{name_prefix}{det_name}__{state_key}"
            # Placeholder values, replaced in the backward rule. The dtype follows
            # the simulation, so a float32 run does not warn about complex128.
            source = AdjointCurrentSource(
                name=src_name,
                amplitudes=jnp.zeros(
                    (len(omegas), len(components), *shape),
                    dtype=jnp.complex128 if config.dtype == jnp.float64 else jnp.complex64,
                ),
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


def design_regions(
    objects: ObjectContainer,
    design: str | Sequence[str] | None,
) -> list[SimulationObject]:
    """Resolve a ``design_detector`` argument to the objects whose cells are the design region.

    Args:
        objects: a placed container.
        design: ``None`` for every :class:`~fdtdx.objects.device.device.Device` in
            the scene, which is what a parameter optimization wants; otherwise the
            name, or names, of any placed objects. A ``Device``, a detector and a
            static material block all work: only the object's grid slice is used.

    Returns:
        The region objects, duplicates removed, in the order given.

    Raises:
        ValueError: if ``design`` is ``None`` and the scene has no ``Device``, if
            it is an empty sequence, or if a name is not in the container.
    """
    if design is None:
        found: list[SimulationObject] = list(objects.devices)
        if not found:
            raise ValueError(
                "design_detector was not given and the scene contains no Device, so there is no "
                "design region to take a gradient over. Pass the name of the object covering the "
                "design region (a Device, a detector, or a static material block)."
            )
        return found
    names = (design,) if isinstance(design, str) else tuple(design)
    if not names:
        raise ValueError("design_detector is an empty sequence; pass None to use every Device in the scene")
    regions: list[SimulationObject] = []
    for name in dict.fromkeys(names):
        regions.append(find_object(objects, name))
    return regions


def design_detector_name(region: SimulationObject) -> str:
    """Name of the internal design detector built over ``region``."""
    return f"{DESIGN_DETECTOR_PREFIX}{region.name}"


def make_design_detector(
    region: SimulationObject,
    wave_characters: Sequence[Any],
    config: SimulationConfig,
    key: jax.Array,
) -> PhasorDetector:
    """A freshly placed design-region detector over ``region``'s cells.

    Built from scratch rather than by rewriting a user's detector: every field it
    does not set is :class:`PhasorDetector`'s default, and every placement-derived
    cache (``_window_sum``, ``_dft_stride``, the on-mask, ...) comes from its own
    ``place_on_grid``. Rewriting an existing detector with ``aset`` leaves those
    caches stale, which with an apodization was a silent rel 0.92 gradient error.

    Args:
        region: a placed object; only its grid slice is used.
        wave_characters: the objective frequencies. The design detector must
            record exactly these, since the kernel pairs them index by index.
        config: the resolved config the solves run with.
        key: PRNG key for ``place_on_grid``.

    Returns:
        The placed detector.
    """
    slice_tuple = region.grid_slice_tuple
    det = PhasorDetector(
        name=design_detector_name(region),
        partial_grid_shape=region.grid_shape,
        wave_characters=tuple(wave_characters),
        dtype=jnp.complex128 if config.dtype == jnp.float64 else jnp.complex64,
        **DESIGN_DETECTOR_SETTINGS,
    )
    det = det.place_on_grid(grid_slice_tuple=slice_tuple, config=config, key=key)
    # Only read by post-processing under mirror symmetry; carried so the internal
    # detector describes the same cells as its region in every coordinate frame.
    unreduced = region._unreduced_grid_slice_tuple
    if unreduced != INVALID_SLICE_TUPLE_3D:
        det = det.aset("_unreduced_grid_slice_tuple", unreduced)
    return det


def internal_scene(
    objects: ObjectContainer,
    arrays: ArrayContainer,
    config: SimulationConfig,
    key: jax.Array,
    *,
    keep_detectors: Sequence[str],
    design: str | Sequence[str] | None,
    wave_characters: Sequence[Any],
) -> tuple[ObjectContainer, ArrayContainer, tuple[str, ...]]:
    """Private copy of a placed scene, set up for one of the two reciprocity solves.

    * Every detector not named in ``keep_detectors`` is dropped, together with its
      ``detector_states`` entry. Its state is never returned, so recording it in
      both solves would only cost time and memory. A stock six-component detector
      over the design region is the common case.
    * One design detector per design region is added, built by
      :func:`make_design_detector`, with a fresh ``detector_states`` entry.

    The caller's containers are not modified (``TreeClass`` is immutable), so the
    caller's own ``run_fdtd`` on the same scene still records everything they
    configured.

    Args:
        objects: a placed container.
        arrays: its arrays.
        config: the resolved config the solves run with.
        key: PRNG key for placement.
        keep_detectors: detectors to keep, i.e. the objective monitors in the
            forward solve and none in the adjoint solve.
        design: see :func:`design_regions`.
        wave_characters: the objective frequencies.

    Returns:
        ``(objects, arrays, design_detector_names)``.

    Raises:
        ValueError: see :func:`design_regions`; also if a scene object already
            uses an internal design-detector name.
    """
    keep = set(keep_detectors)
    regions = design_regions(objects, design)
    detectors = [make_design_detector(r, wave_characters, config, key) for r in regions]
    names = tuple(d.name for d in detectors)
    clash = [n for n in names if any(getattr(o, "name", None) == n for o in objects.object_list)]
    if clash:
        raise ValueError(f"Scene objects {clash} use names reserved for internal design detectors")

    volume = objects.volume
    new_list: list[SimulationObject] = []
    volume_idx = -1
    for obj in objects.object_list:
        if isinstance(obj, Detector) and obj.name not in keep:
            continue
        if obj is volume:
            volume_idx = len(new_list)
        new_list.append(obj)
    new_list.extend(detectors)
    if volume_idx < 0:  # pragma: no cover - a container always has a volume
        raise ValueError("container has no volume object to carry over")

    states = {name: state for name, state in arrays.detector_states.items() if name in keep}
    states.update({d.name: d.init_state() for d in detectors})
    return (
        ObjectContainer(object_list=new_list, volume_idx=volume_idx),
        arrays.aset("detector_states", states),
        names,
    )
