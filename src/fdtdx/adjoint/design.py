"""Design side: the design regions, the internal detector recording them, and the scene both solves run on.

The design-region detector is internal scratch (its phasors never leave the VJP), so it is
built here, fresh, in the one configuration the gradient kernel is calibrated for. Both
solves run on private copies of the placed scene: the caller's containers are untouched,
and the adjoint scene is derived from the placed forward one rather than placed again,
because ``place_objects`` splits its key once per object and a second call with a
different object count would initialize different Device parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax

from fdtdx.adjoint.objective import complex_dtype
from fdtdx.adjoint.validation import as_names, check_applied_outside_devices
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.default_key import default_key
from fdtdx.core.switch import OnOffSwitch
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.objects.detectors.detector import Detector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.object import INVALID_SLICE_TUPLE_3D, SimulationObject

#: Name prefix of the internal design-region detectors.
DESIGN_DETECTOR_PREFIX = "__adjoint_design__"

#: The configuration the gradient kernel is calibrated for: raw Yee E (the kernel contracts
#: the component axis against a scalar permittivity), pulse scale, every step, no apodization.
#: Each setting was a silent error while the detector was user-built (notes/adjoint/03-production.md).
#: ``dtype`` follows the simulation.
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


def design_regions(objects: ObjectContainer, design: str | Sequence[str] | None) -> list[SimulationObject]:
    """The objects whose cells form the design region.

    ``None`` is every Device; otherwise the names of any placed objects (a Device, a detector,
    a static block: only the cells are used), duplicates removed, in the order given.
    """
    if design is None:
        found: list[SimulationObject] = list(objects.devices)
        if not found:
            raise ValueError(
                "design_detector was not given and the scene contains no Device, so there is no design region. "
                "Pass the name of the object covering it (a Device, a detector or a static material block)."
            )
        return found
    names = as_names(design)
    if not names:
        raise ValueError("design_detector is an empty sequence; pass None to use every Device in the scene")
    return [objects[name] for name in dict.fromkeys(names)]


def make_design_detector(
    region: SimulationObject,
    wave_characters: Sequence[Any],
    config: SimulationConfig,
    key: jax.Array,
) -> PhasorDetector:
    """A freshly placed design detector over ``region``'s cells, recording the objective's frequencies.

    Built from scratch, never by ``aset`` on an existing detector, which would leave the
    ``place_on_grid`` caches (window sum, stride, on-mask) stale.
    """
    det = PhasorDetector(
        name=f"{DESIGN_DETECTOR_PREFIX}{region.name}",
        partial_grid_shape=region.grid_shape,
        wave_characters=tuple(wave_characters),
        dtype=complex_dtype(config),
        **DESIGN_DETECTOR_SETTINGS,
    )
    det = det.place_on_grid(grid_slice_tuple=region.grid_slice_tuple, config=config, key=key)
    # only read by post-processing under mirror symmetry; keeps the region's cells in every frame
    unreduced = region._unreduced_grid_slice_tuple
    if unreduced != INVALID_SLICE_TUPLE_3D:
        det = det.aset("_unreduced_grid_slice_tuple", unreduced)
    wrong = {k: getattr(det, k) for k, v in DESIGN_DETECTOR_SETTINGS.items() if k != "switch" and getattr(det, k) != v}
    if det._dft_stride != 1 or det._num_time_steps_on != int(config.time_steps_total):
        wrong.update(_dft_stride=det._dft_stride, _num_time_steps_on=det._num_time_steps_on)
    if wrong:  # pragma: no cover - a PhasorDetector upstream change, not a user error
        raise RuntimeError(f"internal design detector {det.name!r} is misconfigured: {wrong}")
    return det


def internal_scene(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    *,
    keep_detectors: Sequence[str],
    design: str | Sequence[str] | None,
    wave_characters: Sequence[Any],
) -> tuple[ArrayContainer, ObjectContainer, tuple[str, ...]]:
    """Private copy of a placed scene for one solve: ``(arrays, objects, design_detector_names)``.

    Detectors not in ``keep_detectors`` (the objectives in the forward solve, none in the
    adjoint one) are dropped with their state, and one design detector per region of
    ``design`` (see :func:`design_regions`) is added.
    """
    keep = set(keep_detectors)
    detectors = [make_design_detector(r, wave_characters, config, key) for r in design_regions(objects, design)]
    names = tuple(d.name for d in detectors)
    clash = [n for n in names if any(getattr(o, "name", None) == n for o in objects.object_list)]
    if clash:
        raise ValueError(f"Scene objects {clash} use names reserved for internal design detectors")
    kept = [o for o in objects.object_list if not (isinstance(o, Detector) and o.name not in keep)]
    states = {name: state for name, state in arrays.detector_states.items() if name in keep}
    states.update({d.name: d.init_state() for d in detectors})
    volume_idx = next(i for i, o in enumerate(kept) if o is objects.volume)
    return (
        arrays.aset("detector_states", states),
        ObjectContainer(object_list=[*kept, *detectors], volume_idx=volume_idx),
        names,
    )


def apply_objects_once(arrays: ArrayContainer, objects: ObjectContainer, key: jax.Array | None) -> ObjectContainer:
    """Apply, once, the objects ``apply_params`` re-applies on every call; returns a new container.

    ``place_objects`` leaves every object whose projection overlaps a Device's unapplied (a
    plane source above a design, a mode port on its periodic span). This runs
    ``apply_params``' object loop without the parameters: same objects, order, key split and
    ``stop_gradient`` materials (``initial_inv_permittivities`` when set). That is exact for an
    object whose own cells lie outside every Device; one inside is refused.
    """
    check_applied_outside_devices(objects)
    key = default_key(key)
    inv_eps = arrays.initial_inv_permittivities
    if inv_eps is None:
        inv_eps = arrays.inv_permittivities

    def frozen(value):
        return None if value is None else jax.lax.stop_gradient(value)

    new_list = []
    for obj in objects.object_list:
        if any(dev.check_overlap(obj) for dev in objects.devices):
            key, subkey = jax.random.split(key)
            obj = obj.apply(
                key=subkey,
                inv_permittivities=jax.lax.stop_gradient(inv_eps),
                inv_permeabilities=jax.lax.stop_gradient(arrays.inv_permeabilities),
                dispersive_c1=frozen(arrays.dispersive_c1),
                dispersive_c2=frozen(arrays.dispersive_c2),
                dispersive_c3=frozen(arrays.dispersive_c3),
                electric_conductivity=frozen(arrays.electric_conductivity),
            )
        new_list.append(obj)
    return ObjectContainer(object_list=new_list, volume_idx=objects.volume_idx)


def device_dispersion_as_applied(arrays: ArrayContainer, objects: ObjectContainer) -> ArrayContainer:
    """Zero the ADE coefficients in the Device cells, as ``apply_params`` does on every call.

    Device materials are non-dispersive (refused otherwise), so what ``apply_params`` writes
    there is zero whatever the parameters; ``place_objects`` instead leaves the coefficients
    of a dispersive block under a Device, which the solves would otherwise keep simulating.
    """
    if arrays.dispersive_c1 is None:
        return arrays
    for name in ("dispersive_c1", "dispersive_c2", "dispersive_c3"):
        value = getattr(arrays, name)
        if value is None:
            continue
        for device in objects.devices:
            value = value.at[:, :, *device.grid_slice].set(0.0)
        arrays = arrays.aset(name, value)
    return arrays
