"""Design side: the internal detector over each Device, the scene each solve runs on, the late windows.

The design detector is internal scratch (its phasors never leave the gradient), so it is built
here, fresh, in the one configuration the gradient kernel is calibrated for. Both solves run on
private copies of the placed scene: the caller's containers are untouched, and the adjoint scene
is derived from the forward one rather than placed again, because ``place_objects`` splits its
key once per object and a different object count would initialize different Device parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax

from fdtdx.adjoint.objective import complex_dtype
from fdtdx.config import SimulationConfig
from fdtdx.core.switch import OnOffSwitch
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer
from fdtdx.objects.detectors.detector import Detector
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.objects.object import INVALID_SLICE_TUPLE_3D, SimulationObject

#: Name prefix of the internal design detectors.
DESIGN_DETECTOR_PREFIX = "__adjoint_design__"

#: The configuration the gradient kernel is calibrated for: raw Yee E (the kernel contracts the
#: component axis against a scalar permittivity), pulse scale, every step, no apodization. Each
#: setting was a silent error while the detector was user-built (notes/adjoint/03-production.md).
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

#: Share of the run each of the three late windows spans (the last three eighths).
LATE_WINDOW = 0.125


def make_design_detector(
    region: SimulationObject,
    wave_characters: Sequence[Any],
    config: SimulationConfig,
    key: jax.Array,
) -> PhasorDetector:
    """A freshly placed design detector over ``region``'s cells, recording ``wave_characters``.

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


def late_windows(config: SimulationConfig) -> tuple[int, int, int]:
    """The time steps at which the three late windows start (each ``LATE_WINDOW`` long), oldest first.

    Every solve runs in four segments split there (:func:`fdtdx.adjoint.solve.segmented_solve`),
    so the convergence estimate reads the phasors' growth over the windows from snapshots, without
    an extra detector or a per-step conditional.
    """
    n = int(config.time_steps_total)
    w = max(1, int(LATE_WINDOW * n))
    return max(n - 3 * w, 0), max(n - 2 * w, 0), n - w


def internal_scene(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    *,
    keep_detectors: Sequence[str],
    wave_characters: Sequence[Any],
) -> tuple[ArrayContainer, ObjectContainer, tuple[str, ...]]:
    """Private copy of a placed scene for one solve: ``(arrays, objects, design_detector_names)``.

    Detectors not in ``keep_detectors`` are dropped with their state, and one design detector per
    Device is added.
    """
    keep = set(keep_detectors)
    detectors = [make_design_detector(d, wave_characters, config, key) for d in objects.devices]
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
