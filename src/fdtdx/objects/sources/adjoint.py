"""Impressed current with a windowed multi-sinusoid waveform, the adjoint source of reciprocity gradients.

It injects ``J[k](t_n) = window[n] * Re[sum_f amplitudes[f, k] exp(+i w_f t_n)]``, ``t_n = n dt``
for E and H alike (H's update at ``n + 1/2`` included), into field component ``components[k]``
with :class:`PointDipoleSource`'s injection law,
``E <- E - courant * inv_eps * J`` (dually for H), which keeps the discrete Green's
function symmetric between this source and a phasor detector. The ``window`` makes the
excitation a decaying pulse so its DFT converges;
:func:`fdtdx.adjoint.kernel.solve_adjoint_amplitudes` picks ``amplitudes`` so the windowed
current has a requested DFT.
"""

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, field, frozen_field
from fdtdx.objects.sources.source import Source

#: Detector component name -> (field array, component axis), in the order PhasorDetector stores them.
COMPONENT_MAP: dict[str, tuple[str, int]] = {
    "Ex": ("E", 0),
    "Ey": ("E", 1),
    "Ez": ("E", 2),
    "Hx": ("H", 0),
    "Hy": ("H", 1),
    "Hz": ("H", 2),
}


@autoinit
class AdjointCurrentSource(Source):
    """Impressed current ``window * Re[sum_f amplitudes exp(+i w_f t)]`` on the source's cells.

    ``amplitudes`` and ``window`` are traced leaves, not frozen fields: frozen values live
    in the PyTreeDef, so changing them would recompile the FDTD loop on every step.
    """

    #: Complex amplitudes, shape ``(num_frequencies, num_components, *grid_shape)``.
    amplitudes: jax.Array = field()

    #: Real envelope sampled per time step, shape ``(time_steps_total,)``.
    window: jax.Array = field()

    #: Angular frequencies in rad/s.
    angular_frequencies: tuple[float, ...] = frozen_field()

    #: Driven field components, ordered like ``amplitudes``' second axis.
    components: tuple[str, ...] = frozen_field()

    def __post_init__(self):
        bad = [c for c in self.components if c not in COMPONENT_MAP]
        if bad:
            raise ValueError(f"Unknown field components {bad}; expected keys of {sorted(COMPONENT_MAP)}")
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"Duplicate components: {self.components}")
        if self.amplitudes.ndim < 2:
            raise ValueError(
                "amplitudes must have shape (num_frequencies, num_components, *grid_shape), "
                f"got ndim={self.amplitudes.ndim}"
            )
        nf, nc = self.amplitudes.shape[0], self.amplitudes.shape[1]
        if nf != len(self.angular_frequencies):
            raise ValueError(f"amplitudes.shape[0]={nf} != len(angular_frequencies)={len(self.angular_frequencies)}")
        if nc != len(self.components):
            raise ValueError(f"amplitudes.shape[1]={nc} != len(components)={len(self.components)}")
        if self.window.ndim != 1:
            raise ValueError(f"window must be one-dimensional, got shape {self.window.shape}")

    def _waveform(self, time_step: jax.Array, components: tuple[int, ...]) -> jax.Array:
        """Current at ``time_step`` for the component indices ``components``, ``(len(components), *grid)``."""
        # the simulation's precision, not forced float64, so a float32 run does not downcast
        omega = jnp.asarray(self.angular_frequencies)
        # integer time also on update_H's time_step + 0.5, so the current is exactly the one the
        # amplitude solve models; the magnetic half steps are in fdtdx.adjoint.objective.target_factor
        step = jnp.floor(time_step)
        phase = jnp.exp(1j * omega * (step * self._config.time_step_duration))
        lo, hi = components[0], components[-1] + 1
        # contiguous (the canonical E-then-H order) is a plain slice, else a gather
        if tuple(range(lo, hi)) == components:
            amplitudes = self.amplitudes[:, lo:hi]
        else:
            amplitudes = self.amplitudes[:, components, ...]
        acc = jnp.tensordot(phase, amplitudes, axes=((0,), (0,)))
        index = jnp.clip(step.astype(jnp.int32), 0, self.window.shape[0] - 1)
        return jnp.real(acc) * self.window[index]

    def _inject(
        self,
        arr: jax.Array,
        inv_material: jax.Array | float,
        which: str,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        active = [(k, COMPONENT_MAP[c][1]) for k, c in enumerate(self.components) if COMPONENT_MAP[c][0] == which]
        if not active:
            return arr
        waveform = self._waveform(time_step, tuple(k for k, _ in active))
        sign = -1.0 if not inverse else 1.0
        gs = self.grid_slice
        if isinstance(inv_material, jax.Array) and inv_material.ndim > 0:
            inv_local: jax.Array | float = inv_material[:, *gs]
        else:
            inv_local = inv_material
        # one update of the field array per family: an indexed add per component copied
        # the whole array inside the time loop
        update = jnp.zeros((arr.shape[0], *waveform.shape[1:]), dtype=arr.dtype)
        for j, (_k, axis) in enumerate(active):
            if isinstance(inv_local, jax.Array) and inv_local.ndim > 0:
                # (1, ...) when isotropic, (3, ...) when diagonally anisotropic
                factor = inv_local[axis] if inv_local.shape[0] > 1 else inv_local[0]
            else:
                factor = inv_local
            injection = sign * self._config.courant_number * factor * waveform[j]
            update = update.at[axis].set(injection.astype(arr.dtype))
        return arr.at[:, *gs].add(update)

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permeabilities
        return self._inject(E, inv_permittivities, "E", time_step, inverse)

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities
        return self._inject(H, inv_permeabilities, "H", time_step, inverse)
