"""Adjoint current source for reciprocity-based gradients.

Realizes the adjoint field of a :class:`PhasorDetector` objective as an ordinary
forward-in-time FDTD run, so a gradient costs two forward solves instead of
differentiating through the time loop.

Waveform
--------
The source injects

    J[k, x](t_n) = window[n] * Re[ sum_f amplitudes[f, k, x] * exp(+i omega_f t_n) ]

into the field component named by ``components[k]``.

The ``window`` is not cosmetic. Reciprocity is a frequency-domain identity, so
it only holds once both the forward and the adjoint DFT have converged. A
constant-amplitude adjoint source is still radiating at the last time step, its
DFT never converges, and the reconstructed gradient is wrong by order one -- this
was measured, not assumed. The window makes the adjoint excitation a decaying
pulse, and :func:`fdtdx.adjoint.reciprocity.solve_adjoint_amplitudes` then picks
``amplitudes`` so that the *windowed* current still has exactly the requested
DFT at every objective frequency. That is the same problem Meep solves with
``FilteredSource``.

Normalization
-------------
The injection law is :class:`PointDipoleSource`'s, verbatim:

    E[c, x] <- E[c, x] - courant * inv_eps[c, x] * J[c, x](t_n)

with the dual on the magnetic side. Keeping FDTDX's own law is what makes the
discrete Green's function symmetric between this source and the detector the
adjoint amplitudes were derived from.
"""

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, field, frozen_field
from fdtdx.objects.sources.source import Source

#: Detector component name -> (field array, component axis).
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
    """Impressed current with a windowed multi-sinusoid waveform.

    ``amplitudes`` and ``window`` are ordinary traced pytree leaves, not frozen
    fields. That is deliberate: frozen values live in the PyTreeDef and
    ``objects`` is a jit argument, so freezing them would recompile the whole
    FDTD loop on every optimizer step.
    """

    #: Complex amplitudes, shape ``(num_frequencies, num_components, *grid_shape)``.
    amplitudes: jax.Array = field()

    #: Real envelope sampled per time step, shape ``(time_steps_total,)``.
    window: jax.Array = field()

    #: Angular frequencies in rad/s. Frozen: structural, fixed by the detector.
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

    def _waveform(self, time_step: jax.Array, components: tuple[int, ...] | None = None) -> jax.Array:
        """Injected current at ``time_step``, shape ``(num_components, *grid_shape)``.

        ``components`` restricts it to those indices of :attr:`components`, in that
        order. ``update_E`` and ``update_H`` each need only their own family, and the
        frequency contraction is the injection's whole cost.
        """
        dt = self._config.time_step_duration
        # No explicit float64: under x64 this is float64 and under float32 runs it
        # is float32, which matches whatever precision PhasorDetector.update uses.
        # Forcing float64 here would warn and silently downcast on a float32 GPU run.
        omega = jnp.asarray(self.angular_frequencies)
        t = time_step * dt
        phase = jnp.exp(1j * omega * t)
        amplitudes = self.amplitudes
        if components is not None:
            lo, hi = components[0], components[-1] + 1
            # contiguous (the canonical E-then-H order) is a plain slice, else a gather
            amplitudes = amplitudes[:, lo:hi] if tuple(range(lo, hi)) == components else amplitudes[:, components, ...]
        acc = jnp.tensordot(phase, amplitudes, axes=((0,), (0,)))
        # update_H is called with ``time_step + 0.5`` (fdtd/update.py:813), the Yee
        # half-step, so ``time_step`` is not always an integer. The carrier phase
        # above uses that exact half-integer time, which is what makes the magnetic
        # injection land on the right half-step; only the envelope lookup needs an
        # integer, and the envelope varies slowly enough that flooring it is
        # negligible against the carrier.
        index = jnp.clip(jnp.floor(time_step).astype(jnp.int32), 0, self.window.shape[0] - 1)
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
        c_courant = self._config.courant_number
        sign = -1.0 if not inverse else 1.0
        gs = self.grid_slice

        if isinstance(inv_material, jax.Array) and inv_material.ndim > 0:
            inv_local: jax.Array | float = inv_material[:, *gs]
        else:
            inv_local = inv_material

        # One update of the field array per family, not one per component: inside the
        # time loop each indexed add can cost a copy of the whole array, which made
        # the currents of a five-face box cost three bare solves. Absent components
        # get an exact zero.
        update = jnp.zeros((arr.shape[0], *waveform.shape[1:]), dtype=arr.dtype)
        for j, (_k, axis) in enumerate(active):
            if isinstance(inv_local, jax.Array) and inv_local.ndim > 0:
                # inv_permittivities is (1, ...) when isotropic and (3, ...) when
                # diagonally anisotropic; index the component only when present.
                factor = inv_local[axis] if inv_local.shape[0] > 1 else inv_local[0]
            else:
                factor = inv_local
            injection = sign * c_courant * factor * waveform[j]
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
