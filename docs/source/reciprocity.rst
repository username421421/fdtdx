======================
Reciprocity gradients
======================

``GradientConfig(method="reciprocity")`` computes the gradient of a figure of merit on phasor
detectors from two plain forward solves: the forward run, and one adjoint run driven by currents
placed where the figure of merit reads the fields. Nothing is differentiated through the time loop,
so a gradient costs about three forward solves and the memory of one, whatever the run length.

It fails fast. Wherever it cannot return ``GradientConfig(method="checkpointed")``'s gradient, it
raises an exception that names the reason and points to ``method="checkpointed"``; it never
returns a partial or approximate gradient silently, and never substitutes another method.

Using it
========

An existing inverse-design script changes one string:

.. code-block:: python

   config = config.aset("gradient_config", fdtdx.GradientConfig(method="reciprocity"))

   def loss(params):
       arrays_, objects_, _ = fdtdx.apply_params(arrays, objects, params, key)
       _, out = fdtdx.run_fdtd(arrays_, objects_, config, key)
       return -jnp.sum(jnp.abs(out.detector_states["mon"]["phasor"]) ** 2)

   value, grad = jax.value_and_grad(loss)(params)

The forward values are ``run_fdtd``'s, bit for bit, and the gradient is that of
``GradientConfig(method="checkpointed")`` once the fields have decayed. Only the phasor detectors
the figure of merit reads get adjoint currents; several of them share one adjoint solve.
:func:`fdtdx.reciprocity_param_fn` is the same gradient as a function of the Device parameters, and
:func:`fdtdx.reciprocity_phasor_fn` one level down, of ``inv_permittivities``.

What the figure of merit may read
=================================

* ``PhasorDetector`` states at any stock setting (E and/or H components, either ``scaling_mode``,
  ``exact_interpolation``, ``dft_subsample``), and anything computed from them in JAX:
  ``ModeOverlapDetector`` overlaps, a box-mode ``FieldProjectionAngleDetector`` (near-to-far),
  Poynting flux and closed-box net power. Monitors read together may record different frequencies;
  one adjoint solve drives the union of them. List each monitor once (a name given twice is refused)
  and read it as often as you like.
* Materials: isotropic and diagonally anisotropic permittivity, permeability and conductivity,
  lossy monitor and Device cells, lossy and etched Devices, dispersive static blocks (also under a
  Device), PML, periodic and PEC/PMC symmetry boundaries, float32 and float64, CPU and GPU.
* Sources: any stock source outside the Devices, a TFSF box (``TFSFPlaneSourceRegion``) around the
  Devices included, whose waveform has ended before the run does.

The gradient is taken with respect to the Device parameters, through ``apply_params``, in the same
traced function as ``run_fdtd``. It is computed inside the Devices, which is exact for Device
parameters; a differentiated material array ``apply_params`` did not write (``jax.grad`` with
respect to ``arrays.inv_permittivities`` itself, a background parameter, a blur after
``apply_params``) raises, because its sensitivity outside the Devices would be dropped. The check
follows the arrays ``apply_params`` returns, so it also refuses some exact uses: ``apply_params``
under a ``jax.jit`` of its own, or its arrays passed through a ``lax.scan`` or ``lax.cond`` carry or
an ``astype`` before ``run_fdtd``. Call ``apply_params`` directly in the differentiated function
(``jax.jit`` around the whole of it is fine).

Refused
=======

Each of these was measured to give a silently wrong gradient, so it raises instead; use
``method="checkpointed"`` for them:

* a figure of merit reading the fields, a time-domain detector (``EnergyDetector``,
  ``FieldDetector``, ``PoyntingFluxDetector``) or any other ``run_fdtd`` output;
* differentiated inputs other than the ``inv_permittivities`` and ``electric_conductivity``
  ``apply_params`` writes: ``inv_permeabilities``, the magnetic conductivity, the dispersion
  coefficients, an object's fields (a source amplitude), or a material array not from
  ``apply_params``; forward-mode differentiation (``jax.jvp``, ``jacfwd``; under ``jax.jit`` JAX's own
  ``TypeError``). Second derivatives work forward over reverse (``jax.hessian``,
  ``jax.jvp(jax.grad(f))``); reverse over reverse (``jax.jacrev(jax.jacrev(f))``) fails in FDTDX's
  time loop for every method;
* objective detectors with an apodization, a switch skipping time steps, ``reduce_volume=True``
  or ``inverse=True``, or whose cells reach a PML beyond its zero-loss first cell (crop full
  cross-section monitors to the interior);
* a ``dft_subsample`` stride on an objective detector where a source has spectrum at the
  frequencies the stride folds onto an objective frequency (``2 pi m / (k dt) +- w``), which can
  reverse the gradient; record every step there;
* a source still injecting at the end of the run (FDTDX's default ``SingleFrequencyProfile``, or a
  pulse the run does not outlast);
* grids whose cell width varies along an axis, nonzero Bloch vectors, full 3x3 material tensors,
  dispersive Device materials;
* a Device overlapping a PML or containing a stock source (:func:`fdtdx.reciprocity_param_fn`,
  which applies mode ports once at setup, also refuses a mode port inside a Device);
* a run too short to separate the objective frequencies (amplitude solve condition above 1e4);
* a ``Recorder`` in the ``GradientConfig`` (it is used by ``method="reversible"`` only).

Convergence
===========

Reciprocity computes the gradient of the converged figure of merit from the run's discrete Fourier
transforms, so the fields must have left the domain by the end of the run. Every gradient is checked
on every call, and raises when either of two estimates exceeds ``tail_tolerance``
(``GradientConfig``, default ``1e-2``; also an argument of the functional entry points, whose
``diagnostics`` hold the latest gradient's estimates):

* **the objective phasors' truncation** over the channels and frequencies the figure of merit reads.
  Above the tolerance the figure of merit itself is not the converged one, and neither is any
  method's gradient of it;
* **the gradient's distance from the converged gradient**: the truncation of the forward and adjoint
  design-region phasors it pairs, cell by cell, over the frequencies the figure of merit reads, plus
  the objective's, which shifts the adjoint currents as much. A truncation is estimated from the
  field left at the end and from the growth of the phasors over the last three eighths of the run,
  fitted with two decaying modes over the cells and continued (each solve runs in four segments with
  snapshots there, which costs nothing per time step). Measured against runs long enough to
  converge, the gradient's estimate read 1.2x to 4.3x above its true error: it errs on the side of
  refusing.

``tail_tolerance=None`` switches both checks, and the check of sources still injecting at the end, off.
Under ``jax.jit`` the refusal surfaces as a ``JaxRuntimeError`` whose text carries the message
(``equinox.filter_jit`` shows it directly). What usually makes it raise:

* **A run too short**, or a resonance (a cavity, a high-index Device, a grazing diffraction order in
  a periodic cell, a mode between two parallel walls) still ringing at the end. Lengthen the run.
* **A field still arriving**: a reflection from far away, or a pulse's front, reaching a monitor or a
  Device in the last three eighths of the run. How large it will get cannot be told from the run, so it is
  refused, even when it is only the PML's own weak reflection coming back: a 10-30% longer run, or a
  thicker PML, lets it pass.
* **A source carrying DC.** A few-cycle Gaussian pulse has a zero-frequency part; where its current
  ends inside the domain it leaves a static charge whose field never decays, and no run length
  removes it. Make the carrier DC-free: for a ``GaussianPulseProfile``, total carrier phase (the
  source's ``wave_character.phase_shift`` plus the profile's ``center_wave.phase_shift``)
  ``pi/2 - 2 pi f0 t0``, with ``t0 = 6 sigma_t`` and ``sigma_t = 1 / (2 pi spectral_width)``; the
  profile needs its own ``WaveCharacter`` when the source's is shared. Narrowing the bandwidth also
  shrinks it.

The estimates are estimates, not bounds. A weak resonance whose Q is far beyond the run, hidden under
faster modes in the late windows, is not seen (0.03 to 0.05 of its tail in synthetic tests), nor are two
comparably slow standing modes (0.23): the figure of merit is then equally unconverged in every method,
with no warning from any. The checks bound the phasors' relative truncation, not the figure of merit's:
one near a target (least squares, an equality penalty) amplifies it by about ``|P| / |P - P0|``, which no
check inside the gradient sees. Compare the per-channel tails in the diagnostics with that distance.
``method="checkpointed"`` differentiates the truncated run
exactly instead, and where a mode rings at another frequency than the objective's, that exact
gradient converges far more slowly than the figure of merit does: the mode's phase at the end of the
run depends on the design, and its derivative grows with the run length. Distances from the
converged gradient, measured against runs long enough to converge:

================================================  ============  ============  ============
scene, run length                                 reciprocity   its estimate  checkpointed
================================================  ============  ============  ============
Fabry-Perot cavity, off resonance, 3000 fs        2.5e-4        8.1e-4        0.19
the same, 4000 fs                                 4.3e-5        7.3e-5        2.1e-2
eps-12 Device in air, 200 fs                      3.2e-3        1.1e-2        9.5e-3
periodic slab with a guided mode, 8000 fs         1.2e-3 [1]    4.6e-3        0.23 [2]
================================================  ============  ============  ============

[1] its change from 8000 to 12000 fs; [2] from checkpointed at 12000 fs, which is not converged
either: a guided mode that does not radiate rings for ever, and checkpointed moved by 0.2 to 0.96
between run lengths from 300 fs to 12 ps while the figure of merit converged to 1e-5. So the two
methods can disagree by tens of percent with both correct: checkpointed about the run, reciprocity
about its converged limit. On a structural-colour splitter's grey starting design (DC-free pulse,
160 fs) they differed by 1.1% while reciprocity moved 1.5e-4 when the run was doubled and
checkpointed 3e-3. Compare the two only at run lengths where each has converged. Optimize in float32
if you like; report final figures of merit from a float64 evaluation.
