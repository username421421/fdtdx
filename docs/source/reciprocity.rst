======================
Reciprocity gradients
======================

``GradientConfig(method="reciprocity")`` computes the gradient of a figure of merit on phasor
detectors from two plain forward solves: the forward run, and one adjoint run driven by currents
placed where the figure of merit reads the fields. Nothing is differentiated through the time loop,
so a gradient costs about three forward solves and the memory of one, whatever the run length.

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
  Poynting flux and closed-box net power. Monitors read together must record the same frequencies.
* Materials: isotropic and diagonally anisotropic permittivity, permeability and conductivity,
  lossy monitor and Device cells, lossy and etched Devices, dispersive static blocks (also under a
  Device), PML, periodic and PEC/PMC symmetry boundaries, float32 and float64, CPU and GPU.

The gradient is taken with respect to the Device parameters: it is exact there and zero outside
the Devices, also for ``jax.grad`` with respect to ``arrays.inv_permittivities`` itself. Only
``inv_permittivities`` and ``electric_conductivity`` carry it, which is everything
``apply_params`` writes from Device parameters; a parameter written by hand into another
material array gets a zero gradient, or raises where listed below.

Refused
=======

Each of these was measured to give a silently wrong gradient, so it raises instead; use
``method="checkpointed"`` for them:

* a figure of merit reading the fields, a time-domain detector (``EnergyDetector``,
  ``FieldDetector``, ``PoyntingFluxDetector``) or any other ``run_fdtd`` output;
* parameters reaching ``inv_permeabilities``, the magnetic conductivity or an object;
* objective detectors with an apodization, a switch skipping time steps, ``reduce_volume=True``,
  ``inverse=True``, a ``dft_subsample`` stride below 4 samples per period;
* grids whose cell width varies along an axis, nonzero Bloch vectors, full 3x3 material tensors,
  dispersive Device materials;
* a Device overlapping a PML or containing a stock source (:func:`fdtdx.reciprocity_param_fn`,
  which applies mode ports once at setup, also refuses a mode port inside a Device);
* a run too short to separate the objective frequencies (amplitude solve condition above 1e4).

Accuracy
========

Reciprocity equals automatic differentiation up to the truncation of the run's discrete Fourier
transforms, so let the fields leave the domain. A ``ConvergenceWarning`` reports phasors that have
not converged, and a ``PmlWarning`` an objective whose adjoint current reaches the lossy part of a
PML. The convergence estimate assumes the field left at the end does not oscillate near an
objective frequency; in a periodic cell, a diffraction order grazing near one rings without ever
reaching the PML and is not flagged (a 22% gradient error at a tail estimate of 5e-3 in a small
test cell). There, check convergence by rerunning with a longer simulation. Two more things
matter in practice:

* **The source spectrum.** A few-cycle Gaussian pulse carries a DC component whose static remainder
  never decays; in a structural-colour splitter it floored every gradient method, checkpointed
  too, at about 1%. A carrier phase that makes the pulse DC-free, or a narrower bandwidth,
  removes it.
* **Precision.** Optimize in float32 if you like; report final figures of merit from a float64
  evaluation.

Silence the warnings with ``warnings.filterwarnings("ignore", category=fdtdx.adjoint.ConvergenceWarning)``
(or ``PmlWarning``), or with ``tail_tolerance=None`` in the functional entry points.
