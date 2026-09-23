# Production path: parity with the official pipeline, and four corrections

Third in the series. `01-reciprocity-physics.md` is the theory,
`02-implementation.md` is the first implementation and its traps. This file
covers making it usable for real inverse design, and it **corrects four claims
in `02-implementation.md`**, three of which were restrictions that turned out
not to exist.

## Headline: it matches the official pipeline and is much faster

Gradient with respect to a real `fdtdx.Device`'s parameters, same placed scene,
`apply_params -> run_fdtd(GradientConfig(checkpointed)) -> jax.grad` versus
`reciprocity_param_fn -> jax.grad`:

| scene | pipeline | compile | run | speedup |
| --- | --- | --- | --- | --- |
| 24³, 512 params, float64 CPU | official | 4.99 s | 1.629 s | |
| | reciprocity | 0.95 s | 0.173 s | **9.4x** |
| 48³, 32768 params, float32 GPU | official | 5.15 s | 1.698 s | |
| | reciprocity | 0.81 s | **0.048 s** | **35.3x** |

In both cases the forward value is identical to the last digit, the parameter
gradient agrees at `rel_L2` of 1.1e-06 and 1.8e-06, cosine similarity is
`1.0000000000`, sign agreement is 100%, and the returned pytree structure
matches, so it is a drop-in for an optimizer.

## Correction 1: lossy design regions already worked

`02-implementation.md` listed "no lossy case tested" as a gap and I expected a
missing term, because FDTDX's lossy update seems to put `inv_eps` in two places:

    E_new = (1 - c*sigma*eta0*inv_eps/2) * E_old + c * curl * inv_eps

It factors:

    E_new = E_old + c*inv_eps*(curl - sigma*eta0*E_old/2)
    => dE_new/d(inv_eps) = (E_new - E_old)/inv_eps

which is the lossless expression unchanged: the loss is already inside the field
increment. Physically obvious afterwards, since in
`A = curl mu^-1 curl - w^2 eps + i w sigma` the conductivity term has no `eps`
dependence. Measured with the **unmodified** kernel:

| sigma (S/m) | deviation of the damping factor from 1 | rel L2 |
| --- | --- | --- |
| 1e3 | 5.4e-03 | 5.2e-08 |
| 1e4 | 5.4e-02 | 6.3e-09 |
| 1e5 | 5.4e-01 | 2.0e-08 |
| 3e5 | 1.62 | 3.4e-08 |

Loss in fact *improves* accuracy, because the fields decay faster and the DFT
converges better.

Watch the units: `initialization.py` scales physical conductivity by
`conductivity_spacing = c0*dt/courant`, the grid spacing. So sigma = 50 S/m is
2.5e-6 internally and perturbs the damping factor by 2.7e-4, far too weak to
test anything. A first attempt at this calibration was wasted on that.

## Correction 2: the source restriction is about freezing, not position

`02-implementation.md` said a source must not sit inside the design region, on
the evidence that a dipole there held the residual at 0.86 across 32 candidate
kernels. The position was not the cause. Sweeping source position and source
type:

| source cell | in design? | type | rel L2 |
| --- | --- | --- | --- |
| (10,10,10) | yes | `AdjointCurrentSource` | 4.0e-06 |
| (12,10,10) | yes | `AdjointCurrentSource` | 2.8e-06 |
| (10,10,10) | yes | `PointDipoleSource` | 1.16 |
| (10,10,10) | yes | `PointDipoleSource`, caches cleared | 1.9e-06 |
| (5,10,10) | no | `PointDipoleSource` | 1.4e-06 |

The rule is that the source must read `inv_permittivities` **live**. FDTDX
injects an impressed current as `E += -courant * inv_eps * J`, so a source in the
design region contributes there, but every stock source suppresses that term:
`PointDipoleSource` caches the factor in a private field during `apply()`, and
the TFSF plane sources wrap their injection in `jax.lax.stop_gradient`.
`AdjointCurrentSource` reads it live, which is why it is exact anywhere.

Clearing the caches makes our kernel self-consistent, but it then computes the
*physically complete* gradient, which deliberately differs from `run_fdtd`.
Since parity with the official pipeline is the contract, the factory
**refuses** an overlapping stock source and names both remedies rather than
silently picking a convention.

## Correction 3: source bandwidth is a real accuracy knob

Not previously noted, and it explained a residual I nearly misattributed twice.
Same scene, source outside the design region, only the Gaussian spectral width
varying:

| spectral width | rel L2 |
| --- | --- |
| 0.4 x f0 | 2.4e-03 |
| 0.1 x f0 | 2.5e-07 |
| 0.02 x f0 | 2.5e-07 |

At 0.4 x f0 the pulse spans about 2.5 optical cycles, and the single-frequency
reciprocity identity is a poor approximation across the DFT bin. Lengthening
the simulation does **not** fix it (6.9e-03 at 300 fs, 8.8e-03 at 600 fs), so it
is not truncation. Keep the source at or below roughly 0.1 x f0.

## Correction 4: two `place_objects` calls is a correctness bug

The original API took two hand-built scenes. That is not merely inconvenient,
it is wrong whenever a `Device` is present: `place_objects` splits its PRNG key
once per placed object before initializing device parameters, so two scenes
differing only in object count get different initial parameters. Measured on a
16³ scene with one 4³ device, the only difference being one extra source:

    1 source  -> device parameter sum 32.3474464417
    2 sources -> device parameter sum 27.7113399506

The forward and adjoint runs would have been simulating different structures,
and nothing downstream would have noticed. None of the earlier tests caught it
because none of them had a `Device`.

`fdtdx.adjoint.scene.derive_adjoint_objects` now derives the adjoint container
from the **placed** forward container: every `Source` dropped, one
`AdjointCurrentSource` placed on the objective monitor's own cells with
`place_on_grid`. Same grid, same materials, same device parameters by
construction. There is a regression test asserting the PRNG behaviour, so if
upstream ever fixes it we find out rather than carrying the workaround forever.

## The API

```python
from fdtdx.adjoint import reciprocity_param_fn

param_fn = reciprocity_param_fn(
    arrays, objects, config, key,
    objective_detectors="mon",     # the monitor(s) the FoM reads
)                                  # design region: every Device in the scene

loss = lambda p: my_fom(param_fn(p, beta=beta))   # any differentiable FoM
value, grad = jax.value_and_grad(loss)(params)    # grad is a ParameterContainer
```

`reciprocity_phasor_fn` is the same thing one level down, differentiating raw
`inv_permittivities`. `make_reciprocity_phasor_fn` remains as the two-scene
primitive for callers who genuinely want to control both scenes.

### The design region needs no detector

Earlier versions required a `PhasorDetector` over the design region, configured
exactly one way, and refused PhasorDetector's own defaults, because each default
was a silent error: all six components gave rel 3.80 at cosine -0.473,
`scaling_mode="continuous"` a pure scale error at cosine 1.0. That detector's
phasors never leave the VJP, so it is now internal. Both solves run on private
copies of the containers (`scene.internal_scene`) which

* build one design detector per design region with `scene.make_design_detector`,
  from scratch and freshly placed (`(Ex, Ey, Ez)`, pulse, raw Yee fields, every
  step, the objective's `wave_characters`, dtype following the simulation);
* drop every detector whose state is not read: all but the objectives in the
  forward solve, all of them in the adjoint solve.

`design_detector=None` (the default) means every `Device`. A name may be a
`Device`, a detector of any configuration or any other placed object: only its
cells are used. Several regions each get their own detector; where two overlap,
the gradient is written once, not summed. For `reciprocity_param_fn` the default
is exactly right, since `apply_params` writes only the Device cells; naming a
subset of the Devices zeroes the gradient of the others.

Measured on the 24³ float64 scene against `run_fdtd(checkpointed)`, parameter
gradient: stock-default design detector 9.1e-08, no design detector 1.06e-07,
two Devices 1.06e-07, monitor at PhasorDetector's default `continuous` scaling
with no design detector 3.2e-07 (complex64 monitor).

### Scales

`PhasorDetector` multiplies every sample by `_static_scale()`: 1 in pulse mode,
`2/sum(window)` in continuous mode. The objective monitor's scale `s_m` goes on
its own adjoint target, because JAX hands the VJP `dL/dP` and the adjoint current
must reproduce `dL/dP_raw = s_m dL/dP`; it is per detector, since two monitors in
different modes admit no common factor. The design scale rides on both the
forward and the adjoint phasors, so the gradient is divided by
`s_d_fwd * s_d_adj`. With the internal pulse detector that product is 1; it is
still computed from the placed detectors. All four monitor x design cells, a
mixed-mode pair of monitors, and a continuous internal design detector (forced by
monkeypatch) measure 1.169e-06, identical to the pulse/pulse baseline.

Extra keyword arguments to `param_fn` are forwarded to `apply_params`, so a
continuation schedule such as `beta=` stays live per optimizer step.

## Still open

* **Near-to-far.** The colour splitter's `FieldProjectionAngleDetector` is a
  box-mode detector with per-face state; only plain `PhasorDetector` is
  supported. The intended route is to record raw surface phasors and do the
  projection in JAX above the VJP boundary, which should work because the
  boundary sits at the raw phasors, but it is untested.
* **Design-dependent loss.** Meep's `MaterialGrid(damping=...)` makes sigma a
  function of the design variable, which is a genuinely new term. Constant loss
  is covered; this is not.
* **Dispersive design regions.** Untested.
* **`dft_subsample > 1`, `reduce_volume`, `exact_interpolation`** all still raise.
* **Not compared against Meep**, and not run on the RGB metalens.
* **`reversible` gradients contain NaN** in the PML at both precisions
  (46% of cells on a 32³ scene with a 6-cell PML). Unrelated to this work, but
  it is an upstream issue worth reporting.
