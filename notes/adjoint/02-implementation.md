# Implementing the reciprocity gradient: measurements and traps

Companion to `01-reciprocity-physics.md`, which is the theory. This file records
what the implementation actually needed, with the numbers that settled each
point. Everything here was measured against `checkpointed_fdtd` autodiff, which
is the exact derivative of the same discrete program. Nothing was taken on trust
from the derivation, and three of the eight items below contradicted it.

Files: `src/fdtdx/objects/sources/adjoint.py`,
`src/fdtdx/adjoint/reciprocity.py`, `src/fdtdx/adjoint/vjp.py`,
`tests/simulation/adjoint/test_reciprocity.py`.

## 1. JAX's cotangent already is the adjoint source

For a real loss differentiated through the DFT accumulation

    P[f, c, x] += s * w_n * exp(+i omega_f t_n) * E[c, x]

`jax.vjp` returns `Re[K^T ct]` with **no conjugate**. Verified to exactly zero
error against the closed form, and the conjugated variant is off by 100%.

More useful: the cotangent JAX hands a `custom_vjp` whose output is `P` equals
**exactly twice** the textbook `dF/dP` under the convention
`dF = 2 Re[(dF/dP)^T dP]`. Measured ratio `2 + 4.6e-17j`, for both a
random-weighted linear FoM and an intensity FoM.

So there is no Wirtinger bookkeeping to do, no conjugate to place and no factor
of two to remember. `jax.vjp` of `PhasorDetector.update` is the exact transpose
and carries `static_scale`, the apodization window, the component selection and
the H time-average split for free. Do not hand-write it.

## 2. Discrete reciprocity holds in FDTDX, including through CPML

This was an open question worth settling, because CPML is a numerical absorber
with auxiliary state rather than a physical medium, so its discrete transpose
symmetry is not guaranteed by the continuum argument.

Swapping source and monitor cells, same component:

| boundary | slab | relative difference |
| --- | --- | --- |
| periodic | no | `0.0` (bit-identical) |
| periodic | yes | `0.0` (bit-identical) |
| PML | no | `1.4e-15` |
| PML | yes | `1.2e-15` |

CPML preserves it to machine precision, so a 1e-5 gate is reachable with PML
present. `TestDiscreteReciprocity` keeps this honest.

## 3. The adjoint source must be a decaying pulse, not CW

Reciprocity is a frequency-domain identity, so it only holds once both DFTs have
converged. A constant-amplitude adjoint source is still radiating at the final
time step, its DFT never converges, and the reconstructed gradient is wrong by
order one.

Measured: with a CW adjoint source the relative residual sat at 0.86 and did not
improve from 120 fs to 200 fs. Windowing the excitation brought the adjoint
field's final-to-peak energy to `6e-8`.

This is what Meep's `FilteredSource` exists for. It is not an optimization.

## 4. A real sinusoid does not inject the amplitude you wrote

Injecting `Re[a exp(+i omega t)]` and accumulating with the detector's
`exp(+i omega t)` kernel gives a DFT of roughly `(T/2) conj(a)`, not `a`, plus
leakage between frequencies spaced closer than `1/T`.

So the amplitudes have to be solved for, not assigned.
`solve_adjoint_amplitudes` builds the exact square system: `nf` complex
constraints become `2 nf` real equations in `2 nf` real unknowns
(`alpha_f`, `beta_f`), with one matrix shared by every spatial site. At the RGB
metalens wavelengths (450/550/650 nm) the system is essentially perfectly
conditioned: `cond = 1.0001`, residual `1.9e-16`.

An over-short window is rejected with an exception rather than silently
producing meaningless amplitudes.

## 5. The kernel is discrete, and the continuum form is not merely inaccurate

    dL/d(inv_eps)[q] = Re[ sum_f K(omega_f) * sum_i Lambda_i[f,q] * F_i[f,q] ] / inv_eps[q]^2

    K(omega) = (exp(+i omega dt) - 1) / courant

Fitting a free complex constant to that expression returned
`-1.7492 - 0.0006j` times `(1 - exp(+i omega dt))` against a predicted
`-1/courant = -1.74954`, i.e. agreement to the run's truncation floor. The form
is therefore used with no fitted constant.

Substituting the textbook continuum `i omega` does not give a slightly worse
gradient, it gives the **wrong sign**: relative L2 `2.22` and cosine similarity
`-0.978` against autodiff. At the tested `omega dt` of 0.28 to 0.40 the discrete
factor is mandatory.

Note the power: `inv_eps^-2`, i.e. `eps^2`. A plausible-looking re-derivation
through an equivalent current gives `eps^1`; that one is wrong.

## 6. The design region must not contain a source

FDTDX sources scale their injected amplitude by the local `inv_eps`
(`PointDipoleSource.update_E`), which contributes a gradient term the
reciprocity kernel does not model.

This cost the most time, because the symptom does not look local. One offending
cell dominates the gradient norm, so with a dipole inside the design region the
relative residual stayed at **0.86 across all 32 candidate structures** — every
combination of conjugation pattern and permittivity power. Separating source,
design region and monitor dropped it to `3e-4` immediately, and the per-cell
ratios went to `1.000`.

The lesson generalizes: when a scan over plausible variants gives the same bad
answer for all of them, the fault is upstream of everything being scanned.

## 7. Accuracy is a convergence property, not a fixed tolerance

Reciprocity equals the exact discrete adjoint only up to DFT truncation, in both
runs, and the error is a product of two truncated transforms.

| sim time | steps | relative L2 vs checkpointed AD |
| --- | --- | --- |
| 100 fs | 1049 | `1.6e-04` |
| 200 fs | 2098 | `1.8e-04` |
| 400 fs | 4196 | `6.1e-06` (cosine `1.0000000000`) |

End to end through `jax.value_and_grad`, at 400 fs, on three unrelated figures
of merit:

| FoM | gradient relative L2 | cosine | value |
| --- | --- | --- | --- |
| random-weighted linear | `6.1e-06` | `1.0000000000` | exact |
| metalens `-abs(E)^2` | `1.7e-05` | `1.0000000000` | exact |
| log intensity ratio | `2.0e-05` | `0.9999999999` | exact |

So the test asserts that the error **falls** with decay time, not that it sits
below some number. A fixed tolerance would pass or fail for reasons unrelated to
correctness.

## 8. Process notes

`frozen_field` values live in the PyTreeDef and `objects` is a jit argument, so
the adjoint amplitudes and window are ordinary traced leaves. There is a test
asserting the tree structure is invariant when the amplitudes change; without
that guarantee every optimizer step would recompile the whole FDTD loop.

The adjoint `ObjectContainer` is rebuilt **inside** the VJP's backward rule.
Closing over it instead is what raises `UnexpectedTracerError`.

`ObjectContainer.sources` is a filtered view of `object_list`, so the source is
swapped by index in that list rather than through an `aset` path.

`jax.config.update("jax_enable_x64", True)` at test-module import leaks into the
whole pytest session and broke 9 unrelated tests in `test_recorder.py` and
`test_diffractive.py` by changing default dtypes. It is now an autouse fixture
that restores the previous value.

## What is not done yet

* `dft_subsample > 1`, `reduce_volume=True` and `exact_interpolation=True` all
  raise `NotImplementedError` rather than being approximated.
* Magnetic-component objectives: `AdjointCurrentSource` implements `update_H`,
  but the sign of the `B^T dF/dH` term from section 7 of the physics note has
  not been verified against finite differences on `inv_permeabilities`.
* The gradient is returned only on the design detector's region.
* No lossy or dispersive case has been tested. Both FDTDX paths agree in a real,
  lossless `eps`, where `A^T = A` and `A^dagger = A` coincide, so a conjugation
  error would currently be invisible. A complex-`eps` case should be added
  before anyone trusts this on absorbing materials.
* Not yet compared against Meep, and not yet run on the RGB metalens itself.
