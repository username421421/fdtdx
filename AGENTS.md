# AGENTS.md

## Read this before touching anything

This is a **development fork of FDTDX**, not a working install and not a
consumer of FDTDX. There are three separate FDTDX-related trees on this machine
and they must not be confused:

| Tree | What it is | Interpreter | May an agent edit it? |
| --- | --- | --- | --- |
| `/home/zhuwei/fdtdx-dev/fdtdx` | **this repo.** Fork of `ymahlau/fdtdx`, branch `reciprocity-adjoint` | `/home/zhuwei/fdtdx-dev/fdtdx/.venv/bin/python` | **Yes.** This is the work. |
| `/home/zhuwei/miniconda3/envs/fdtdx` | FDTDX 0.6.2 installed non-editable, pinned at commit `f4e610c3`. Backs the colour-splitter and NeuroShaper campaigns. | `/home/zhuwei/miniconda3/envs/fdtdx/bin/python` | **No.** Installing anything here silently changes published campaign results. |
| `/home/zhuwei/fdtdx/color_splitter_d4` | Research project that *uses* FDTDX 0.6.2. Has its own `AGENTS.md`; follow it there. | the pinned env above | Only when the task is about that project. |

There is also `/home/zhuwei/miniconda3/envs/mp` with Meep 1.34.0, used as a
cross-solver reference. Never install FDTDX into it.

If you are here because a task said "optimize FDTDX" or "work on the adjoint
method", you are in the right place. If a task mentions the colour splitter, the
D4 optimizer, `balanced_480`, NeuroShaper or effective rank, you are in the
wrong tree.

## Status

The reciprocity gradient is **implemented, validated against FDTDX's own
pipeline, and usable for inverse design**.

```python
from fdtdx.adjoint import reciprocity_param_fn

param_fn = reciprocity_param_fn(arrays, objects, config, key,
                                objective_detector="mon",
                                design_detector="design_dft")
value, grad = jax.value_and_grad(lambda p: my_fom(param_fn(p, beta=beta)))(params)
```

Parity with `apply_params -> run_fdtd(GradientConfig(checkpointed)) -> jax.grad`,
same placed scene, gradient with respect to real `Device` parameters:

| scene | official | reciprocity | speedup | gradient rel L2 |
| --- | --- | --- | --- | --- |
| 24³, 512 params, float64 CPU | 1.629 s | 0.173 s | 9.4x | 1.1e-06 |
| 48³, 32768 params, float32 GPU | 1.698 s | **0.048 s** | **35.3x** | 1.8e-06 |

Forward values identical to the last digit, cosine similarity 1.0000000000,
100% sign agreement, matching pytree structure.

Files:

* `src/fdtdx/objects/sources/adjoint.py` — `AdjointCurrentSource`
* `src/fdtdx/adjoint/reciprocity.py` — window, amplitude solve, gradient kernel
* `src/fdtdx/adjoint/scene.py` — `derive_adjoint_objects`
* `src/fdtdx/adjoint/vjp.py` — the `jax.custom_vjp`
* `src/fdtdx/adjoint/api.py` — `reciprocity_param_fn`, `reciprocity_phasor_fn`
* `tests/simulation/adjoint/` — 23 tests, all passing; unit suite unchanged at 2578

**Supported:** any differentiable FoM over a plain `PhasorDetector`; lossy design
regions (tested to sigma = 3e5 S/m); `Device` parameters with `param_transforms`;
float32 and float64; CPU and GPU.

**Not supported:** near-to-far and other box-mode detectors, design-dependent
loss (Meep's `MaterialGrid(damping=)`), dispersive design regions,
`dft_subsample > 1`, `reduce_volume`, `exact_interpolation`. All raise rather
than approximating.

**Keep the source below about 0.1 x f0 in bandwidth.** At 0.4 x f0 the pulse is
about 2.5 optical cycles and the error is 2.4e-03 instead of 2.5e-07. This is
not fixed by running longer.

Read `notes/adjoint/03-production.md` before changing any of it: it records four
corrections to earlier conclusions, including one (two `place_objects` calls
re-randomizing `Device` parameters) that was a silent correctness bug.

## Environment

Install and run through `uv`, which is what CI uses and what honours `uv.lock`:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/zhuwei/fdtdx-dev/fdtdx
uv sync
uv run python -m pytest tests -m unit
```

The venv currently carries **CPU jax** (0.10.1). That is deliberate: the
validation work in `notes/adjoint/` is float64 and CPU-only, and float64 on GPU
is slow and in places unsupported. Add a GPU wheel only when we get to timing
measurements, and record it here when you do.

Do not `conda activate` anything in this repo.

## What this branch is for

Adding a **reciprocity-based gradient** to FDTDX: place an explicit adjoint
source and run a second forward-in-time simulation, instead of differentiating
through the time loop. Measured on this machine, FDTDX's existing
`value_and_grad` costs about 7x a forward solve; the reciprocity route should
cost about 2x.

Scope decisions already made, do not re-litigate without asking:

1. **Base:** fork `main` at `60c1c271`, which is 15 commits ahead of the
   campaign pin `f4e610c3` and 0 behind, so the pin's history is included.
2. **Scope:** the raw `PhasorDetector` DFT field monitor only, E and H
   components. The `custom_vjp` boundary sits at the **raw stored phasors**, so
   JAX differentiates any post-processing layered above it and mode overlap,
   near-to-far and diffraction orders come along without their own
   adjoint-source rules.
3. **Integration:** a **standalone opt-in wrapper first**, not a new
   `GradientConfig` method. Nothing in `run_fdtd` or `config.py` changes until
   the numbers hold, so existing gradient paths cannot regress.

## Notes and their status

- `notes/adjoint/01-reciprocity-physics.md` — the theory the implementation must
  reproduce: reciprocity as transpose symmetry, the adjoint source definition,
  worked `dF/dE` for common FoMs, the E-and-H case, and where it breaks. Read
  this before writing any adjoint code.
- `notes/adjoint/_source-transcript.txt` — the raw AI chat transcript the above
  was distilled from. Reference only. It contains errors; trust the distilled
  note, and trust the code's own expressions over both.

## Validation, and what not to gate on

The acceptance ladder, strongest first. Do not promote code on a weaker rung
than the one that is available.

1. **Against `checkpointed_fdtd`**, float64, `rel < 1e-5`. This is the real
   gate. `checkpointed` is exact AD of the same discrete program, so agreement
   here proves the kernel is the exact discrete adjoint.
2. Multi-frequency, `rel < 1e-4`.
3. Central finite differences on `inv_permittivities`, `rel < 1e-3`.
4. End-to-end through `Device.apply_params`, float64 `rel < 1e-2`.
5. Cross-solver against Meep, cosine similarity above 0.95.

**Do not gate on `reversible_fdtd`.** It carries float32 reconstruction drift
and its own upstream test only requires 1e-2.

**Do not gate on Meep.** The two codes discretize and subpixel-smooth
differently, so cross-solver agreement is percent-level at best, which is too
loose to catch a sign flip on one field component or a missing factor of `dt`.
Meep earns rung 5 because it catches systematic errors that both FDTDX paths
could share, not because it is precise.

**Do not gate on a fixed tolerance either.** Reciprocity is a frequency-domain
identity, so it equals the exact discrete adjoint only once both DFTs have
converged. Measured relative L2 against `checkpointed` autodiff went 1.6e-04 at
100 fs, 1.8e-04 at 200 fs and 6.1e-06 at 400 fs. The meaningful assertion is
that the error *falls* with decay time, which is what
`test_gradient_converges_with_runtime` checks. A fixed number passes or fails
for reasons unrelated to correctness.

**Settled:** FDTDX's CPML does preserve discrete reciprocity, to 1.2e-15 with a
slab present, and periodic boundaries give bit-identical results. The earlier
worry that PML might cap the achievable tolerance was unfounded.

## Test problems come from the test bed, not from imagination

Do not invent scenes or figures of merit. Draw them from
`C:\Users\ZhuWei\Desktop\FDTDX\Photonics inverse design test bed\` (Windows
side, readable directly).

The one problem that matches this branch's scope is **RGB metalens**, at
`NanoComp testbed/photonics-opt-testbed/RGB_metalens/`:

- Monitor and FoM, verbatim from `metalens_check.py:63`:
  `mpa.FourierFields(sim, mp.Volume(center=(0,1.92), size=(0.1,0,0)), mp.Ex)`
  with `J(Ex) = -abs(Ex[:,2])**2`. A DFT field monitor on a 0.1 um line at the
  focus; three wavelengths at 450, 550 and 650 nm.
- 13 committed reference designs in `Ex/` and `Ez/`, each labelled by minimum
  feature size, with FoMs cross-validated by three independent codes (Rasmus's
  FEM, Mo's Meep, Wenjin's BEM).
- A committed reference FoM for the empty design:
  `[-0.16528992, -0.16579352, -0.16187553]`.
- `metalens_check.py` imports `ruler`, which is **not vendored**. It only feeds
  a minimum-length printout, not the FoM, so drop that line.

The other test-bed problems are out of scope for now: the Ceviche challenges are
FDFD with mode-overlap objectives, and `Metagrating3D` and
`waveguide_mode_converter` use `EigenmodeCoefficient`.

## Known traps

- **`frozen_field` recompiles.** Frozen values live in the PyTreeDef and
  `objects` is a jit argument, so per-iteration adjoint amplitudes stored in a
  `frozen_field` recompile the whole FDTD loop every optimizer step. Use
  `private_field()`, and assert treedef invariance across amplitude changes.
- **Build the adjoint container inside `bwd`.** Closing over `objects` in a
  bare `custom_vjp` raises `UnexpectedTracerError` on a traced source leaf.
- **Route the adjoint run through the `gradient_config=None` branch** of
  `fdtd/wrapper.py`. It is a plain forward run and must never go through
  `reversible_fdtd`.
- **Use the discrete `iomega`.** The continuum `1/(i*omega)` in the textbook
  adjoint source becomes `(1 - exp(-1j*omega*dt))/dt` in FDTDX's leapfrog.
  Substituting the continuum factor degrades agreement at coarse resolution and
  near Nyquist.
- **Get the monitor transpose from `jax.vjp`,** never by hand.
  `PhasorDetector.update` is linear in `(E, H)`, so its VJP *is* the transpose
  and it carries `static_scale`, the co-location stencil, the region restriction
  and the H time-average split for free.
- **v1 must hard-error, not silently approximate,** unless
  `dft_subsample == 1`, `reduce_volume=False`, `exact_interpolation=False`, and
  `type(detector) is PhasorDetector`. Box-mode and projection detectors store
  per-face keys and bypass `PhasorDetector.update`.
- **Scenes must decay** to about 1e-8 of peak field before the gradient is
  trusted at 1e-5. Reciprocity equals AD only up to DFT truncation, and the
  error is the product of two truncated transforms.

## A separate bug, in the other repo

`/home/zhuwei/fdtdx/color_splitter_d4/fdtdx_color_d4/solver_compare.py:341`
builds `project_root/"scripts"/"run_meep_solver_compare.py"`, but that file
lives at `scripts/color_splitter_d4/run_meep_solver_compare.py`. The harness
raises `FileNotFoundError` before doing any work, so it has not run since the
path moved. Fix it there, not here, before attempting rung 5.
