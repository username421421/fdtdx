# AGENTS.md

## Read this before touching anything

This is a **development fork of FDTDX** that adds a reciprocity
(adjoint-method) gradient. It is not a working install and not a consumer of
FDTDX. Keep these trees apart:

| Tree | What it is | May an agent edit it? |
| --- | --- | --- |
| `/home/zhuwei/fdtdx-dev/fdtdx` | this repo, the only tree, branch `main`, pushed to `github.com/username421421/fdtdx` (a fork of `ymahlau/fdtdx`) | **Yes.** |
| `/home/zhuwei/miniconda3/envs/fdtdx` | FDTDX 0.6.2 installed non-editable at `f4e610c3`; backs the colour-splitter and NeuroShaper campaigns | **No.** Installing anything here silently changes published results. |
| `/home/zhuwei/fdtdx/color_splitter_d4` | research project that *uses* the pinned 0.6.2; has its own `AGENTS.md` | Only when the task is about it. |

`main`, locally and on GitHub: upstream FDTDX (merged up to `d66442e`) plus
the reciprocity commits. `git log upstream/main..main` lists them, `git status -sb`
shows what is not yet pushed. Work on `main`. `reciprocity-full-20260926` (also on
GitHub) keeps the earlier, larger version: `reciprocity_param_fn` /
`reciprocity_phasor_fn`, lossy Devices, the `apply_params` provenance guard.
Do not create branches or worktrees without asking.
There is also `/home/zhuwei/miniconda3/envs/mp` with Meep 1.34.0, used as a
cross-solver reference. Never install FDTDX into it.

If you are here because a task said "optimize FDTDX" or "work on the adjoint
method", you are in the right place. If a task mentions the colour splitter, the
D4 optimizer, `balanced_480`, NeuroShaper or effective rank, you are in the
wrong tree.

## Status

The reciprocity gradient is **implemented, validated against FDTDX's own
pipeline, and usable for inverse design**. An existing
`apply_params -> run_fdtd -> FoM -> jax.grad` script changes one string:

```python
config = config.aset("gradient_config", fdtdx.GradientConfig(method="reciprocity"))
```

The forward stays `run_fdtd`'s bit for bit; the phasor detectors the FoM reads
get the adjoint currents; the gradient is taken with respect to
`inv_permittivities` inside the Devices, which is where `apply_params` writes the
Device parameters: exact for them, zero outside. Once the fields have decayed it
equals `GradientConfig("checkpointed")`'s, gradient rel L2 1e-8 to 1e-6 in
float64. A Device under a stock box far field (64x64x72, three wavelengths,
GPU): rel 2.1e-08 (float64) and 2.0e-06 (float32) at 2.1-2.2x a forward run,
9-18x faster than checkpointed with 3-7x less memory
(`notes/adjoint/05-guards.md`). It fails fast inside the method: every gradient
carries an estimate of its DFT truncation error and raises, on every call, above
`GradientConfig.tail_tolerance` (default 1e-2), and every configuration it cannot
compute exactly raises when the gradient is traced. It never falls back to
another method, and it changes nothing outside the method.

**Footprint.** Outside `src/fdtdx/adjoint/`, upstream files change in two places
only: `config.py` (the `"reciprocity"` value of `GradientConfig.method`,
`tail_tolerance`, and refusing a `Recorder` with it) and `fdtd/wrapper.py` (the
`run_fdtd` branch). Keep it that way: no fail-fast check may live in, or change
the behaviour of, upstream code.

**Supported:** any differentiable figure of merit over `PhasorDetector`
phasors: E, H or both, components in any declared order, either
`scaling_mode`, `exact_interpolation` on or off, `dft_subsample`; several
monitors in one adjoint solve, also at different frequencies (the solve runs at
their union); `ModeOverlapDetector`, box-mode
`FieldProjectionAngleDetector` (near-to-far), Poynting flux and closed-box net
power. `Device` parameters with `param_transforms`, several (also overlapping)
Devices. Isotropic and diagonally anisotropic permittivity, permeability and
conductivity, lossy monitor and Device cells (loss placed under a Device:
`apply_params` writes a Device's permittivity only, in every method), etched
Devices; static dispersive blocks, also under a Device; PML, periodic and PEC/PMC
symmetry boundaries; `UniformGrid`, `QuasiUniformGrid` and any grid with one cell
width per axis; float32 and float64; CPU and GPU; `jax.jit`, `eqx.filter_jit` and
`jax.checkpoint` around the loss.

**Refused** by `src/fdtdx/adjoint/validation.py` and `reciprocity.py`. Each was
measured silently wrong; the numbers are in `notes/adjoint/05-guards.md`.

* A FoM reading the fields, a time-domain detector or any other `run_fdtd`
  output; a differentiated input other than `inv_permittivities` (the dispersion
  coefficients count as constants: the blend `apply_params` writes for a
  non-dispersive Device material has zero derivative).
* Objective detectors that have an apodization, a switch skipping time steps,
  `reduce_volume=True`, `inverse=True`, cells in a PML beyond its zero-loss first
  cell, a `dft_subsample` stride that folds source spectrum onto an objective
  frequency, or (box projections) fewer than six components.
* Grids whose cell width varies along an axis (a stretched `RectilinearGrid`),
  nonzero Bloch vectors, full 3x3 material tensors, a scene without a Device.
* A Device overlapping a PML or containing a stock source (an
  `AdjointCurrentSource` is fine); dispersive Device materials.
* An amplitude solve with condition number above 1e4: the run is too short to
  separate the objective frequencies.

**Not refused, by design:** a figure of merit whose parameters reach the
permittivity outside the Devices (a background parameter written into the
arrays) gets no gradient there. Detecting it needs a hook in `apply_params`; the
earlier version had one (branch `reciprocity-full-20260926`), removed to keep
upstream code untouched. Forward mode (`jax.jvp`) raises JAX's own `custom_vjp`
error.

**Sources must end, and should be DC-free.** A source still injecting at the end
of the run is refused. A few-cycle pulse carries DC whose static remainder never
decays where its current ends inside the domain (0.8% at 0.4 x f0, 8e-8 with the
DC-free carrier); the convergence estimate raises on it. Make the carrier DC-free
(`docs/source/reciprocity.rst`) or keep the bandwidth near 0.1 x f0.

Files, all in `src/fdtdx/adjoint/`:

* `reciprocity.py`: `reciprocity_fdtd`, the `custom_vjp` behind `run_fdtd`
* `solve.py`: `AdjointSolve` (the backward rule), `segmented_solve`
* `objective.py`: monitor channels and their transposes, adjoint-current
  placement, the scale, magnetic and lossy factors
* `design.py`: the internal design detector, `internal_scene`, the late windows
* `kernel.py`: window, amplitude solve, gradient kernel, `dft_tail`,
  `gradient_error_estimate`, `refuse_unconverged`
* `validation.py`: the refusals
* `source.py`: `AdjointCurrentSource`

Tests: `tests/unit/adjoint/` (all in CI) and `tests/simulation/adjoint/` (parity
against checkpointed `run_fdtd`). CI (`-m "unit or integration or docs"`) also
runs the parity tests marked `integration`. `tests/conftest.py` forces the CPU,
so the suite never runs on the GPU; validate on the GPU with scripts.

## Environment

The interpreter is `/home/zhuwei/fdtdx-dev/fdtdx/.venv/bin/python`; its
editable install points at this tree's `src`, so no `PYTHONPATH` is needed.
Print `fdtdx.__file__` to prove which tree you are running.

The venv carries jax 0.10.1 with the CUDA 13 plugin, added with `uv pip` and not
in `uv.lock`, so a plain `uv sync` removes it; `uv sync --inexact` keeps it.
Export `XLA_PYTHON_CLIENT_PREALLOCATE=false` for every JAX process on the shared
GPU. Install only through `uv`:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/zhuwei/fdtdx-dev/fdtdx
uv sync --inexact
uv run python -m pytest tests -m unit
```

Do not `conda activate` anything in this repo.

## What this branch is for

Adding a **reciprocity-based gradient** to FDTDX: place adjoint currents and
run a second forward-in-time simulation, instead of differentiating through the
time loop. FDTDX's built-in gradients cost about 8.9x (reversible) and 52x
(checkpointed) a forward solve; reciprocity costs about 2.6x.

Scope decisions already made, do not re-litigate without asking:

1. **Base:** fork `main` at `60c1c271`, which is 15 commits ahead of the
   campaign pin `f4e610c3` and 0 behind, so the pin's history is included.
2. **Boundary:** the `custom_vjp` sits at the **raw stored phasors** of the
   objective monitors, so JAX differentiates any post-processing above it, and
   mode overlap, near-to-far and flux come along without their own
   adjoint-source rules.
3. **Integration:** opt-in, a contained method beside `checkpointed` and
   `reversible` (2026-09-26): one `Literal` value, one config field and one
   `run_fdtd` branch; everything else in `src/fdtdx/adjoint/`. Upstream behaviour,
   and every other method, are untouched. The functional entry points, lossy
   Devices and the `apply_params` provenance probe were removed then, and kept on
   branch `reciprocity-full-20260926`.
4. **Fail fast** (asked for on 2026-09-24): where reciprocity cannot return
   checkpointed's gradient it raises, naming the reason and `method="checkpointed"`.
   No warning-only path where the gradient can be wrong, and never a silent
   fallback to another method. Refusals live inside the method only (2026-09-26).
   Scope stays parity with native FDTDX.

## Notes

`notes/adjoint/`, in order: `01-reciprocity-physics.md` is the theory the
implementation must reproduce; read it before writing adjoint code.
`02-implementation.md`, `03-production.md` and `04-near-to-far.md` are the
implementation history, with corrections to earlier claims. `05-guards.md` holds
the measurement behind every refusal, correction factor and default. The notes
were written for the functional API (`reciprocity_param_fn`,
`reciprocity_phasor_fn`) and also describe lossy Devices and the provenance
guard, all removed on 2026-09-26; their measurements hold for the method, whose
gradients are bit-identical to the ones they took. Module names there: `vjp.py`
is now `solve.py`, `dropin.py` is `reciprocity.py`, `objects/sources/adjoint.py`
is `adjoint/source.py`. `_source-transcript.txt` is the raw AI chat the theory
was distilled from; it contains errors, so trust the note and the code over it.

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

Always report rel L2, cosine **and** best-fit scale: several bugs here were pure
scale errors at cosine 1.0000000000.

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

**Checkpointed is exact for the run, not for the converged answer.** Where a
mode rings off the objective frequency, checkpointed's gradient of the truncated
DFT converges far more slowly than the figure of merit (its truncation carries a
factor `(w_r - w) T`), and in a periodic slab with a guided mode that does not
radiate it never converges. There reciprocity and checkpointed disagree by tens
of percent with reciprocity the converged one (Fabry-Perot off resonance at
3000 fs: reciprocity 2.5e-4 from the converged gradient, checkpointed 0.19;
`notes/adjoint/05-guards.md`, "Which gradient converges"). Use rung 1 only on
scenes whose fields decay; otherwise compare each method against a longer run.

**Settled:** FDTDX's CPML preserves discrete reciprocity for sources and
monitors outside it, to 1.2e-15 with a slab present, and periodic boundaries
give bit-identical results. Inside the lossy part of the layer the pairing does
not hold: a design region overlapping a PML is refused, and an objective there
is refused wherever the local CPML strength is not zero (the first PML cell, graded to zero
loss, is exact; one cell deeper was 5e-4 wrong; notes/adjoint/05-guards.md).

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
- **Route both solves through the `gradient_config=None` branch** of
  `fdtd/wrapper.py`. They are plain forward runs and must never go through
  `reversible_fdtd`.
- **Use the discrete `iomega`.** The continuum `1/(i*omega)` in the textbook
  adjoint source becomes `(1 - exp(-1j*omega*dt))/dt` in FDTDX's leapfrog.
  Substituting the continuum factor degrades agreement at coarse resolution and
  near Nyquist.
- **Get every monitor transpose from JAX** (`jax.linear_transpose` of FDTDX's
  own recording code), never by hand. It carries the co-location stencil, the H
  time average and the boundary padding for free.
- **Derive the adjoint scene, never place it again.** `place_objects` splits its
  key once per object, so a second call with a different object count
  re-randomizes the Device parameters (`solve._adjoint_objects`).
- **The design detector is not the user's.** Never go back to validating a
  user-built one: each of its settings was a *silent* error while it was
  user-facing (six components: cosine -0.473; `continuous`: pure scale at cosine
  1.0; `inverse=True`: exactly zero; different `wave_characters` from the
  monitor: rel 0.90). It is built fresh by `design.make_design_detector`, never
  by `aset` on an existing detector, because `aset` leaves the `place_on_grid`
  caches stale (a stale apodization window was a silent rel 0.92).
- **Scales are divided out, never assumed to be 1**, per objective monitor and
  per design detector. Test on rel L2 and best-fit scale, not cosine.
- **Adjoint currents follow the stored component order.** `PhasorDetector`
  stacks components canonically (Ex..Hz) whatever order `components` declares;
  use `objective.canonical_components`.
- **Isotropic scenes do not reach every path.** A diagonally anisotropic
  material (three `inv_permittivities` rows) reaches the kernel's component
  axis, the injection factor and the lossy divisor, and `mu != 1` reaches the H
  injection. Two of the bugs there were pure scale errors; test changes on
  anisotropic and `permeability=2` scenes.
- **Scenes must decay** to about 1e-8 of peak field before the gradient is
  trusted at 1e-5. Reciprocity equals AD only up to DFT truncation, and the
  error is the product of two truncated transforms.
- **Rerun `tests/unit/adjoint` after any JAX upgrade.** The rule relies on
  `custom_vjp` with `symbolic_zeros` (`CustomVJPPrimal.perturbed`,
  `custom_vjp_primal_tree_values`), and on a grid rebuilt inside a trace having
  traced edges (`validation.check_grid` then refuses when the gradient runs).

## A separate bug, in the other repo

`/home/zhuwei/fdtdx/color_splitter_d4/fdtdx_color_d4/solver_compare.py:341`
builds `project_root/"scripts"/"run_meep_solver_compare.py"`, but that file
lives at `scripts/color_splitter_d4/run_meep_solver_compare.py`. The harness
raises `FileNotFoundError` before doing any work, so it has not run since the
path moved. Fix it there, not here, before attempting rung 5.
