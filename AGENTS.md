# AGENTS.md

## Read this before touching anything

This is a **development fork of FDTDX** that adds a reciprocity
(adjoint-method) gradient. It is not a working install and not a consumer of
FDTDX. Keep these trees apart:

| Tree | What it is | May an agent edit it? |
| --- | --- | --- |
| `/home/zhuwei/fdtdx-dev/fdtdx` | this repo's main worktree, branch `reciprocity-adjoint`, pushed to `github.com/username421421/fdtdx` (a fork of `ymahlau/fdtdx`) | **Yes.** |
| `/home/zhuwei/fdtdx-dev/fdtdx-cs-probe` | worktree, branch `probe/accept-defaults`: **the current adjoint code**, local only, ahead of `reciprocity-adjoint` | Yes, when the task names it. |
| `/home/zhuwei/fdtdx-dev/fdtdx-lossy` | worktree, branch `probe/lossy-design`: design-dependent loss, in progress | Only for that task. |
| `/home/zhuwei/miniconda3/envs/fdtdx` | FDTDX 0.6.2 installed non-editable at `f4e610c3`; backs the colour-splitter and NeuroShaper campaigns | **No.** Installing anything here silently changes published results. |
| `/home/zhuwei/fdtdx/color_splitter_d4` | research project that *uses* the pinned 0.6.2; has its own `AGENTS.md` | Only when the task is about it. |

`git worktree list` shows the current worktrees, and
`git log reciprocity-adjoint..probe/accept-defaults` what is not yet pushed.
There is also `/home/zhuwei/miniconda3/envs/mp` with Meep 1.34.0, used as a
cross-solver reference. Never install FDTDX into it.

If you are here because a task said "optimize FDTDX" or "work on the adjoint
method", you are in the right place. If a task mentions the colour splitter, the
D4 optimizer, `balanced_480`, NeuroShaper or effective rank, you are in the
wrong tree.

## Status

The reciprocity gradient is **implemented, validated against FDTDX's own
pipeline, and usable for inverse design**. The scene needs nothing added: no
adjoint source and no design-region detector.

```python
from fdtdx.adjoint import reciprocity_param_fn

param_fn = reciprocity_param_fn(arrays, objects, config, key,
                                objective_detectors="mon")
value, grad = jax.value_and_grad(lambda p: my_fom(param_fn(p, beta=beta)))(params)
```

`objects` may come straight from `place_objects`. Once the fields have decayed
the gradient equals `jax.grad` of `apply_params -> run_fdtd(GradientConfig(
"checkpointed"))`: forward value bit-identical, gradient rel L2 about 1e-6 in
float64, 9.4x faster at 24³ on the CPU and 35.3x at 48³ float32 on the GPU
(`notes/adjoint/03-production.md`). `reciprocity_phasor_fn` is the same one
level down, on `inv_permittivities`. `param_fn.diagnostics` and a
`ConvergenceWarning` report DFTs that have not converged.

**Supported:** any differentiable figure of merit over `PhasorDetector`
phasors: E, H or both, components in any declared order, either
`scaling_mode`, `exact_interpolation` on or off, `dft_subsample`; several
monitors in one adjoint solve; `ModeOverlapDetector`, box-mode
`FieldProjectionAngleDetector` (near-to-far), Poynting flux and closed-box net
power. `Device` parameters with `param_transforms`, several Devices, and a
design region that defaults to every Device or is named (a Device, any
detector's cells, a static block). Isotropic and diagonally anisotropic
permittivity, permeability and conductivity, lossy monitor and design cells
included; static dispersive blocks, also under a Device; PML, periodic and
PEC/PMC symmetry boundaries; `UniformGrid` and `QuasiUniformGrid` (one cell
width per axis); float32 and float64; CPU and GPU.

**Refused at setup,** by `src/fdtdx/adjoint/validation.py`. Each was measured
silently wrong; the numbers are in `notes/adjoint/05-guards.md`.

* Objective detectors that are not phasor detectors, or that have an
  apodization, a switch skipping time steps, `reduce_volume=True`, a
  `dft_subsample` stride below 4 samples per period, or (box projections) fewer
  than six components. Objective monitors with different frequencies.
* Grids whose cell width varies along an axis (a stretched `RectilinearGrid`),
  nonzero Bloch vectors, full 3x3 material tensors.
* A design region overlapping a PML, or containing a stock source (an
  `AdjointCurrentSource` is fine). A mode port or TFSF source overlapping a
  Device. Dispersive Device materials. A named design region that misses Device
  cells (refused by `reciprocity_param_fn`, a warning in
  `reciprocity_phasor_fn`).
* An amplitude solve with condition number above 1e4: the run is too short to
  separate the objective frequencies.

**Not available:** design-dependent loss. `apply_params` writes no Device
conductivity in either pipeline; `probe/lossy-design` is working on it.

**Keep the source below about 0.1 x f0 in bandwidth.** At 0.4 x f0 the pulse is
about 2.5 optical cycles and the error is 2.4e-03 instead of 2.5e-07. This is
not fixed by running longer.

Files:

* `src/fdtdx/adjoint/api.py` — `reciprocity_param_fn`, `reciprocity_phasor_fn`
* `src/fdtdx/adjoint/vjp.py` — the `jax.custom_vjp`, `derive_adjoint_objects`
* `src/fdtdx/adjoint/objective.py` — monitor channels and their transposes,
  adjoint-current placement, the scale, magnetic and lossy factors
* `src/fdtdx/adjoint/design.py` — design regions, the internal design detector,
  `internal_scene`, `apply_objects_once`, `device_dispersion_as_applied`
* `src/fdtdx/adjoint/kernel.py` — window, amplitude solve, gradient kernel,
  `dft_tail`, `ConvergenceWarning`
* `src/fdtdx/adjoint/validation.py` — every refusal
* `src/fdtdx/objects/sources/adjoint.py` — `AdjointCurrentSource`
* `tests/unit/adjoint/` (all in CI) and `tests/simulation/adjoint/` (parity
  against checkpointed `run_fdtd`). CI (`-m "unit or integration or docs"`)
  also runs the parity tests marked `integration`, one cheap test per
  correction. `tests/conftest.py` forces the CPU, so the suite never runs on the
  GPU; validate on the GPU with scripts.

## Environment

Every worktree uses one interpreter, `/home/zhuwei/fdtdx-dev/fdtdx/.venv/bin/python`,
and its editable install points at the **main** worktree. From another
worktree, set `PYTHONPATH=<worktree>/src` and print `fdtdx.__file__` to prove
which tree you are running.

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
3. **Integration:** a **standalone opt-in wrapper**, not a new `GradientConfig`
   method. Nothing in `run_fdtd` or `config.py` changes, so existing gradient
   paths cannot regress.

## Notes

`notes/adjoint/`, in order: `01-reciprocity-physics.md` is the theory the
implementation must reproduce; read it before writing adjoint code.
`02-implementation.md`, `03-production.md` and `04-near-to-far.md` are the
implementation history, with corrections to earlier claims. `05-guards.md` maps
the old module names to the current ones and holds the measurement behind every
refusal, correction factor and default. `_source-transcript.txt` is the raw AI
chat the theory was distilled from; it contains errors, so trust the note and
the code over it.

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

**Settled:** FDTDX's CPML preserves discrete reciprocity for sources and
monitors outside it, to 1.2e-15 with a slab present, and periodic boundaries
give bit-identical results. Inside the layer the kernel's pairing does not hold,
which is why a design region overlapping a PML is refused.

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
  re-randomizes the Device parameters (`vjp.derive_adjoint_objects`).
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

## A separate bug, in the other repo

`/home/zhuwei/fdtdx/color_splitter_d4/fdtdx_color_d4/solver_compare.py:341`
builds `project_root/"scripts"/"run_meep_solver_compare.py"`, but that file
lives at `scripts/color_splitter_d4/run_meep_solver_compare.py`. The harness
raises `FileNotFoundError` before doing any work, so it has not run since the
path moved. Fix it there, not here, before attempting rung 5.
