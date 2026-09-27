# Guards, correction factors and diagnostics: the numbers behind them

Fifth in the series. The code keeps its docstrings and error messages short; this file
keeps the measurements that justified each refusal, correction factor and default, which
used to live in those docstrings. Unless stated otherwise every number is a relative L2
error of the gradient against `jax.grad` of `run_fdtd(GradientConfig("checkpointed"))` on
the same placed scene, GPU, float64, measured on the code *before* the guard or correction
existed. Always read a gradient comparison as rel L2, cosine **and** best-fit scale: several
of the errors below had cosine 1.0000000000.

## Module map

The lean refactor (after `5d02475`) renamed and regrouped the modules the earlier notes cite.
Behaviour, and every bit of every result on the golden set, is unchanged.

| earlier notes | now |
| --- | --- |
| `fdtdx.adjoint.reciprocity` (window, amplitude solve, kernel) | `fdtdx.adjoint.kernel`, with `dft_tail` and the convergence refusal (formerly `ConvergenceWarning`) |
| `fdtdx.adjoint.recording` (channel transposes) | `fdtdx.adjoint.objective`, with the channel layouts, adjoint-current placement, scale / magnetic / lossy factors |
| `fdtdx.adjoint.scene.make_design_detector`, `internal_scene`, `design_regions` | `fdtdx.adjoint.design`, with `apply_objects_once`, `device_dispersion_as_applied` |
| `fdtdx.adjoint.scene.derive_adjoint_objects` | `fdtdx.adjoint.vjp.derive_adjoint_objects` |
| refusals spread over `vjp.py`, `api.py`, `scene.py` | `fdtdx.adjoint.validation` |
| `make_reciprocity_phasor_fn` (two hand-built scenes) | removed: the adjoint scene is always derived; a hand-built one gave bit-identical results |
| `design_region_slice(objects, name)` | `objects[name].grid_slice` |

## Cost

On an RTX 3080 Ti FDTDX's built-in gradients cost about 8.9x a forward solve (reversible)
and 52x (checkpointed); reciprocity was measured at 2.56x.

Adjoint current injection: one indexed add per field family instead of one per component.
The per-component adds each copied the field array inside the time loop, so the five face
currents of a box cost three bare solves (0.427 s vs 0.104 s at 72x72x90, float32); after,
0.204 s, equal to the forward solve (0.197 s).

## Amplitude-solve conditioning (`DEFAULT_COND_LIMIT = 1e4`)

float32 solve error of the amplitude matrix against its float64 solution, per condition
number: 8 -> 1.8e-07, 3e2 -> 1.5e-06, 5e3 -> 4.9e-05, 4e5 -> 5.7e-03, 8e6 -> 8e-02,
7e7 -> 0.46-0.75, i.e. about `1e-8 * cond`; 1e4 keeps it near 1e-4.

It is also where the colour splitter's gradient stops being usable. With the default window
its `cond` is 7.97 at 80 fs, 5.1e3 at 40 fs (gradient rel 0.29, which only the convergence
check catches), 3.5e4 at 35 fs (rel 0.78), 4.3e5 at 30 fs (rel 2.3) and 7.2e7 at 22 fs
(rel 20), which the former 1e8 ceiling let through. Every scene in the test suite is at 1.2e3
or below.

## Convergence refusal (`DEFAULT_TAIL_TOLERANCE = 1e-2`)

Fail fast (2026-09-24): the former `ConvergenceWarning` fired once per compiled function, so a
jitted optimizer was silent after its first step (rel 5.7e-2 on every later call, and rel 0.78 at
tails above the tolerance with no warning), and it measured only the field left at the last step,
as if it were static. It missed resonant ringing by about the resonance's Q (26x to 1e4x): a
high-index Device at 60 fs was rel 0.41 at a largest tail of 8.6e-3; a Fabry-Perot cavity rel 2.25;
periodic grazing orders 3-60%; two parallel PEC/PMC walls 2-40%; a CW source 1-4%, all silent.

Now every gradient is checked on every call, through `equinox.error_if`, against `tail_tolerance`
(`GradientConfig.tail_tolerance`, the functional entry points' argument; `None` switches it off):

* `dft_tail_cells` (`dft_tail` is its relative part), per frequency and per recording (forward and
  adjoint design regions, objective channels): the larger of the static estimate
  `||E_end|| / |1 - exp(i w dt)|` and ringing, from the phasors' growth over the last three eighths of
  the run, windows `D0, D1, D2` (`design.late_windows`). Two decaying modes are fitted over the cells
  (`D2 = -(a0 D0 + a1 D1)` by least squares, modified Gram-Schmidt, `z^2 + a1 z + a0 = 0`) and summed
  on, `v z^3 / (1 - z)`; what they leave of `D2` is continued like the slowest of them and of the
  last pair's ratio `z = <D1, D2> / <D1, D1>`, `|z / (1 - z)|`. Under 8 entries per frequency (a
  one-cell monitor) or with collinear windows (one mode, or a standing wave's two rotating parts):
  one mode, `||D2|| |z| / (|1 - z| - s)`, `s = (1 + |z|) / 2 * static / ||D1||` the shift of `z` the
  counter-rotating part's boundary terms can make (taken as 0 where it exceeds 0.5: windows no
  bigger than the static remainder hold no ringing). `|1 - z|` is held at 1e-3 or more; `|z|` is
  not held below 1. It also returns the tails cell by cell (over the components), for the gradient
  check. Norms are scaled before squaring (float32 underflowed to 0 at small amplitudes).
* Why (2026-09-25). The former estimate, one mode fitted to the last two windows, was measured
  against the phasors of runs long enough to converge (fine-segmented histories,
  `scratchpad/verify2/ringing/s_history.py`, `offline_est.py`), true over estimated: exact for one
  mode at any detuning (Fabry-Perot cavity off resonance, 0.99 to 1.2, once 4.0), but an eps-12
  Device's mixture of modes 1.5 to 2.4 on the forward design phasors and 1.9 to 6.7 on the adjoint
  ones: a fast non-resonant part dominates the next-to-last window and the complex ratio projects
  onto it (`|z| = 0.06` where the tail decays by 0.4 per window). Now: the Device 0.72 to 1.35, the
  cavity 0.96 to 1.09, a periodic slab's guided mode 0.62 to 1.01. Rejected on the same histories:
  the rest continued with the windows' magnitude ratio, which assumes no cancellation (the Device
  0.39 to 0.80, but the guided mode, which does not decay, 0.08 to 0.31 and an objective tail of 0.26
  on a figure of merit converged to 1e-5); one mode with both window ratios held below 1 (a one-cell
  objective tail of 6 where the figure of merit was 6e-3 off: beating modes make one window grow);
  two modes alone (the Device 1.4 to 3.3).
* Review (2026-09-25, `scratchpad/review3/`), each now a unit test in `TestDftTail`: a standing wave
  on resonance decaying by 1% per window read 3.7x low (the counter-rotating shift; now 1.1x to 2.2x
  high, an ordinary resonance 1.3x high); in complex64 the Gram determinant lost a slow mode (z 0.999)
  at 3e-3 of a fast one (0.09 of its tail, one Gram-Schmidt projection 0.39; modified Gram-Schmidt
  0.997 to 1.000, and the collinearity threshold at 1e-10 instead of 1e-6 keeps it at 1e-3 in
  float64); per-cell norms of 1e-21 adjoint phasors squared to 0 in float32 and switched the check
  off. Also fixed: `progress_callback` restarted per segment (one report now,
  `test_progress_is_one_run`), and the drop-in kept three snapshots of every detector, time series
  included, to the backward pass (phasor detectors only now). On one cell a detuned real carrier's
  two rotating parts still count as one mode (1.42x low at detune 0.3); over cells with a complex
  mode shape the fit separates them.
* Review 4 (2026-09-25, `scratchpad/review4/`: four reviewers, each finding reproduced or refuted
  by a skeptic; 17 of 22 confirmed), estimator part, each a unit test in `TestDftTail`: a field
  still arriving (an echo from an eps-12 slab 250 cells away reaching the monitor in the last window,
  real FDTD) was continued as a series, estimate 4.5e-5 where the figure of merit was 89% off and the
  gradient's cosine -0.72: a last window more than 10x both earlier ones is now infinite, which also
  covers zero earlier windows (exact, or flushed to 0 in float32). A standing wave on resonance plus
  a 1e-3 remainder elsewhere switched to the two-mode fit (0.28 of the tail): nearly collinear
  windows (`||b_perp||^2 < 1e-3 ||b||^2`) also take the shifted one-mode estimate. A float32
  continuation overflowing to NaN made `where(total > 0)` read the truncation as 0: non-finite now
  refuses. Not fixed, and a limit of any estimate from three windows: a weak resonance (1e-3, Q far
  beyond the run) hidden under faster modes, 0.05 of its tail over cells under two, 0.03 on one cell
  under one (where the real field's two rotating parts already make two modes). Six windows and an
  order-2 fit (`scratchpad/verify2/ringing/v2est.py`) did not catch it either (it needs order 4 to 6);
  a floor at the windows' magnitude ratio on few-entry monitors read single detuned modes up to 15x
  high without catching it. The same figure of merit is equally unconverged in every method, with no
  warning from any; the design-region check, fitted over many cells, sees such a resonance where it
  rings in the Device.
* Every solve runs in four `custom_fdtd_forward` segments split at the windows
  (`vjp.segmented_solve`), and `D0, D1, D2` are differences of detector-state snapshots; the
  segmented forward is bit-identical to `checkpointed_fdtd`'s. Two internal late-window detectors per
  recording were tried first: FDTDX guards every detector update with a `lax.cond`, a device sync per
  step on the GPU, and the lossy near-to-far benchmark went from 0.33 s to 1.21 s per value+grad
  (float32). Segmented (three segments): 0.327 s (2.4x a forward), peak 70 MiB against 82; float64
  1.22 s, rel 2.0e-08 against checkpointed as before.
* The objective check: the largest objective tail over the channels and frequencies the figure of
  merit reads with a share of its first-order change, `||cotangent|| ||phasor||`, above 1e-6 (the
  first estimate flagged unread frequencies: tail 1.0 on a gradient exact to 1.7e-4; an unweighted
  maximum then refused a figure of merit converged to 1e-7 for a read monitor carrying 1e-17 of it;
  a share-weighted average passed a penalty term near its target, share 2e-2 and tail 2e-2, with
  the gradient 11-28% off). The shares are those of one gradient call: a call reading that monitor
  alone is its gradient, and refused. Above the tolerance the figure of merit itself is off, with
  its own message (DC advice). Not seen by any check inside the VJP, which never sees the figure of
  merit's curvature: a figure of merit near a target (least squares, equality penalties) amplifies
  the phasors' truncation by about `|P| / |P - P0|` (a single target-matching channel at a 1.5e-3
  tail was accepted with the gradient's sign reversed, and checkpointed was equally wrong). There
  compare the per-channel tails in the diagnostics with the distance to the target.
* The gradient check, `gradient_error_estimate`: the distance from the converged gradient,
  `sum_f ||truncation_f|| / ||grad|| + eta_obj`, with `truncation_norms`
  `|K| (|dF| |Lambda| + |F| |dLambda|) / min(inv_eps)^2` cell by cell, over the components, plus the
  objective tail, which shifts the cotangent the adjoint solve is driven by about as much. Cell by
  cell because the norm product (a term's norm times the phasors' relative tails) read the periodic
  slab's gradient truncation 2x to 2.6x low: its guided mode rings where the adjoint field is strong
  (per cell 1.3x to 2x high; the Device and the cavity 1.1x to 2.6x high). The objective part: on the
  cavity at 2000 fs the design part was 4.1e-3 and the objective 5.2e-3 against a true 9.3e-3. (It
  was once added and taken out again because on the colour splitter's DC pulse the sum was 6.1e-02
  against a "true" 9.4e-03, but that true error was the distance from checkpointed at the same run,
  which shares the cotangent's truncation.) Measured, estimate over true distance from the converged
  gradient: the eps-12 Device 1.9 to 4.3 (200 fs: 1.06e-2 against 3.2e-3, refused), the cavity 1.2 to
  3.2 (2000 fs: 1.13e-2 against 9.3e-3, refused; one 1.7x low at 1000 fs, where the true error is
  8.7e-2 and it is refused), the periodic slab 1.2 to 4 against reciprocity's change to a doubled run
  (1000 fs: 4.1e-2 against 3.3e-2, where the norm product accepted 9.4e-3).
* 24^3 plain scene (Device eps 2.25), float64, CPU, estimate over true: 35 fs 5.2e-4 / 1.2e-4, 40 fs
  1.1e-6 / 5.5e-7, 60 fs 3.4e-9 / 5.0e-9; the eps-12 Device at 60 fs raises, true rel above 1e-2
  (`TestConvergenceDiagnostic`). Colour splitter, balanced, GPU float32, DC-free pulse, 160 fs: both
  designs accepted (estimates 8.9e-4 and 1.2e-3; the optimized design rel 1.7e-4 against
  checkpointed; 2.4 s against 18.8 s per value and gradient); the grey starting design differs from checkpointed by 1.1%, which is
  checkpointed still converging (06-color-splitter.md: reciprocity moved 1.5e-4 from 320 to 640 fs,
  checkpointed 3e-3). The production DC pulse at 80 fs is refused by the objective check (tails 3.8e-2
  to 4.0e-2).

The adjoint window moved from 0.22 to 0.30 of the run: at 0.22 it started at exp(-8) of its peak, a
DC step in the adjoint current that never decays (2e-2 on a weak frequency sharing a monitor with a
strong one; 1.5e-5 at 0.30), and which also made the adjoint tail estimate 1.6e-3 on a gradient
exact to 9e-7.

A source still injecting over the last tenth of the run is refused at setup
(`validation.check_source_waveforms`, off with `tail_tolerance=None`). DC in a source's waveform is
not refused: a UniformPlaneSource across the whole cell at 0.3 f0 (DC 3.3e-3 of its band) is exact
(6.1e-07), because its current ends in no cell of the domain. Where it does end inside the domain
(the colour splitter's 3.2 um plane source), the static remainder is what the estimate measures.

## Which gradient converges (2026-09-25)

The audit's "silent" ringing cases (reciprocity against checkpointed rel 0.19 to 0.93 with every
estimate under 1e-2) were measured again against the converged gradient: checkpointed at a run
long enough that both agree, or, where none is, each method's change between run lengths
(`scratchpad/verify2/ringing/s_conv.py`, float64, CPU; estimates by `s_est.py`). In every one
reciprocity is the converged gradient and checkpointed the slow one:

| scene | run | reciprocity vs converged | estimate now | checkpointed vs converged |
| --- | --- | --- | --- | --- |
| Fabry-Perot cavity, FoM at 1400 nm, resonance at 1255 nm | 2000 fs | 9.3e-3 | 1.1e-2 | 0.25 |
| | 3000 fs | 2.5e-4 | 8.1e-4 | 0.19 |
| | 4000 fs | 4.3e-5 | 7.3e-5 | 2.1e-2 |
| eps-12 Device, 24^3 fixture | 150 fs | 1.1e-2 | 3.6e-2 | 2.8e-2 |
| | 200 fs | 3.2e-3 | 1.1e-2 | 9.5e-3 |
| periodic slab, 400 nm period | 1000 fs | 3.3e-2 (to 2000 fs) | 4.1e-2 | 0.32 (to 12 ps) |
| | 8000 fs | 1.2e-3 (to 12 ps) | 4.6e-3 | 0.23 (to 12 ps) |
| periodic slab, 550 nm period | 600 to 8000 fs | 0.6% to 2.2% per doubling | | 0.84 to 0.95 (to 8 ps) |

Why: checkpointed differentiates the DFT truncated at `T`. A mode ringing at `w_r != w` leaves in the
figure of merit a tail `a exp((i (w_r - w) - g) T) / (g - i (w_r - w))`, whose design derivative
carries `i T dw_r/de`: the gradient's truncation is about `(w_r - w) T` times the value's (the
cavity at 3000 fs: `(w_r - w) T = 470` rad, checkpointed's error 760x reciprocity's). Reciprocity
pairs phasors, whose truncation is the value's, without the factor `T`. In the periodic slabs a
guided mode below the light line does not radiate: the figure of merit converges (1e-5), the
checkpointed gradient never does. Two consequences. The parity gate, "matches checkpointed", holds
only once checkpointed has converged too; with such modes compare each method against a longer run.
And the audit's comparison of reciprocity's estimate with its difference from checkpointed measured
checkpointed's truncation, not reciprocity's.

## Review 5 (2026-09-26, `scratchpad/review5/`, 13 of 14 confirmed)

* **Arrival behind a decaying first window**: a Device's ringing (n0 = n2 / 9.5) hid an echo arriving
  in the last window (n2 / n1 = 590), accepted at 80% off. Now also arriving when `n2 > 10 n1` and
  `n2 > n0`. The price, measured: the stock PML's own weak reflection arriving late refuses runs
  converged to 3e-6..2e-4 (1D scene, 120-150 fs; a periodic waveguide at 110-125 fs); 10-30% longer
  runs pass. The refusal now says so (a field still arriving, lengthen or thicken the PML).
* **An unread frequency** of a read monitor: its adjoint phasor is the solve's noise (6e-8 to 7e-5
  of a read one), whose windows grow like an arrival; infinite times the forward field refused a
  gradient exact to 3e-6. The adjoint design phasors and tails of frequencies no channel reads are
  set to their converged value, 0.
* **Staged provenance, memoized**: JAX memoizes the staged rule and the probe's JVP per tangent
  pattern, so the flag read in the rule fired once at most (the second background gradient passed, d/dbg 0),
  and never after a Device-parameter gradient (rel 0.79). The probe now raises in its own JVP rule
  once a reciprocity run consumed its record: a raising rule is not memoized. A refusal also clears
  the flags (a stale one refused the next, exact, gradient).
* **A jit value in an object field under `jax.grad(jax.jit(f))`**: a bare `TypeError: No constant
  handler` at lowering. The rule compares the trace that value belongs to with its own, and refuses
  it clearly; under `jax.jit(jax.grad(f))` they are one trace, and it is exact (5e-9).
* **The alias check** refused strided monitors over a row the figure of merit does not read (exact
  to 1.9e-9), and float32 runs from the waveform's own float32 round-off (1.5e-6). Per-row ratios are
  computed at setup and refused in the rule only where read; the limit is `max(1e-6, 100 eps)`.
* **Near-duplicate frequencies** written differently (600e-9 and float32(600e-9), 3.5e-8 apart) were
  refused as ill-conditioned with advice to run longer: now named, to build them from one WaveCharacter.
* Limits, documented: two comparably slow standing modes over many cells (0.23 of the tail: three
  windows cannot hold four rotating parts); an echo not yet in any late window is invisible.
* Refuted: forward mode through `jax.linearize` or `jax.jvp(jax.jit(f))` gives JAX's TypeError, as
  natively.

## Review 4, the rest (2026-09-25)

Each confirmed by a reproducing skeptic, fixed, and a test:

* **A monitor named twice** in `objective_detectors` drove one adjoint current under one name, and
  one cotangent was dropped (a figure of merit reading it twice was half its gradient): refused.
* **Frequencies merged at rtol 1e-6**: 600 nm and 600 (1 + 5e-7) nm shared one row, 3e-5 off, with
  an estimate of 3e-9. Merged now only within 4 ulps of the dtype they are stored in; farther ones
  are separate rows (and ill-conditioned ones refused by the amplitude solve).
* **A frequency recorded twice** had its objective tails summed (refusing converged runs): its tails
  and sizes are placed by `max`, its cotangents still added.
* **The alias check** compared a strided row's folded content with the strongest row's spectrum:
  a weak row folded at 6e-2 of its own content passed (8.96e-7 of the peak < 1e-6), and its gradient
  was 11% off. Each row is now compared with its own spectrum (`ALIAS_TOLERANCE`, as documented).
* **Provenance under staging.** `jax.grad(jax.jit(loss))` (also `eqx.filter_jit`, `jax.checkpoint`)
  with a background written before `apply_params`: the input probe's JVP runs only when the staged
  program is differentiated, after `apply_params` returned, and the flag read at the call was empty:
  d/dbg 0 against checkpointed's 1.9e-2, the gradient rel 0.93. `apply_params` now marks its outputs
  with an `AppliedRecord` holding the probe's flag list, read when the backward rule is traced, and
  cleared after each differentiation (a cached `jit` program reuses its records: read stale, a
  gradient with respect to the Device parameters alone was refused after one with respect to the
  background). The probe applies to traced inputs only (concrete ones carry no derivative, and
  upstream's mock-based `apply_params` tests failed on it).
* **Devices missing from the container** `run_fdtd` is given, while `apply_params` wrote them: their
  gradient was 0 (checkpointed's is not). The record keeps the Device slices written, refused.
* **An undifferentiated `jit` argument in a frozen field** (a source amplitude) was refused, though
  exact (rel 2.4e-10): staging tracers are let through (a differentiated one JAX refuses itself).
* **Forward-mode detection** by the `JVPTracer` class refused plain `jax.grad` when JAX linearizes by
  JVP (`jax_use_direct_linearize=False`; exact once let through, rel 8.7e-9): checked with the flag.
* **Docs.** Forward over reverse (`jax.hessian`, `jax.jvp(jax.grad(f))`) works and matches
  checkpointed (rel 9e-9); the reverse-over-reverse Hessian the message recommended fails in FDTDX's
  time loop for every method. Under `jax.jit` forward mode is JAX's own `TypeError`. The functional
  entry points' `diagnostics` are written by gradient calls.

Refuted: the drop-in's three snapshots of every phasor detector are its documented cost (still below
checkpointed's); a duplicate progress report at a window boundary (native reports are noisier); the
probe refusing an input `apply_params` overwrites anyway (the documented contract).

## Objective currents in a PML (refused)

An objective's adjoint current inside a PML breaks the reciprocal pairing where the PML is lossy:
FDTDX grades it as (d/L)^3 from zero at the interface, so its first cell is exact. Every block of
an objective's adjoint current is checked with `objective.pml_weight` (the local CPML strength
`|a|`, normalised to the PML's peak) and refused wherever it is not zero
(`validation.check_objectives_outside_pml`). A mode port whose evanescent tail sits in the z-PML
reads no lossy cell (weight 0) and stays admitted (exact, 6.8e-05 at 300 fs, converging). Box far
field (faces at cells 6 and 17 of 24), GPU float64, 150 fs: PML 4-6 exact (5.6e-07 to 6.1e-07);
PML 7, one cell deeper, 5.4e-04, which the former share-weighted warning let pass (share 6.0e-03 under
its 1e-2 tolerance); PML 8 1.0e-02. The ceviche corner's full cross-section monitors were 38% wrong
under one warning; cropped to the interior, 5e-6.

## The internal design detector

Each setting was a user-facing, silent error before the detector became internal
(`design.DESIGN_DETECTOR_SETTINGS`):

* components other than exactly (Ex, Ey, Ez): rel 3.80 at cosine -0.473 (the kernel contracts
  the component axis against a scalar permittivity);
* `scaling_mode="continuous"`: a pure scale error at cosine 1.0;
* `exact_interpolation=True`: co-located E where the kernel needs raw Yee E;
* `inverse=True`: never updated in a forward run, gradient exactly zero;
* a restricted `switch`: rel 7.9e-05; an `apodization`: rel 9.2e-01, both at cosine ~1;
* `wave_characters` different from the monitor's: rel 0.90 (650 vs 600 nm);
* complex64 on a float64 run: 12x worse accuracy, hence `dtype` follows the simulation.

Rewriting an existing detector with `aset` instead of placing a fresh one left its
`place_on_grid` caches stale; with an apodization that was a silent rel 0.92.

Components are stored canonically (Ex..Hz) whatever order `components` declares, and the
adjoint current must follow the stored order: `components=("Hx", "Ez")` driven in declared
order was rel 1.006 at cosine 0.27, after 1.18e-06.

## Scales

Uncorrected best-fit scale against `run_fdtd`, 24^3 float64: 6.19e+05, 1.27e-03 and 7.87e+02
for the three non-pulse monitor x design combinations; all four at 1.169e-06 after.

## Magnetic factor

Hx objective: plain sign flip 1.04e-01, with `exp(+i w dt/2)` 2.26e-01, with `exp(-i w dt/2)`
6.3e-07 at cosine 1.000000000. With `exact_interpolation=True`, dropping the time-average
factor `(1 + exp(+i w dt)) / 2` costs rel 8.5e-02 at cosine 0.996 and its conjugate rel
1.8e-01, both silent; an Hx monitor recorded as if raw was rel 2.0e-01 at cosine 0.980
(8.3e-07 with both).

Those numbers are from one frequency, with the H current's carrier at update_H's half-integer
time and one half step in the factor. That is exact only where the amplitude solve is diagonal:
the solve models integer times, so with several frequencies whose windowed spectra overlap the
off-diagonal terms kept a phase error `(w_f - w_g) dt / 2`. Measured, silent: Hx at 594/606 nm,
150 fs, rel 3.5e-03 (scale 1.0035); the periodic mode port 1.7e-03 at 150 fs and 6.8e-05 at
300 fs, put down to slow ModeOverlap convergence; in 12^3 cells (review, 20 fs) raw (Hx, Hy)
at 140/160 nm 2.6e-03, Hy at four wavelengths 3.9e-02, a box far field 1.1e-03. Now the carrier
sits at the integer step for H too and the factor carries both half steps, `-exp(-i w dt)`:
594/606 nm and the mode port (150 and 300 fs) 2.3e-08 and 1.5e-08; every E-only gradient is
bit-identical. On the user's problems (GPU, float32): the 90/10 splitter (six wavelengths
1.31-1.65 um, 800 fs) against its recorded checkpointed gradient, directional derivative, 1.8e-04
before and 1.3e-06 after (param_fn 8.8e-08); the colour splitter's far field against production
at 80 fs 1.1e-02 -> 7.9e-03 to 9.4e-03, and against the converged gradient at 160 fs (DC-free)
9.0e-04 -> 1.9e-04 (06-color-splitter.md); the test-bed mode ports unchanged at their float32
floor (1e-05). With it the looser parity gates (1e-4, 1e-3) of the magnetic, near-to-far, flux
and official-pipeline tests measured 3e-08 to 3.3e-06, and went to 1e-5.

## Refusals

* **Restricted objective switch.** Recording from 20 fs of 150 fs: rel 1.8e-01 at cosine
  0.984; from 40 fs: rel 8.2e+02 at cosine 0.008. Neither raised.
* **`dft_subsample` aliasing.** Only the principal term of the strided DFT is transposed, which
  is exact only if the fields have no content at the frequencies the stride folds onto an
  objective frequency, `2 pi m / (k dt) +- w`. The former rule (at least 4 samples per period of
  the monitor's own frequencies) admitted a stride-5 monitor under a second source band at
  `1 / (5 dt) - f`: rel 1.0009 at cosine -0.628, the gradient pointing the wrong way, and
  `dft_subsample='auto'` 0.947; recorded every step, 1.6e-6. Now refused when a source's sampled
  waveform has more than 1e-6 of its objective-frequency spectrum at a folded frequency
  (`check_source_waveforms`); a narrowband source clear of every image keeps its stride.
* **Bloch wave vector.** Rel 1.24 at cosine 0.35 (200 fs) and 1.54 at 0.37 (500 fs), forward
  value exact: the adjoint needs the opposite wave vector.
* **Full 3x3 tensors.** For nine components the adjoint current's `inv[axis]` picks xx, xy, xz
  instead of the diagonal, and a full conductivity tensor switches FDTDX to the coupled
  anisotropic update whose lossy factor is a matrix. Refused, not measured.
* **Dispersive Device material.** 24^3, one Lorentz material: forward FoM off 15%, gradient
  rel 0.82 at cosine 0.67.
* **Design region not covering every Device** (parameter level; warned at the phasor level):
  one cell short rel 8.0e-01, shifted by a cell 4.6e-01, a second Device left out 7.0e-01.
* **Stock source inside the design region.** A dipole at its centre: rel 1.16
  (`03-production.md`, correction 2).
* **Cell widths varying along an axis.** FDTDX's curls scale E by the primal and H by the
  dual (averaged) widths, a pairing the adjoint currents and the kernel do not weight. The
  24^3 test scene with x widths varying by +-20%: rel 1.49e-01 at cosine 0.9916, scale 0.911;
  by +-5%: rel 3.82e-02 at cosine 0.99947, scale 0.979; forward value exact, nothing raised
  (CPU, float64). One width per axis (`QuasiUniformGrid`, dz = 40 nm) is exact, 1.55e-07.
* **Design region in a PML.** The same scene, PML x 0..4, Device x 1..9: rel 2.51e-01 at
  cosine 0.968, scale 0.927, all of it on the three PML cell layers (rel 5.1e-01 there,
  2.2e-07 on the rest). A Device touching the PML (x 4..12) is exact, 3.0e-07.
* **`inverse=True` objective monitor.** It records during a backward run only: FoM 0 and
  checkpointed's gradient 0, the reciprocity gradient 7.8e-03 in norm, silently.
* **`phasor_fn(inv_eps)` where `apply_params` writes a Device's conductivity** (a lossy
  material, or loss placed under the Device): the placed conductivity was kept, FoM -5.56e-02
  against -2.32e-03 (24x), nothing raised. Now refused; pass both arrays `apply_params` returns.
* **The drop-in runs every scene refusal.** Without each, `GradientConfig("reciprocity")`
  (GPU, float64): x widths +-20% rel 2.7e-01 at cosine 0.992; a Lorentz Device material
  3.0e-01; a dipole in the Device 1.41 at cosine -0.05. The grid check reads the static
  `RectilinearGrid._is_uniform` and `_uniform_axes`, so it holds when a config set inside `jit`
  traces the grid edges; it used to need concrete edges and refused a `QuasiUniformGrid` there
  as "traced". A grid uniform overall is accepted whatever its per-axis jitter below the
  tolerance: the curls then apply no metric, and the gradient is bit-identical to `UniformGrid`'s.
* **Dispersive Device materials at the phasor level.** `reciprocity_phasor_fn` now refuses
  them and zeroes the Device cells' ADE coefficients, as `apply_params` does; with both arrays
  `apply_params` returned it gave FoM 1.08x (Lorentz Device) and 0.64x (Lorentz block under
  the Device) of `apply_params -> run_fdtd`, silently (12^3, 20 fs).

* **Differentiated inputs the drop-in does not carry** (fail fast, 2026-09-24), each silent
  before: `jax.grad` with respect to `arrays.inv_permittivities` or `electric_conductivity`
  themselves, a background parameter or a blur applied after `apply_params`: 97.8% to 99% of
  the gradient norm lay outside the Devices and came back zero (rel 0.99), and a blur corrupted
  the Device part as well (0.15). Refused unless the differentiated array is the very object
  `apply_params` returned (`initialization.written_by_apply_params`, a weak registry by id).
  Differentiated dispersion coefficients (a static Lorentz block): zero here against a
  checkpointed gradient finite differences confirm; refused, with every other differentiated leaf
  but the fields and detector states the run resets. `apply_params` now writes the coefficients of
  a Device without dispersive materials as constants, so the admitted dispersive-block workflow
  does not trip this (same values and gradients natively). A traced value in a frozen object field
  (a source amplitude) and forward-mode tracers: refused instead of an `UnexpectedTracerError` and a
  generic `TypeError`. A `Recorder` under `method="reciprocity"`: refused (it filled the reversible
  recording every step, 13.7x the memory, for nothing).
* **TFSF box.** A `TFSFPlaneSourceRegion` around the Devices was refused as "never applied": the
  check read the plane source's `_E`/`_H`, which a box never sets, and the source-in-design check
  tested the box's volume, which contains the scatterer. It now reads the box's `_face_*` fields
  and tests its faces; box around the Device: rel 1.7e-08 (float64, CPU).

## Paths no isotropic scene reaches

Breaking any of these passed the whole suite until `TestAnisotropicAndMagneticMaterials`
(parameter level, 24^3, CPU): the kernel summing the component axis of a three-row
`inv_permittivities`, rel 2.0 at scale 3.0 and cosine 1.0; the adjoint current's injection
factor read from row 0, rel 2.0e-01 (scale 1.17) at a stock monitor over a (2, 3, 4) block and
9.4e-01 (scale 1.92) at a raw monitor over a lossy one; the lossy divisor read from row 0,
rel 2.1e-01 (scale 0.80); the H current ignoring `inv_mu` at an Hx monitor over a mu = 2
block, rel 1.0 at scale 2.0 and cosine 1.0. Overlapping named design regions must set, not
add, their shared cells: adding was rel 9.7e-01 at scale 1.93.

## Corrections

* **Lossy objective monitor.** FDTDX adds sources after the lossy division, so the current in
  a lossy monitor cell was `1 + a` too strong. sigma_E 1e5 around a one-cell Ez monitor:
  rel 2.39e-01 at cosine 1.0000000000, best-fit scale 1.239 = `1 + a`, after 8.7e-11; loss on
  two of three cells of a line monitor rel 1.50e-01 at cosine 0.99922, after 1.0e-06;
  sigma_H 6e9 around an Hx monitor rel 2.28e-01 (scale `1 + b`), after 5.4e-07; E and H loss
  under a stock (Ez, Hx) line monitor rel 6.62e-02, after 9.3e-07.
* **Dispersive block under a Device.** `apply_params` zeroes the Device cells' ADE
  coefficients on every call; the solves kept the block's: forward FoM off 60%, gradient rel
  8.3e-01 at cosine 0.66, nothing raised (half the Device over the block: 4.6e-01); after,
  2.6e-06.

## Lossy Device materials

`apply_params` interpolated a Device's permittivity and dispersion but never wrote its
conductivity: a lossy Device material was lossless in every gradient method (FoM equal to the
lossless one), and a Device over a lossy block kept the block's loss. It now writes the scaled
conductivity (`sigma * c0 dt / courant`) with the permittivity's weights, and on the discrete
path gathers it by index without a straight-through term, so a discrete Device's gradient is
unchanged. Magnetic conductivity and permeability of Device materials are still not written,
in every method alike.

`update_E` steps `E1 = ((1 - a) E0 + c inv_eps curl H) / (1 + a)`, `a = c sigma eta0 inv_eps / 2`.
At fixed `sigma`, `dE1/d inv_eps = (E1 - E0) / (inv_eps (1 + a))`, and the `1 + a` cancels
against the lossy injection, so the permittivity kernel stays exact with a design-dependent
`sigma`. `dE1/d sigma = -c eta0 inv_eps (E0 + E1) / (2 (1 + a))` gives the conductivity kernel
`eta0 (1 + exp(+i w dt)) / 2`: same pairing, no `1 / inv_eps^2`.

Parameter level against `apply_params -> run_fdtd(checkpointed)`, GPU, float64, 150 fs, FoM
bit-equal in every case, cosine 1.0000000000 and best-fit scale 1 +- 7e-07 throughout:

| scene (24^3, box 24x24x30) | sigma (S/m) | rel L2 | rel L2 without the conductivity term |
| --- | --- | --- | --- |
| dipole, PhasorDetector, eps 2.25 Device | 1e3 / 1e4 / 1e5 / 1e6 / 1e7 | 4.5e-07 / 3.6e-07 / 4.1e-07 / 3.2e-07 / 2.2e-07 | 0.05 / 0.58 / 1.31 / 0.98 / 1.00 |
| eps 1 absorber (conductivity term alone) | 1e3 / 1e5 / 1e7 | 4.6e-07 / 5.0e-07 / 2.2e-07 | 1 |
| + lossy slab below / under half the Device | 1e5 | 6.5e-07 / 5.9e-07 | 1.36 / 1.32 |
| monitor inside the lossy Device | 1e5 / 1e7 | 4.6e-07 / 7.3e-07 | 1.02 / 1.00 |
| stock 5-face FieldProjectionAngleDetector box | 1e4 / 1e5 / 1e6 / 1e7 | 5.7e-07 / 6.2e-07 / 4.6e-07 / 6.4e-07 | 0.31 / 0.91 / 0.98 / 1.00 |
| box on a lossy substrate | 1e5 | 6.2e-07 | 0.93 |

float32: 0.8e-06 to 1.5e-06 against float32 checkpointed; against float64 checkpointed it
tracks float32 checkpointed (box: 9.2e-05 and 9.2e-05, the float32 FDTD floor).

Cost, RTX 3080 Ti, 64x64x72, 1574 steps, lossy Device (32x32x8) on a lossy substrate, stock
box far field, everything jitted, min of 5 interleaved calls: float32 forward 0.137 s,
reciprocity value and gradient 0.355 s (2.60x), checkpointed 3.35 s (24.5x), rel 2.5e-06;
the same scene with a lossless Device 0.353 s (2.59x), so the conductivity kernel costs
nothing. float64: 1.46 s (2.39x) against 7.87 s (12.9x), rel 2.1e-06 (1.8e-06 at 300 fs;
lossless Device 1.4e-06).

With the magnetic carrier fix (7e06ecf; session scratchpad `finalbench/bench.py`): the same
layout (64x64x72, 150 fs, 1574 steps; Device 32x32x8 air / eps 2.25 + 1e5 S/m on an eps 2.1 +
1e4 S/m substrate; stock five-face box, 550/600/650 nm), `apply_params -> run_fdtd ->
project_all` jitted, GradientConfig("reciprocity") against ("checkpointed", 8) and (.., 40),
RTX 3080 Ti, median of 3 steady calls, FoM bit-identical throughout:

| | reciprocity | checkpointed 8 | checkpointed 40 | rel L2 vs checkpointed |
| --- | --- | --- | --- | --- |
| float32 | 0.306 s (2.20x fwd), 82 MiB | 5.60 s, 270 MiB | 3.63 s, 836 MiB | 2.0e-06, cos 0.999999999999 |
| float64 | 1.243 s (2.12x fwd), 123 MiB | 11.23 s, 389 MiB | 7.27 s, 1634 MiB | 2.1e-08, cos 1.000000000000 |

The float64 gap was 2.1e-06 before the fix (three wavelengths, H read by the box).

The same layout at 32x32x40 put the box's side faces inside the 8-cell PML: rel 1.36e-01 at
cosine 0.992, lossless or lossy, at 60, 150 and 300 fs alike; with a 3-cell PML 2.4e-06. An
objective monitor with cells beyond the PML's zero-loss first cell is now refused (see above).

**Etched Devices** interpolate the conductivity from the placed background, which
`_init_arrays` keeps in `initial_electric_conductivity` like `initial_inv_permittivities`.
Reading the current array instead etched it again on every application to returned arrays
(an optimization loop feeding `run_fdtd`'s output back): Device mean 2.49e-03, 1.47e-03,
9.58e-04 over three applications with the same parameters, FoM -2.31e-03 to -6.32e-03.
Where etching cannot change the conductivity, no backup is kept and etching leaves it as placed:
every etched Device's placed background (checked on the concrete placed arrays) and every Device
overlapping it have its etch material's conductivity. Checking the placed arrays alone left a
lossy Device's 1e5 S/m in fully etched cells above it, at eps 1. `extend_material_to_pml` extends
both backups, or `apply_params` restoring them undid it, and makes the conductivity backup when
it extends loss under an etched Device. The choice is per scene: once one etched Device etches
loss, a reversible gradient through any etched Device raises (loudly; checkpointed and
reciprocity are exact there).

**`GradientConfig("reversible")`** does not differentiate the conductivity; it closes over it.
Once `apply_params` wrote it, any conductive scene with a Device made it a function of the
parameters, and reversible raised `UnexpectedTracerError`. A conductivity that does not depend
on the parameters is now written as a constant (a Device whose materials share one) or left as
placed (etching a background of the etch material's own), with the same values, so reversible
is as before there: an etched air Device next to a lossy block matched checkpointed upstream
(rel 5.7e-07) and still does. A lossy Device, or one etching loss, raises a
`NotImplementedError` when differentiated (its forward runs). Upstream and pre-existing:
before `ebdfc00`, reversible on a lossless Device next to a lossy block diverged in its reverse
reconstruction (rel inf with `num_checkpoints_reversible=0`), and a lossy Device was lossless
in the forward.

**The gradient is zero outside the Devices** also for `jax.grad` with respect to
`arrays.inv_permittivities` itself (`GradientConfig("reciprocity")`): checkpointed's norm
there was 13x the Device cells' (7x for the conductivity). Device parameters, the documented
input, see only the Device cells.
