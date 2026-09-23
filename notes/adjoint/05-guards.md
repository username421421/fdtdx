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
| `fdtdx.adjoint.reciprocity` (window, amplitude solve, kernel) | `fdtdx.adjoint.kernel`, with `dft_tail` and `ConvergenceWarning` |
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

## Convergence warning (`DEFAULT_TAIL_TOLERANCE = 1e-2`)

Colour splitter, real cell at 100 nm, float64, GPU, against checkpointed autodiff of the same
run; largest tail -> gradient rel, DC-free pulse: 1.35 -> 19 (22 fs), 0.30 -> 0.29 (40 fs),
6.8e-2 -> 0.14 (50), 1.65e-2 -> 8.8e-2 (80), 1.04e-2 -> 4.3e-2 (120), 5.5e-3 -> 2.0e-2 (160),
1.0e-3 -> 2.6e-3 (320), 1.75e-4 -> 1.2e-4 (640). The error is 2-5x the tail from 50 fs on,
so 1e-2 flags every run measured at 4% or worse and none at 2% or better. The adjoint design
tail was 1.46 at 22 fs, where the gradient was rel 20. With the DC-carrying production pulse
the objective tail floors at 3.1e-2 (a static remainder the far-field phasors never lose), so
that stage keeps warning while the gradient itself converges (5.1e-3 at 320 fs).

## PML warning (`PmlWarning`, same tolerance)

An objective's adjoint current inside a PML breaks the reciprocal pairing, but only where the
PML is lossy: FDTDX grades it as (d/L)^3 from zero at the interface, so its first cell is
harmless. A blanket refusal of monitors touching a PML was tried and removed: it refused a
mode port whose evanescent tail sits in the z-PML and whose gradient is exact (6.8e-05 at
300 fs, converging). The backward pass instead reports the share of each objective's
adjoint current weighted by the local CPML strength ``|a|`` (normalised to the PML's peak),
``objective_pml_share`` in the diagnostics. Box far field (faces at cells 6 and 17 of 24),
GPU float64, 150 fs, share -> gradient rel: PML 4-6 0 -> 5.6e-07 to 6.1e-07 (at 6 the
stencil enters the zero-loss first cell, 69% of one face's current); PML 7 6.0e-03 ->
5.4e-04; PML 8 2.9e-02 -> 1.0e-02. The mode port: 0. The share runs 3-10x the error, so the
1e-2 tolerance flags the 1% case and neither exact one.

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

## Refusals

* **Restricted objective switch.** Recording from 20 fs of 150 fs: rel 1.8e-01 at cosine
  0.984; from 40 fs: rel 8.2e+02 at cosine 0.008. Neither raised.
* **`dft_subsample` aliasing.** Only the principal term of the strided DFT is transposed.
  Against the exact gradient of the strided recording: rel 5.4e-07 to 5.7e-07 at 21 down to
  2.6 samples per period (forward aliasing at most 7.6e-09), but 4.5e-01 at 2.1 samples per
  period with a 0.1 f0 source (forward phasor 70% aliased) and 7.5e-01 at 2.6 with a 0.5 f0
  source. Refused below 4 samples per period, where FDTDX itself warns.
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

The same layout at 32x32x40 put the box's side faces inside the 8-cell PML: rel 1.36e-01 at
cosine 0.992, lossless or lossy, at 60, 150 and 300 fs alike; with a 3-cell PML 2.4e-06. An
objective monitor in the PML is not refused; it raises a `PmlWarning` (see above).
