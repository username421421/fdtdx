# The colour splitter: is its gradient correct?

The user's D4 structural-colour optimization (`/home/zhuwei/fdtdx/color_splitter_d4`, read-only),
run on this branch without touching their code: their `build_scene`, their fabrication pipeline
(raw -> D4 -> masks -> cone -> projection -> density), their `structural_color_metrics` with the
empty-cell reference, and their production gradient `optimize.make_value_and_grad_functions`
(nlopt stubbed; it never calls it). Balanced preset: 50 nm, 80 fs, 1.35 M cells, 839 steps,
float32, GPU. Structures: their initial state (`_initialize_raw`, grey, beta 4) and the
`runs/balanced_480` final design (validation said `fabrication_ready=false`). Harness:
session scratchpad `colorsplitter/` and `cscorrect/` (`cc.py`, `cc64.py`, `ccan.py`).

## It runs on the unmodified scene

`reciprocity_param_fn(scene.arrays, scene.objects, config, key, objective_detectors="far_field")`
straight from `build_scene`: the stock `FieldProjectionAngleDetector` box (exact interpolation,
z- excluded, `dft_subsample="auto"`), the `UniformPlaneSource` placed over the Device, the Device
as design region. Forward FoM bit-identical to production in every run. The 10 fs smoke preset
is refused (amplitude solve cond 6e15).

## Agreement with production, and with the truth

Raw MMA variables, rel L2 (cosine >= 0.99993 in every row):

| run | reciprocity vs production (same run) |
| --- | --- |
| user's pulse, 80 fs, initial | 1.1e-02 |
| user's pulse, 80 fs, balanced_480 | 1.1e-02 |

The 1% is not reciprocity error. Against a converged reference (production checkpointed at
1280 fs with a DC-free pulse), balanced_480:

| pulse, T | production vs truth | reciprocity vs truth |
| --- | --- | --- |
| user's, 80 fs (production today) | 1.18e-02 | 1.09e-02 |
| user's, 640 fs | 7.5e-03 | 8.3e-03 |
| DC-free, 160 fs | 2.4e-04 | 9.0e-04 |
| DC-free, 320 fs | 2.3e-04 | 2.3e-04 |

Float64, DC-free, 320 fs, balanced_480: reciprocity vs production rel **1.9e-05**, cosine
0.9999999998. Grey initial structure, float64, DC-free: 3.4e-03 at 320 fs, 1.8e-03 at 640 fs,
with reciprocity moving 1.5e-04 between the two and production 3.1e-03 (it converges slower).

**The user's source carries DC.** `GaussianPulseProfile` at centre 1.875 / width 1.25
(normalized) is a few-cycle pulse with |A(0)|/max|A| = 0.59. The static remainder never decays,
floors both gradients at ~1e-2 and the FoM at ~1.5e-2, and keeps the objective-stage
`ConvergenceWarning` firing. A sine carrier centred on the envelope removes it: on the PROFILE's
own `WaveCharacter` only, `phase_shift = pi/2 - 2*pi*f0*t0` with `t0 = 6*sigma_t`,
`sigma_t = 1/(2*pi*fw)` (scene.py shares `center_wave` with the source's `wave_character`, whose
phase `get_amplitude` adds on top, so putting it on the shared object doubles it).
With it, the reference-normalized response is within 3.3e-04 of the converged one at every
wavelength by 160 fs; with the user's pulse, on balanced_480, it stays 1.1e-02 off at 160 fs and 1.4e-02 at 640 fs.

## Float32 is the largest error at production settings

Same method, float32 vs float64 (user's pulse, 80 fs): FoM 0.7% (initial) and **4.7%**
(balanced_480: 0.2159 vs 0.2265); gradient 3.1-3.5%, identical for production and reciprocity
(cosine 0.9994). The device run's far-field power moves up to 4.2% at single wavelengths, the
empty-cell reference ~0.5%. Optimizing in float32 is fine for direction; report FoM values from a
float64 evaluation.

## Cost

Balanced, exclusive GPU, median of 3 steady value+grad calls: forward 0.394 s, reciprocity 1.17 s,
production checkpointed 8.37 s (7.1x). Fabrication (25 nm, 120 fs, 10.8 M cells): forward 9.6 s,
reciprocity 27.5 s, production checkpointed (3 checkpoints, the preset's setting) ~4000 s per
call (~145x).

## An optimization with each gradient

Their production value-and-gradient, and the same scene with the one string changed, driven by
the same simple optimizer (nlopt is not in the fork venv, so not their MMA): projected normalized
ascent `x <- clip(x + 0.05 g / max|g|, 0, 1)`, 15 steps from their `_initialize_raw` start at the
epoch-1 values (beta 4), balanced preset, user's pulse, 80 fs, float32, GPU (session scratchpad
`cstraj/`). FoM -1.622 -> +1.1147 (checkpointed) and -1.622 -> +1.1140 (reciprocity): gain ratio
0.9997, the per-step FoM gap at most 3.6e-03 and shrinking to 6.9e-04 by step 15, final designs
0.9% of the distance travelled apart. Wall time for the 16 evaluations with compile: 154 s
against 37 s.

## Recommendation

DC-free pulse + 160 fs: reciprocity within 9e-4 of the converged gradient at ~2.3 s per call,
against 8.4 s for today's production gradient, which is 1.2e-02 from it.
