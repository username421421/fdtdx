# Near-to-far, and the magnetic side it needed first

Fourth in the series. `03-production.md` listed near-to-far as the one gap
blocking the colour splitter. It now works, and getting there needed the
magnetic side first.

## Result

Objective is a figure of merit on the projected far field of a five-face
near-to-far box, using FDTDX's own `FieldProjectionAngleDetector`. Compared with
`apply_params -> run_fdtd(GradientConfig(checkpointed)) -> jax.grad` on the same
placed scene:

| quantity | value |
| --- | --- |
| forward far field | identical to the last digit |
| gradient relative L2 | `6.4e-07` |
| gradient cosine | `1.0000000000` |
| sign agreement | 100% |

Cost is unchanged at **two forward solves**, not one per face. All five faces
are driven in a single adjoint solve: their adjoint currents superpose, and so
do their contributions to the design gradient.

## How it works, and why nothing had to be reimplemented

The `custom_vjp` boundary already sat at the monitor's raw phasors. A box-mode
projection detector stores one phasor array per face under its own state key, so
the boundary just returns the whole state dict, and
`FieldProjectionAngleDetector.project(state, theta, phi)` runs above it as
ordinary JAX. The projected far field is therefore FDTDX's own code, not a
reimplementation, and JAX differentiates it:

```python
fn = reciprocity_phasor_fn(arrays, objects, config, key,
                           objective_detectors="ff")   # design region: the Device(s)

def loss(inv_eps):
    out = ff.project(fn(inv_eps), theta, phi)
    return -jnp.sum(jnp.abs(out["power"]))
```

Internally each objective detector expands into *channels*, one per state key:
one for a plain `PhasonDetector`, one per included face for a box. Each channel
gets its own `AdjointCurrentSource`, a one-cell-thick slab on that face's normal
axis, at the face's absolute grid slice.

## The magnetic side was the real prerequisite

Box-mode projection concatenates E and H, so its adjoint source drives all six
components. Magnetic objectives had never been exercised and did not work. Two
factors were missing, and the combination is

    magnetic_factor(w) = -exp(-i * w * dt / 2)

**The sign** comes from Lorentz reciprocity's asymmetry between the electric and
magnetic pairings,

    int(E_a . J_b - H_a . M_b) = int(E_b . J_a - H_b . M_a)

so a cotangent on a magnetic component maps to an adjoint magnetic current of
the opposite sign. This is in section 7 of `01-reciprocity-physics.md`; I had
written it down and then not applied it.

**The phase** is a Yee half-step. With `exact_interpolation=False` the detector
stores the post-update H, which lives at n+1/2, but weights it with the
integer-step kernel `exp(+i w n dt)`, while `update_H` injects the adjoint
magnetic current at `time_step + 0.5` (`fdtd/update.py:813`).

Measured on an Hx objective:

| factor | rel L2 | cosine |
| --- | --- | --- |
| none | 1.91 | -0.998 |
| sign flip only | 1.04e-01 | 0.998 |
| sign flip and half-step | **6.3e-07** | **1.000000000** |

The `-0.998` is worth dwelling on: without the sign the gradient is almost
perfectly anti-correlated, so an optimizer would walk confidently uphill.

Every component set now agrees at about 5e-07: `(Ez,)` 4.1e-07, `(Hx,)` 6.3e-07,
`(Hy,)` 6.3e-07, `(Ez, Hx)` 4.1e-07, all six at once 4.7e-07.

Two crashes on the magnetic path also had to be fixed, both from code that had
never run: `update_H` is called with a non-integer `time_step`, which broke the
window lookup, and a scalar `inv_permeabilities` has no leading axis to index.
The carrier phase keeps the exact half-integer time, which is what puts the
injection on the right half-step; only the envelope index is floored.

## Exact interpolation (formerly the one caveat)

`FieldProjectionAngleDetector` fixes `exact_interpolation=True`. That
interpolation runs in `update_detector_states`, outside the detector's own
`update`, so it used to be refused, with `aset("exact_interpolation", False)` as
the workaround (a second-order change to the forward far field: 1.53e-02 at 12
cells per wavelength, 3.39e-03 at 24). It is now transposed
(`fdtdx.adjoint.recording`), so the detector runs as FDTDX configures it.

The co-location map `R` is linear, purely spatial and component-diagonal, so it
commutes with the DFT. The H half of it records `(H_prev + H) / 2`, whose DFT is
`(1 + exp(+i w dt)) / 2` times the post-update one: a per-frequency factor, not
two injections at different half-steps. Composed with the existing half-step
factor the magnetic target becomes `-exp(-i w dt/2) (1 + exp(i w dt)) / 2 =
-cos(w dt/2)`, kept in the code as the product of the two named terms.

`R^T` is `jax.linear_transpose` of FDTDX's own `interpolate_fields`, on the same
branch `update_detector_states` takes: the `(s-1, e+1)` block for an interior
detector, and the padded whole domain (`pad_fields_with_symmetry_mirror`:
periodic wrap, zero halo, mirror) for one touching an edge. The cells its
cotangent lands on are found numerically at setup and one adjoint current is
placed per contiguous block, so a stencil that wraps around a periodic axis gets
two. For an interior face the block is `(s-1, e)` in x and y and `(s, e+1)` in z.

Measured against `run_fdtd(checkpointed)` through `reciprocity_param_fn` on the
colour splitter's layout (UniformPlaneSource with `normalize_by_energy`, Device
on a substrate, stock box, `z-` excluded, `project_all` power at five angles,
two wavelengths), float64 GPU: rel 6.1e-07 at 150 fs, 2.7e-07 at 300 fs, cosine
1.0000000000, forward identical. float32: 1.3e-06.

## Still open

* Design-dependent loss, i.e. Meep's `MaterialGrid(damping=...)`, where sigma is
  a function of the design variable.
* Dispersive design regions.
* `reduce_volume` and diffraction-order detectors still raise.
* Not compared against Meep, and not yet run on the RGB metalens itself.
