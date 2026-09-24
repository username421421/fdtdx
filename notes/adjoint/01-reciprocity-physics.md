# Reciprocity and the adjoint source: the physics we are implementing

Working notes for the reciprocity gradient (branch `main`). This is the theory the
implementation has to reproduce, written so the code can be checked against it
line by line.

**Provenance.** Distilled from `Adjoint design basics - Reciprocity and Maxwell
Operators.pdf`, a 107-page AI chat transcript (the source ends with "ChatGPT can
make mistakes"). The physics below is standard and every formula in sections 1
to 8 has been re-derived independently before being written here; the
self-consistency checks are noted where they matter. Section 9 is our own
bridge to FDTDX and is not from the transcript.

Sign and phasor convention throughout: $e^{-i\omega t}$.

---

## 1. Reciprocity is transpose symmetry, not Hermitian symmetry

This is the point the whole method rests on, and it is the one most often
garbled.

Two different pairings:

| Pairing | Definition | Adjoint defined by | Adjoint is |
| --- | --- | --- | --- |
| Hermitian inner product | $\langle u,v\rangle = u^{*T} v$ | $\langle u, Av\rangle = \langle A^\dagger u, v\rangle$ | $A^\dagger = A^{*T}$ |
| Unconjugated bilinear | $B(u,v) = u^{T} v$ | $B(u, Av) = B(A^{T} u, v)$ | $A^{T}$ |

**Reciprocity is the statement $A^{T} = A$** under the unconjugated pairing. It
is *not* $A^\dagger = A$. Those coincide only when $A$ is real.

A complex matrix can be symmetric without being Hermitian:

$$
A = \begin{bmatrix} 1+i & 2 \\ 2 & 3+i \end{bmatrix},
\qquad A^{T} = A, \qquad A^\dagger \neq A .
$$

That is exactly the situation for lossy reciprocal Maxwell: $\varepsilon$ is
complex, so the operator is not Hermitian, but if the material tensors are
symmetric the operator is still reciprocal.

**Reciprocity is not time-reversal symmetry.** Loss breaks time reversal,
because a decaying wave cannot be run backwards without gain. Loss does not
break source-receiver symmetry: a lossy reciprocal waveguide still has
$S_{21} = S_{12}$. Keeping these separate is what makes the next section work.

Equivalent statements of reciprocity:

$$
u^{T} A v = v^{T} A u \quad \forall u,v
\qquad\Longleftrightarrow\qquad
A^{T} = A
\qquad\Longleftrightarrow\qquad
G_{ij}(\mathbf r, \mathbf r') = G_{ji}(\mathbf r', \mathbf r) .
$$

With a weighted pairing $B(u,v) = u^{T} W v$, reciprocity reads
$W A = A^{T} W$, so for symmetric $W$ the condition is $A^{T} W = W A$. This
matters in discretized form: you may see a weighted transpose symmetry rather
than a literal $A^{T} = A$, depending on how the curl operators and field
weights are defined.

## 2. Why this distinction is the whole point computationally

Take $A x = b$ and a real objective $F$.

**Hermitian route.** Introduce $y$ with $A^\dagger y = g$. Then
$\delta F = -2\,\mathrm{Re}[\,y^\dagger\, \delta A\, x\,]$. But
$A^\dagger = \nabla\times \mu^{-*}\nabla\times - \omega^2 \varepsilon^{*}$, so
running this adjoint simulation would require $\varepsilon \to \varepsilon^{*}$
and $\mu \to \mu^{*}$. **For a lossy material that converts absorption into
gain.** It is not a simulation you can hand to a forward solver.

**Transpose route.** Introduce $\lambda$ with $A^{T}\lambda = q$. Then
$\delta F = -2\,\mathrm{Re}[\,\lambda^{T}\, \delta A\, x\,]$. If the medium is
reciprocal, $A^{T} = A$, so the adjoint equation becomes

$$
A\lambda = q ,
$$

**the same operator, the same physical medium, the same solver.** That is what
reciprocity buys: not a simpler formula, but an adjoint problem your existing
forward solver can actually run.

The two adjoint fields are related by $y = \lambda^{*}$. Mathematically
equivalent, computationally not.

## 3. The Maxwell operator and why it is reciprocal

With $e^{-i\omega t}$:

$$
\nabla\times \mathbf E = i\omega \mathbf B, \qquad
\nabla\times \mathbf H = \mathbf J - i\omega \mathbf D,
$$

and for a linear local scalar material $\mathbf D = \varepsilon \mathbf E$,
$\mathbf B = \mu \mathbf H$, with $\varepsilon = \varepsilon' + i\varepsilon''$
possibly complex. Eliminating $\mathbf H$:

$$
\boxed{\;\bigl(\nabla\times \mu^{-1}\nabla\times \;-\; \omega^2\varepsilon\bigr)\mathbf E = i\omega \mathbf J\;}
\qquad
A \equiv \nabla\times \mu^{-1}\nabla\times - \omega^2\varepsilon .
$$

Reciprocity means
$\int \mathbf E_2 \cdot A \mathbf E_1\, dV = \int \mathbf E_1 \cdot A \mathbf E_2\, dV$,
with **no conjugation**. Two parts:

**Material part**, $-\omega^2\varepsilon$. Since $\varepsilon$ is a scalar,
$\varepsilon \mathbf E_1$ is a pointwise product and
$\mathbf E_2\cdot \mathbf E_1 = \mathbf E_1\cdot \mathbf E_2$. Symmetric.
Nothing here required $\varepsilon^{*} = \varepsilon$.

**Curl-curl part.** Use

$$
\int \mathbf F \cdot \nabla\times \mathbf G\, dV
= \int \mathbf G \cdot \nabla\times \mathbf F\, dV
- \oint_{\partial V} (\mathbf F \times \mathbf G)\cdot \hat{\mathbf n}\, dS,
$$

with $\mathbf F = \mathbf E_2$, $\mathbf G = \mu^{-1}\nabla\times\mathbf E_1$.
If the boundary term vanishes, this gives
$\int \mathbf E_2 \cdot \nabla\times\mu^{-1}\nabla\times \mathbf E_1\, dV
= \int \mu^{-1}(\nabla\times\mathbf E_1)\cdot(\nabla\times\mathbf E_2)\, dV$,
which is manifestly symmetric because $\mu$ is scalar. Integrating back gives
the result.

**So the hypotheses are:** linear, local, symmetric (e.g. scalar) $\varepsilon$
and $\mu$, and boundary conditions that kill the surface term (PEC, matched
periodic phases, radiation, or properly paired ports). Complex $\varepsilon$ is
fine. **Loss breaks Hermiticity, not reciprocity.**

The source-level consequence is Lorentz reciprocity:

$$
\int \mathbf E_2 \cdot \mathbf J_1 \, dV = \int \mathbf E_1 \cdot \mathbf J_2 \, dV .
$$

The field from source 1 measured by source 2 equals the field from source 2
measured by source 1. No conjugation appears.

## 4. The adjoint derivation

$F$ is real, $\mathbf E$ is complex, so $F$ is not holomorphic in $\mathbf E$
(e.g. $|E|^2 = E^{*}E$). Treat $\mathbf E$ and $\mathbf E^{*}$ as independent
(Wirtinger). For real $F$ the two terms are conjugates, so

$$
\boxed{\; dF = 2\,\mathrm{Re}\!\left[\left(\frac{\partial F}{\partial \mathbf E}\right)^{T} d\mathbf E\right] \;}
$$

Define the **adjoint source** as exactly that derivative:

$$
\boxed{\; b_{\mathrm{adj}} \equiv \frac{\partial F}{\partial \mathbf E} \;}
\qquad
A^{T}\lambda = b_{\mathrm{adj}}
\qquad\xrightarrow[\ A^{T}=A\ ]{}\qquad
A\lambda = \frac{\partial F}{\partial \mathbf E} .
$$

Because the physical equation is $A\mathbf E = i\omega \mathbf J$, realizing
$\lambda$ as the field of a *physical current* gives

$$
\boxed{\; \mathbf J_{\mathrm{adj}} = \frac{1}{i\omega}\,\frac{\partial F}{\partial \mathbf E} \;}
$$

And the gradient. From $A(p)\mathbf E(p) = b$ with $b$ independent of $p$:
$A\,\frac{d\mathbf E}{dp} = -\frac{dA}{dp}\mathbf E$. Substituting and using
$A^{T} = A$:

$$
\boxed{\;
\frac{dF}{dp} = -2\,\mathrm{Re}\!\left[\lambda^{T}\frac{dA}{dp}\mathbf E\right]
+ \left.\frac{\partial F}{\partial p}\right|_{\mathbf E}
\;}
$$

For permittivity-only design, $\frac{dA}{dp} = -\omega^2 \frac{dM_\varepsilon}{dp}$, so

$$
\frac{dF}{dp} = 2\omega^2\,\mathrm{Re}\!\left[\lambda^{T}\frac{dM_\varepsilon}{dp}\mathbf E\right]
\qquad\text{continuum:}\qquad
\frac{\partial F}{\partial p} = 2\omega^2\,\mathrm{Re}\!\int_{\text{design}} \lambda(\mathbf r)\cdot \mathbf E(\mathbf r)\,\frac{\partial \varepsilon(\mathbf r)}{\partial p}\, dV .
$$

Note there is **no complex conjugate** in the $\lambda \cdot \mathbf E$ overlap.
That is a direct consequence of using the transpose pairing, and it is the most
common place to introduce a bug by reflex.

## 5. Why the adjoint source lives at the monitor

$\partial F/\partial \mathbf E$ answers one question: *which field components
does $F$ directly read?* Split the field vector,

$$
\mathbf E = \begin{bmatrix}\mathbf E_{\text{design}} \\ \mathbf E_{\text{monitor}} \\ \mathbf E_{\text{other}}\end{bmatrix},
\qquad F = F(\mathbf E_{\text{monitor}}),
$$

then $\partial F/\partial \mathbf E_{\text{design}} = 0$ and
$\partial F/\partial \mathbf E_{\text{other}} = 0$. So $b_{\mathrm{adj}}$ is
**nonzero only on the monitor**.

This does not mean only the monitor affects $F$. The design region affects $F$
*indirectly*, through Maxwell's equation, and that is precisely what the adjoint
field carries: $b_{\mathrm{adj}}$ is local, but $\lambda$ solving
$A\lambda = b_{\mathrm{adj}}$ is nonzero throughout the device.

In general, if the monitor extracts $y = M\mathbf E$ and $F = F(y, y^{*})$, then

$$
\frac{\partial F}{\partial \mathbf E} = M^{T}\frac{\partial F}{\partial y} ,
$$

and $M^{T}$ is what scatters the derivative back onto the monitor grid points
and leaves everything else zero. **This is the single most important structural
fact for our implementation**: we never hand-derive where the adjoint source
goes. We transpose the monitor's own extraction operator.

The extra term $\left.\partial F/\partial p\right|_{\mathbf E}$ is the *direct*
dependence of $F$ on the design at fixed field. In photonic inverse design it is
almost always zero, because the FoM is a function of measured fields only and
does not mention $\varepsilon$ explicitly.

## 6. Worked adjoint sources

The pattern: write $dF$, force it into the form
$2\,\mathrm{Re}[(\cdot)^{T} d\mathbf E]$, and read off the bracket.

| FoM | intermediate | $\partial F/\partial \mathbf E$ |
| --- | --- | --- |
| $\lvert E_z(\mathbf r_0)\rvert^2$ | $s = E_z(\mathbf r_0)$ | $s^{*}\,\hat{\mathbf z}\,\delta(\mathbf r - \mathbf r_0)$ |
| $\lvert \hat{\mathbf e}\cdot\mathbf E(\mathbf r_0)\rvert^2$ | $s = \hat{\mathbf e}\cdot\mathbf E(\mathbf r_0)$ | $s^{*}\,\hat{\mathbf e}\,\delta(\mathbf r - \mathbf r_0)$ |
| $\lvert m^{T}\mathbf E\rvert^2$ (mode overlap) | $a = m^{T}\mathbf E$ | $a^{*} m$ |
| $\mathrm{Re}(m^{T}\mathbf E)$ | $a = m^{T}\mathbf E$ | $\tfrac{1}{2} m$ |
| $\int_{\text{mon}} w(\mathbf r)\lvert E_z\rvert^2 dS$ | | $w(\mathbf r)\,E_z^{*}(\mathbf r)\,\hat{\mathbf z}$ |

Two things worth internalizing:

- For any $F = \lvert a\rvert^2$ with $a = m^{T}\mathbf E$, the adjoint source is
  $a^{*}m$: **the monitor profile injected backwards, weighted by the conjugate
  of the forward overlap.** The conjugate comes from differentiating
  $a^{*}a$, not from the adjoint formalism.
- The $\tfrac{1}{2}$ in the $\mathrm{Re}(m^{T}\mathbf E)$ row exists only
  because our convention carries an explicit factor $2\,\mathrm{Re}[\cdot]$.
  Change the convention and that factor moves.

## 7. When the FoM also depends on H

This is our v1 scope, so it matters. $F = F(\mathbf E, \mathbf H, p)$ gives two
field-derivative pieces, but $\mathbf H$ is not independent. From
$\nabla\times\mathbf E = i\omega\mu\mathbf H$,

$$
\mathbf H = B\mathbf E, \qquad B \equiv \frac{1}{i\omega}\mu^{-1}\nabla\times .
$$

Substituting $\frac{d\mathbf H}{dp} = B\frac{d\mathbf E}{dp} + \frac{dB}{dp}\mathbf E$
and collecting the terms that multiply $d\mathbf E/dp$:

$$
\boxed{\; b_{\mathrm{adj}} = \frac{\partial F}{\partial \mathbf E} + B^{T}\frac{\partial F}{\partial \mathbf H} \;}
$$

**If $\mu$ is design-independent** (our case; $dB/dp = 0$), that is the *only*
change. The gradient formula in section 4 is untouched. H-dependence does not
break the method, it just adds a term to the adjoint source.

If $\mu$ does depend on the design, there is one extra term,
$+2\,\mathrm{Re}[(\partial F/\partial \mathbf H)^{T}(dB/dp)\mathbf E]$.

**Alternative, fully general.** Keep $\mathbf E$ and $\mathbf H$ as independent
states, $x = [\mathbf E; \mathbf H]$, write Maxwell as a first-order residual
$R(x,p) = 0$, and solve

$$
\left(\frac{\partial R}{\partial x}\right)^{T}\Lambda
= \begin{bmatrix}\partial F/\partial \mathbf E \\ \partial F/\partial \mathbf H\end{bmatrix},
\qquad
\frac{dF}{dp} = -2\,\mathrm{Re}\!\left[\Lambda^{T}\frac{\partial R}{\partial p}\right] + \left.\frac{\partial F}{\partial p}\right|_{x} .
$$

This is the version to reach for if we ever touch nonlinear materials.

## 8. When this breaks

$dA/dp$ can be any correct derivative of whatever you are perturbing; the
gradient formula is more general than the reciprocity assumption. Reciprocity is
used for exactly one thing: replacing $A^{T}$ with $A$ so the adjoint solve runs
in the same medium.

**Nonreciprocal media** (magneto-optic, biased, gyrotropic): $\varepsilon^{T}
\neq \varepsilon$, so $A^{T} \neq A$. The adjoint simulation must use the
*transpose* medium, which physically usually means reversing the bias. For
nonzero Bloch vector the analogue is $k \to -k$.

**Nonlinear media**: there is no single linear $A$. You must linearize about the
forward solution, and the adjoint operator is the transpose of the Jacobian of
the nonlinear residual. Because $\varepsilon$ depends on $\mathbf E$, that
Jacobian is generally **not** symmetric even when the material is reciprocal, so
"run the same solver" fails. Use the first-order residual formulation.

---

## 9. What this means for the FDTDX port

Our own bridge, not from the source document.

**We are in the easy regime, and we should not let that hide bugs.** FDTDX's
colour-splitter and metalens scenes use real, non-dispersive, lossless
$\varepsilon$. There $A^{T} = A$ *and* $A^\dagger = A$, so the section 1
distinction does not bite and a conjugation error can be invisible. It becomes
load-bearing the moment absorbing materials appear. Any test that gates the
implementation should therefore include a complex-$\varepsilon$ case even though
v1 does not need one.

**"A" is not the same object in a time-domain solver.** Sections 1 to 8 are
frequency-domain. FDTDX steps in time. What reciprocity gives us is the
*frequency-domain* statement that the adjoint field can be produced by a
physical source in the same medium; the implementation realizes it by running a
second forward-in-time simulation and taking its DFT to get $\lambda(\omega)$.

**The monitor transpose is free.** Section 5 says
$\partial F/\partial \mathbf E = M^{T}\,\partial F/\partial y$. In FDTDX, $M$ is
`PhasorDetector.update`, which is linear in $(E, H)$. So $M^{T}$ is exactly
`jax.vjp` of that method, and we get the co-location stencil, the region
restriction, the `static_scale` factor and the H time-average split without
re-deriving any of them. Do not hand-write $M^{T}$.

**The $1/(i\omega)$ factor is continuum and must be replaced.** In
$\mathbf J_{\mathrm{adj}} = \frac{1}{i\omega}\partial F/\partial \mathbf E$ the
$i\omega$ comes from $A\mathbf E = i\omega\mathbf J$. FDTDX's leapfrog update
produces the discrete analogue

$$
(i\omega)_{\text{disc}} = \frac{1 - e^{-i\omega \Delta t}}{\Delta t},
$$

which agrees with $i\omega$ only as $\omega\Delta t \to 0$. Using the continuum
factor degrades agreement at coarse resolution and near Nyquist. This is the
same factor Meep carries as `iomega` in `meep/adjoint/objective.py`.

**No conjugate in the design-region overlap.** Section 4's gradient is
$\mathrm{Re}[\lambda^{T}(\cdots)\mathbf E]$, not
$\mathrm{Re}[\lambda^{\dagger}(\cdots)\mathbf E]$. Both FDTDX and Meep
accumulate phasors with an $e^{+i\omega t}$ kernel, so they share the
$e^{-i\omega t}$ convention and no conjugation flip is needed between the two
codes. Verify this against the accumulation expression rather than trusting it.

**H in the FoM is section 7, and it is cheap.** Our scope decision to support E
and H costs one extra term, $B^{T}\partial F/\partial \mathbf H$, because
$\mu$ is design-independent in every scene we care about. Confirm the sign of
that term by finite difference on `inv_permeabilities` before enabling it.
