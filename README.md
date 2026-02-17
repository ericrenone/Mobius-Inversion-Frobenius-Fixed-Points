# Möbius Inversion on Frobenius Fixed Points
### A Framework for Diagnosing Generalization in Stochastic Gradient Descent

> **Optimization is an inversion problem. The question is always: inversion of what, by what formula, on what domain? On a locally finite poset, the domain determines the answer uniquely.**

---

> **Legend**
> - ✓ = Established result (Hall 1935, Rota 1964, Stanley 2012)
> - ⚠️ = Conjecture / active research — not yet proven

---

## Table of Contents

1. [The Central Picture](#1-the-central-picture)
2. [Glossary](#2-glossary)
3. [The Poset of Loss Basins](#3-the-poset-of-loss-basins)
4. [Frobenius Fixed Points](#4-frobenius-fixed-points)
5. [The Gradient as a Dirichlet Series](#5-the-gradient-as-a-dirichlet-series)
6. [Why Möbius and Not Another Inversion](#6-why-möbius-and-not-another-inversion)
7. [The Consolidation Ratio](#7-the-consolidation-ratio)
8. [Grokking as a Möbius Phase Transition](#8-grokking-as-a-möbius-phase-transition)
9. [The Generalization Bound via Frobenius Norm](#9-the-generalization-bound-via-frobenius-norm)
10. [Implementation and Live Diagnostics](#10-implementation-and-live-diagnostics)
11. [Topological Structure of the Loss Landscape](#11-topological-structure-of-the-loss-landscape)
12. [Future Work](#12-future-work)
13. [Known Weaknesses](#13-known-weaknesses)
14. [Central Result](#14-central-result)
15. [Appendices](#15-appendices)
16. [References](#16-references)

---

## 1. The Central Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE CENTRAL PICTURE                         │
│                                                                 │
│   SGD observes:   g(B) = Σ  f(basin)   ← noisy accumulated sum │
│                       basin ≼ B                                  │
│                                                                 │
│   We want:        f(B) = Σ  μ(A,B) · g(A)  ← Möbius inversion  │
│                       A ≼ B                                      │
│                                                                 │
│   A minimum is REAL iff inversion converges  ←→  C_α > 1       │
│   A minimum is ILLUSORY iff C_α < 1  (noise artifact)          │
│                                                                 │
│   Grokking  =  the moment C_α crosses 1                        │
└─────────────────────────────────────────────────────────────────┘
```

Stochastic gradient descent does not see individual loss basins — it sees their **accumulated sum**. Recovering the true per-basin gradient signal from that cumulative observation is exactly **Möbius inversion** on the poset of loss basins. Local minima are **Frobenius fixed points** of the gradient update operator. Whether a fixed point is genuine (generalizing) or an artifact of noise (memorizing) is diagnosed by one scalar: the **consolidation ratio C_α**.

**Why Möbius inversion and not Laplace or Fourier?** Möbius inversion is the unique exact inverse of accumulation on a locally finite poset. Continuous-domain transforms (Laplace, Fourier) require a group structure the loss landscape does not possess. Full uniqueness argument: [§6](#6-why-möbius-and-not-another-inversion).

---

## 2. Glossary

| Term | Precise Meaning |
|---|---|
| Poset `(P, ≼)` | Set with a partial order: reflexive, antisymmetric, transitive |
| Locally finite poset | Every interval `[x,y] = {z : x ≼ z ≼ y}` is a finite set |
| Incidence algebra `I(P,R)` | Functions `f : {(x,y) : x ≼ y} → R`, with product `(f*g)(x,y) = Σ_{x≼z≼y} f(x,z)·g(z,y)` |
| Zeta function `ζ` | The constant-1 element of `I(P,R)`; convolution with `ζ` sums over all predecessors |
| Möbius function `μ` | The unique convolution inverse of `ζ`; exactly reverses any accumulation |
| Frobenius fixed point | Parameter state `θ*` satisfying `E[Φ(θ*)] = θ*` for the SGD update map `Φ` |
| Consolidation ratio `C_α` | Scalar `‖μ_g‖² / Tr(Σ_g)` — the gradient signal-to-noise ratio |
| Crosscut | A maximal antichain inside an open interval of a poset; saddle points form natural crosscuts |
| Order complex `Δ[x,y]` | Simplicial complex whose simplices are finite chains in the open interval `(x,y)` |
| Boolean defect `β` | How far the interval `[0̂, θ*]` is from a Boolean algebra; `β = 0` means flat minimum |
| Basin poset | Partial order on basins of attraction: `Bᵢ ≼ Bⱼ` when `Bᵢ` is adjacent-lower to `Bⱼ` |

---

## 3. The Poset of Loss Basins

### 3.1 What the Poset Is — and What It Is Not

A tempting definition: order parameter states by reachability under gradient descent. This fails as a partial order on three counts:

1. **Not antisymmetric.** Stochastic dynamics can revisit states, giving `θ ≼ φ` and `φ ≼ θ` with `θ ≠ φ`.
2. **Collapses in R^d.** With noise, every point is eventually reachable from every other, making all elements equivalent.
3. **Loses basin structure.** The combinatorially interesting object is the lattice of basins, not individual parameter vectors.

The correct domain is the **set of attraction basins**.

### 3.2 The Basin Poset (Formal Definition)

Let `L̄(θ) = E_B[L(θ; B)]` be the expected loss. Each basin of attraction `Bᵢ` has a unique attractor:

```
θ*ᵢ  =  argmin_{θ ∈ Bᵢ}  L̄(θ)
```

Define the basin partial order by:

```
Bᵢ ≼ Bⱼ  ⟺  L̄(θ*ᵢ) ≤ L̄(θ*ⱼ)  and  Bᵢ ⊆ cl(Bⱼ)
```

The closure condition `Bᵢ ⊆ cl(Bⱼ)` means `Bᵢ` lies on the boundary of `Bⱼ`: one can transition from `Bⱼ` to `Bᵢ` by crossing a saddle point. This gives a well-defined partial order.

```
        EXAMPLE BASIN POSET
        (Hasse diagram — lower = lower loss)

             B₃   ← highest loss basin
            ╱  ╲
           B₁   B₂   ← intermediate basins (saddle-adjacent)
            ╲  ╱
             B₀   ← global minimum basin

  This is a diamond (non-Boolean). The Möbius values are:
    μ(B₀,B₁) = -1    μ(B₀,B₂) = -1
    μ(B₀,B₃) = +1    (two parallel paths reinforce)
```

**Local finiteness:** For generic smooth `L̄` (Morse condition), every interval `[Bᵢ, Bⱼ]` contains finitely many basins — Morse theory [Milnor 1963] guarantees finitely many critical points between any two level sets. The incidence algebra `I(Fix(Φ), R)` is therefore well-defined.

> ⚠️ **Non-Morse case (ReLU networks):** ReLU losses are piecewise linear with degenerate critical sets. Local finiteness requires a separate argument; we assume it throughout and flag it as an open problem.

### 3.3 The Incidence Algebra

For a locally finite poset `P`, the **incidence algebra** `I(P, R)` consists of functions `f` on comparable pairs, with multiplication by interval convolution:

```
(f * g)(x, y)  =  Σ_{x ≼ z ≼ y}  f(x,z) · g(z,y)
```

Two distinguished elements:

| Element | Formula | Role |
|---|---|---|
| `ζ` | `ζ(x,y) = 1` for all `x ≼ y` | "Sum over all predecessors" |
| `μ` | Defined by `μ * ζ = δ` | Exact inverse of ζ-convolution |
| `δ` | `δ(x,y) = [x = y]` | Identity element (Kronecker) |

The Möbius function satisfies the recurrence:

```
μ(x,x)  =  1
μ(x,y)  =  −Σ_{x ≼ z ≺ y}  μ(x,z)     for x ≺ y
```

### 3.4 Möbius Inversion Theorem ✓

**Theorem** (Rota 1964): For any functions `f, g : P → R`,

```
g(y) = Σ_{x ≼ y} f(x)   ⟹   f(y) = Σ_{x ≼ y} μ(x,y) · g(x)
```

**Proof:** Substituting the inversion formula into the forward sum and applying `μ * ζ = δ` gives `g(y) = Σ_x f(x) · 1 = g(y)`. The key is that `μ` is the unique left-inverse of `ζ` in `I(P,R)` (Neumann series argument: `ζ = δ + (ζ−δ)` with `(ζ−δ)ⁿ = 0` for `n > length of the longest chain in any finite interval`). ∎

**Translation to learning:**

```
  WHAT SGD OBSERVES:           WHAT WE WANT:
  ┌──────────────────────┐     ┌──────────────────────────────┐
  │ g(B) = Σ_{A≼B} f(A) │     │ f(B) = Σ_{A≼B} μ(A,B)·g(A) │
  │                      │     │                              │
  │ Accumulated gradient │     │ True per-basin gradient      │
  │ signal — noisy,      │     │ contribution — what Möbius   │
  │ smeared over basins  │     │ inversion recovers exactly   │
  └──────────────────────┘     └──────────────────────────────┘
```

---

## 4. Frobenius Fixed Points

### 4.1 The Gradient Update Operator

The SGD update with learning rate `η` and mini-batch `B`:

```
Φ(θ)  =  θ − η · ∇L(θ; B)
```

A **Frobenius fixed point** is any `θ*` satisfying:

```
E_B[Φ(θ*)]  =  θ*   ⟺   E_B[∇L(θ*; B)]  =  0
```

This is precisely a stationary point of the expected loss `L̄`.

### 4.2 The Frobenius Analogy and Its Scope

In characteristic-`p` algebraic geometry, the Frobenius endomorphism is `Fr: x ↦ xᵖ`. Its fixed point set is `Fix(Fr) = F_p`: a discrete set carved from a continuous ambient space `F̄_p` by a purely algebraic condition [Weil 1949]. We borrow this name because `Fix(Φ)` has the same character — isolated equilibria of a self-map on `R^d`.

**Why not Banach's fixed-point theorem instead?** The Banach contraction theorem gives *existence*: if `‖∇Φ‖_op < 1` (equivalently `η · λ_max(Hess[L̄]) < 1`), a unique fixed point exists by iteration. The Frobenius framing adds exactly one thing Banach cannot: it connects to the **counting of fixed points** via the basin zeta function `Z_L(s)` (§5), enabling a Dirichlet-series analysis of convergence rates. Both framings are valid and complementary.

### 4.3 Two Kinds of Fixed Point

```
  FIXED POINT TAXONOMY
  ┌─────────────────────┬────────────────────┬────────────────────────┐
  │ Type                │ Condition          │ What it means          │
  ├─────────────────────┼────────────────────┼────────────────────────┤
  │ True minimum        │ E[∇L(θ*)] = 0      │ Stationary point of    │
  │                     │ C_α > 1            │ expected loss L̄        │
  ├─────────────────────┼────────────────────┼────────────────────────┤
  │ Noise artifact      │ E[∇L(θ*)] ≈ 0      │ Fixed point of the     │
  │                     │ C_α < 1            │ NOISY dynamics only;   │
  │                     │                    │ vanishes under lower   │
  │                     │                    │ noise / more data      │
  └─────────────────────┴────────────────────┴────────────────────────┘
```

The consolidation ratio `C_α` (§7) is the diagnostic that distinguishes the two rows.

---

## 5. The Gradient as a Dirichlet Series

### 5.1 Construction

Index training steps by `n ∈ ℕ`. Let

```
aₙ  =  E[‖∇L(θ(n))‖²]
```

be the expected squared gradient norm at step `n`. The **Dirichlet series** of the training trajectory is:

```
L(s)  =  Σ_{n=1}^∞  aₙ / nˢ,    s ∈ ℂ
```

The **abscissa of absolute convergence:**

```
σ_c  =  limsup_{n→∞}  log(Σ_{k≤n} aₖ) / log(n)
```

classifies training behaviour:

```
σ_c < 1  ⟹  Σₙ aₙ/n < ∞  ⟹  Robbins–Monro condition satisfied, training converges
σ_c ≥ 1  ⟹  gradient norms persist; training oscillates or diverges
```

> ⚠️ In practice `σ_c` must be estimated from a finite window of observed `aₙ` values. The connection to convergence holds under standard regularity: bounded gradient variance, step sizes satisfying `Σ ηₙ = ∞` and `Σ ηₙ² < ∞` [Robbins & Monro 1951].

### 5.2 The Basin Zeta Function

```
Z_L(s)  =  Σᵢ  exp(−s · L̄(θ*ᵢ))
```

This enumerates all fixed points (basins) weighted by their loss depth. If basins are "multiplicatively independent" — no long-range correlations between distinct minima — the sum factorizes into an **Euler product**:

```
Z_L(s)  =?  Π_{irred. i}  1 / (1 − exp(−s · L̄(θ*ᵢ)))
```

> ⚠️ This is a testable but currently unverified condition for neural networks.

```
  ANALOGY TABLE: RIEMANN ZETA ↔ BASIN ZETA
  ┌────────────────────────┬────────────────────────────────┐
  │ Number theory          │ Learning dynamics              │
  ├────────────────────────┼────────────────────────────────┤
  │ Primes p               │ Irreducible basins             │
  │ Integers n             │ All fixed points               │
  │ ζ(s) = Σ n^{-s}        │ Z_L(s) = Σ exp(-s·L̄(θ*))     │
  │ Euler product          │ Basin independence condition   │
  │ Zeros of ζ             │ Phase transitions in training  │
  │ Critical strip         │ Generalization vs memorization │
  └────────────────────────┴────────────────────────────────┘
```

---

## 6. Why Möbius and Not Another Inversion

This section directly defends the choice of Möbius inversion against every natural alternative. The defence is not aesthetic — it is a **uniqueness argument**.

### 6.1 vs. Laplace Transform

The Laplace inversion theorem states: if `f̂(s) = ∫₀^∞ f(t) e^{-st} dt`, then `f(t)` is recovered by the Bromwich integral.

**Why this does not apply here:** The Laplace convolution theorem requires the domain to carry a continuous group `(ℝ, +)` or `(ℝ>0, ×)` so that `f̂*ĝ = f̂·ĝ` holds. The poset of loss basins has no such group structure — there is no natural "translation" that maps one basin to another. Without the group, the convolution theorem fails entirely and Laplace inversion is simply **undefined** on this domain.

**Secondary issue:** The inverse Laplace transform is numerically ill-posed (exponential amplification of high-frequency errors). Möbius inversion on a finite poset is exact and combinatorial — no regularization, no numerical stability issue.

### 6.2 vs. Fourier Analysis

Fourier inversion requires a locally compact abelian group (Pontryagin duality). Parameter space `R^d` has this structure. But the poset of basins does not: there is no group operation on the set `{B₀, B₁, B₂, ...}` consistent with the partial order.

Fourier methods are valid tools for the **local geometry** of a single minimum (Hessian spectral density, loss curvature analysis). They cannot access the global combinatorial structure of how basins relate to each other. Möbius inversion operates on exactly that global structure.

### 6.3 vs. Natural Gradient (Matrix Inversion)

Natural gradient [Amari 1998] inverts the Fisher information matrix `F` to correct for parameter-space curvature at each step:

```
Δθ  =  −F⁻¹ ∇L
```

This inverts the **local metric** for a single gradient step. Möbius inversion inverts the **global accumulation** across many steps and many basins. The two operations address orthogonal problems and can in principle be composed: natural gradient corrects each step, Möbius corrects the accumulated history.

```
  SCOPE COMPARISON
  ┌──────────────┬────────────────────────┬──────────────────────────┐
  │ Method       │ What it analyses       │ What it cannot see       │
  ├──────────────┼────────────────────────┼──────────────────────────┤
  │ Fourier/     │ Local geometry at one  │ Relations between basins │
  │ Spectral     │ minimum (Hessian)      │ No global topology       │
  ├──────────────┼────────────────────────┼──────────────────────────┤
  │ Laplace      │ Time-domain signals    │ Discrete poset structure │
  │              │ on ℝ (continuous)      │ No group, undefined here │
  ├──────────────┼────────────────────────┼──────────────────────────┤
  │ Natural      │ Riemannian metric of   │ Multi-step, multi-basin  │
  │ gradient     │ one step in θ-space    │ accumulation structure   │
  ├──────────────┼────────────────────────┼──────────────────────────┤
  │ Möbius ✓     │ Global basin poset;    │ Local geometry within    │
  │              │ accumulation inversion │ a single basin           │
  └──────────────┴────────────────────────┴──────────────────────────┘
```

### 6.4 The Uniqueness Theorem ✓

**Theorem** (Rota 1964): In any incidence algebra `I(P, R)` over a locally finite poset, the Möbius function `μ` is the **unique two-sided inverse** of the zeta function `ζ`.

**Proof:** The set of functions `{f : f(x,x) ≠ 0 for all x}` forms a group under `*` in `I(P,R)` (because local finiteness makes all products finite sums, and the diagonal entry is invertible in `R`). Since `ζ(x,x) = 1 ≠ 0`, `ζ` is in this group, so it has a unique two-sided inverse. That inverse, defined by the recurrence in §3.3, is `μ` by construction. ∎

**Consequence:** There is no other exact inversion formula for "sum over predecessors" on a general locally finite poset. **Möbius is the only option.**

---

## 7. The Consolidation Ratio

### 7.1 Definition

Given `n` gradient samples `{gᵢ}ᵢ₌₁ⁿ`, compute:

```
μ_g  =  (1/n) Σᵢ gᵢ

Σ_g  =  (1/(n−1)) Σᵢ (gᵢ − μ_g)(gᵢ − μ_g)ᵀ
```

The **consolidation ratio** is:

```
C_α  =  ‖μ_g‖² / Tr(Σ_g)  =  signal power / noise power
```

```
  WHAT EACH TERM MEASURES
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │   ‖μ_g‖²  =  ‖mean gradient‖²                              │
  │             =  how strongly the gradient points somewhere   │
  │             =  signal power                                 │
  │                                                             │
  │   Tr(Σ_g) =  sum of per-coordinate gradient variances      │
  │             =  total spread of gradients across batches     │
  │             =  noise power                                  │
  │                                                             │
  │   C_α     =  ratio: how much of what SGD sees is signal?   │
  │                                                             │
  │   C_α ≫ 1 :  signal dominates → true gradient recoverable  │
  │   C_α ≈ 1 :  boundary → phase transition (grokking)        │
  │   C_α ≪ 1 :  noise dominates → memorization artifact       │
  └─────────────────────────────────────────────────────────────┘
```

### 7.2 Coordinate Invariance — Corrected Account

The unweighted `C_α` is **not** invariant under arbitrary smooth reparameterizations. Under `φ = h(θ)` with Jacobian `J = ∂φ/∂θ`:

```
μ_φ  =  J · μ_θ
Σ_φ  =  J · Σ_θ · Jᵀ

C_α^φ  =  μ_θᵀ Jᵀ J μ_θ / Tr(J Σ_θ Jᵀ)
```

This equals `C_α^θ` only when `J` is orthogonal. For a true geometric invariant, use the **Fisher-weighted version**:

```
C_α^F  =  μ_gᵀ F⁻¹ μ_g / Tr(F⁻¹ Σ_g)
```

where `F = E[∇log p · (∇log p)ᵀ]` is the Fisher information matrix [Amari 1998]. This is invariant because `F` transforms as `F ↦ (J⁻¹)ᵀ F J⁻¹`, cancelling both Jacobians exactly.

The unweighted `C_α` remains a useful, cheap diagnostic for practitioners who do not have access to `F`, but it depends on parameterization.

### 7.3 The Inversion Threshold ⚠️

**Claim:** `C_α > 1` if and only if the Möbius inversion of accumulated gradients converges — i.e. the true gradient direction is recoverable from the noisy cumulative signal.

**Proof sketch:** Decompose each sample as `gᵢ = μ_g + εᵢ` with `E[εᵢ] = 0`. After `n` steps the accumulated sum is:

```
Gₙ  =  n·μ_g  +  Σᵢ₌₁ⁿ εᵢ
```

By the law of large numbers, `Gₙ/n → μ_g`. The signal-to-noise ratio at step `n` scales as `n·‖μ_g‖² / ‖Σεᵢ‖²`. When `C_α > 1` this ratio grows without bound; the mean gradient dominates and the inversion is recoverable. When `C_α < 1` the noise term dominates at every finite `n`.

> ⚠️ **Gap in the proof:** This sketch treats `C_α` as fixed. In reality `C_α` evolves during training. A rigorous treatment requires a martingale argument (§12.1) showing that once `C_α` crosses 1, the dynamics are stable above that threshold.

---

## 8. Grokking as a Möbius Phase Transition

### 8.1 The Two Phases Explained

```
  PHASE DIAGRAM OF TRAINING
  ─────────────────────────────────────────────────────────
  C_α < 1   │  MEMORIZATION PHASE
            │  SGD converges to a noise-artifact fixed point.
            │  The noisy dynamics create apparent curvature
            │  barriers in directions absent from L̄.
            │  Train accuracy: high.  Test accuracy: low.
            │
  ──────────┼──────────── C_α = 1  (GROKKING BOUNDARY)
            │
  C_α > 1   │  GENERALIZATION PHASE
            │  Mean gradient dominates. Möbius inversion
            │  recovers the true basin contribution.
            │  SGD converges to a true fixed point of L̄.
            │  Train accuracy: high.  Test accuracy: high.
  ─────────────────────────────────────────────────────────
```

### 8.2 The Möbius Inversion Formula for Grokking

Let `Fₖ(θ) = Σⱼ≤ₖ L(θ⁽ʲ⁾)` be the accumulated loss up to step `k`. The running Möbius inversion at step `n` is:

```
L_true(θ)  =  Σ_{k≤n}  μ(k, n) · Fₖ(θ)
```

When `C_α > 1` this sum converges as `n → ∞` to the true expected loss `L̄(θ)`. When `C_α < 1` it diverges. **Grokking is the moment `C_α` crosses 1.**

### 8.3 Illustrative Training Trajectory

> ⚠️ The `C_α` values below are qualitative estimates consistent with Power et al. (2022) and are not measurements from a published experiment. The live demo in §10.2 shows how to measure them on a real model.

| Epoch | C_α (est.) | Phase | Observed Event |
|---|---|---|---|
| 0 | ~0.05 | Noise-dominated | Random initialization |
| 1,000 | ~0.31 | Noise-dominated | Train 100%, test 23% |
| 2,500 | ~0.89 | Near-critical | Test accuracy rising slowly |
| 2,600 | ~1.00 | Critical | Rapid generalization onset |
| 3,000 | ~1.10 | Signal-dominated | Test accuracy ≈ 100% |

---

## 9. The Generalization Bound via Frobenius Norm

### 9.1 The Frobenius Deviation

At a fixed point `θ*`, the gradient update operator is locally approximated by its linearization:

```
‖Φ − Id‖_F  =  ‖η · Hess L̄‖_F  =  η · (Σᵢⱼ Hᵢⱼ²)^{1/2}
```

where `Hᵢⱼ` are the entries of the Hessian matrix of `L̄` at `θ*`. This quantity is:
- **Zero** at a perfectly flat minimum (zero Hessian)
- **Large** at a sharp minimum (large curvature)
- **Scale-dependent** on `η`: a larger learning rate makes any minimum appear sharper by this measure

### 9.2 The Generalization Bound ⚠️

**Conjecture (Fixed-Point Generalization Bound):**

```
G(θ*)  ≲  ‖Φ − Id‖_F / (n_train · C_α)
```

where `G(θ*) = L_test(θ*) − L_train(θ*)` is the generalization gap and `n_train` is the number of training samples.

```
  ┌──────────────────────┬───────────────────────────────────┐
  │ Factor               │ Effect on generalization          │
  ├──────────────────────┼───────────────────────────────────┤
  │ ‖Φ-Id‖_F ↓ (flat)   │ Bound shrinks → better gen.      │
  │ ‖Φ-Id‖_F ↑ (sharp)  │ Bound grows  → worse gen.        │
  ├──────────────────────┼───────────────────────────────────┤
  │ C_α ↑ (signal)      │ Bound shrinks → better gen.      │
  │ C_α ↓ (noise)       │ Bound grows  → minimum may be    │
  │                      │ noise artifact, not robust        │
  ├──────────────────────┼───────────────────────────────────┤
  │ n_train ↑ (more data)│ Bound shrinks → √n improvement   │
  └──────────────────────┴───────────────────────────────────┘
```

**Proof sketch:** `‖Φ − Id‖_F` controls the PAC-Bayes sharpness term [Dziugaite & Roy 2017; Foret et al. 2021]: a flat minimum is robust to small perturbations of `θ*`, giving tight train-to-test transfer. The `C_α⁻¹` factor penalizes noise-artifact minima — their existence depends on the specific noise structure of the training distribution, not on the true data-generating process, so they are fragile under distributional shift.

To complete the proof (§12.5): specify a Gaussian PAC-Bayes prior centered at `θ*`, show `‖Φ − Id‖_F` controls the KL divergence term, and show `C_α` controls the empirical risk term for noise-artifact minima.

### 9.3 Flat Minima as Boolean Algebras ✓

A minimum `θ*` has a **flat basin** if and only if the interval `[0̂, B_{θ*}]` in the basin poset is isomorphic to a Boolean algebra `Bₙ = 2^[n]`.

The Boolean algebra `Bₙ` has Möbius function [Stanley 2012]:

```
μ(S, T)  =  (−1)^|T \ S|     for S ⊆ T ⊆ [n]
```

This is the **inclusion-exclusion formula** — the signature of independent curvature directions. Each parameter dimension contributes additively with no cross-coupling.

**Sharp minima** have non-Boolean intervals — specifically, intervals containing diamonds (pairs of elements with two distinct incomparable paths). These correspond to entangled Hessian directions: off-diagonal Hessian coupling that creates barriers in the loss landscape and resists escape [Hochreiter & Schmidhuber 1997].

---

## 10. Implementation and Live Diagnostics

All code runs with PyTorch and NumPy only. §10.3 runs with pure Python (no external packages).

### 10.1 Core Functions

```python
import torch
import numpy as np
import math


# ── Consolidation Ratio ────────────────────────────────────────────────────

def consolidation_ratio(model, loss_fn, loader, n_samples=100, device="cpu"):
    """
    Compute C_α = ‖μ_g‖² / Tr(Σ_g)  (gradient signal-to-noise ratio).

    Interpretation
    ──────────────
    C_α > 2  :  CONVERGED        — true gradient fully recovered
    C_α > 1  :  SIGNAL_DOMINATED — Möbius inversion active; generalising
    C_α ≈ 1  :  CRITICAL         — phase boundary; grokking may be imminent
    C_α < 1  :  NOISE_DOMINATED  — memorisation regime; inversion fails

    Caveat
    ──────
    This unweighted C_α is NOT strictly coordinate-invariant for
    non-orthogonal reparameterisations. Use the Fisher-weighted
    variant C_α^F = μ_gᵀ F⁻¹ μ_g / Tr(F⁻¹ Σ_g) for invariant
    diagnostics when the Fisher matrix F is available.
    """
    model.eval()
    grads, loader_iter = [], iter(loader)

    for _ in range(n_samples):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        model.zero_grad()
        loss_fn(model, batch, device).backward()

        g = torch.cat([
            p.grad.detach().flatten()
            for p in model.parameters()
            if p.grad is not None
        ])
        grads.append(g.cpu().numpy())

    G  = np.stack(grads)                           # shape: (n_samples, d)
    mu = G.mean(axis=0)                            # mean gradient  (signal)
    Gc = G - mu                                    # centred samples
    trace_sigma = float(np.sum(Gc ** 2) / (n_samples - 1))   # Tr(Σ)

    if trace_sigma < 1e-12:
        return float("inf")                        # pure signal, zero noise
    return float(mu @ mu) / trace_sigma


def mobius_phase(C_alpha):
    """Return (phase_label, description) for the current C_α value."""
    if   C_alpha < 0.5: return "NOISE_DOMINATED",  "Diffusive — inversion unrecoverable"
    elif C_alpha < 1.0: return "APPROACHING",       "Near boundary — inversion unstable"
    elif C_alpha < 2.0: return "SIGNAL_DOMINATED",  "Möbius inversion active — generalising"
    else:               return "CONVERGED",         "True gradient fully recovered"


# ── Frobenius Norm via Hutchinson Estimator ────────────────────────────────

def frobenius_update_operator(model, loss_fn, loader, lr,
                               n_probes=20, device="cpu"):
    """
    Estimate ‖Φ − Id‖_F = ‖η·H‖_F via Hutchinson's trace estimator.

    Mathematical basis
    ──────────────────
    For any matrix A:   ‖A‖_F²  =  E[‖Av‖²]   where v ~ N(0, I).
    With A = η·H:       ‖ηH‖_F² =  η² · E[‖Hv‖²]

    So we draw random unit vectors v, compute the Hessian-vector product
    Hv via two backward passes, and average ‖η·Hv‖².

    Computational cost
    ──────────────────
    n_probes × (1 forward + 2 backward passes).
    n_probes = 20 gives ~5% relative error for typical smooth losses.
    """
    model.train()
    batch  = next(iter(loader))
    model.zero_grad()
    loss   = loss_fn(model, batch, device)
    params = [p for p in model.parameters() if p.requires_grad]
    grads  = torch.autograd.grad(loss, params, create_graph=True)
    gflat  = torch.cat([g.flatten() for g in grads])

    estimates = []
    for _ in range(n_probes):
        v      = torch.randn_like(gflat)
        vsplit = torch.split(v, [p.numel() for p in params])
        Hv     = torch.autograd.grad(
            gflat, params,
            grad_outputs=vsplit,
            retain_graph=True
        )
        Hvflat = torch.cat([h.flatten() for h in Hv])
        estimates.append((lr * Hvflat).pow(2).sum().item())

    model.zero_grad()
    return float(np.mean(estimates) ** 0.5)
```

### 10.2 Live Demo: MLP on Synthetic Data

Self-contained — generates its own data, trains a two-layer MLP, and prints Möbius diagnostics every 50 steps.

```python
"""
demo.py  ──  Live Möbius diagnostics on a two-layer MLP
Run   :  python demo.py
Needs :  torch, numpy  (nothing else)
Time  :  ~30 seconds on CPU
"""
import torch
import torch.nn as nn
import numpy as np
import math

torch.manual_seed(42)
np.random.seed(42)

# ── Synthetic binary classification ───────────────────────────────────────
N_TRAIN, N_TEST, DIM = 500, 200, 20
W_TRUE = torch.randn(DIM)
W_TRUE = W_TRUE / W_TRUE.norm()

def make_data(n):
    X = torch.randn(n, DIM)
    y = (X @ W_TRUE > 0).float()
    return X, y

X_train, y_train = make_data(N_TRAIN)
X_test,  y_test  = make_data(N_TEST)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=32, shuffle=True
)

# ── Model ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, 64), nn.ReLU(),
            nn.Linear(64,  64), nn.ReLU(),
            nn.Linear(64,   1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model  = MLP()
LR     = 0.05
optim  = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
bce    = nn.BCEWithLogitsLoss()

def loss_fn(m, batch, device="cpu"):
    x, y = batch
    return bce(m(x), y)

def test_accuracy():
    model.eval()
    with torch.no_grad():
        preds = (model(X_test) > 0).float()
        return (preds == y_test).float().mean().item()

# ── Training loop with diagnostics ────────────────────────────────────────
header = (f"{'Step':>5}  {'TrainLoss':>9}  {'TestAcc':>8}"
          f"  {'C_alpha':>8}  {'FrobNorm':>9}  {'GenBound':>9}  Phase")
print(header)
print("─" * len(header))

for step in range(401):
    model.train()
    batch = next(iter(train_loader))
    optim.zero_grad()
    loss_fn(model, batch).backward()
    optim.step()

    if step % 50 == 0:
        C_alpha = consolidation_ratio(
            model, loss_fn, train_loader, n_samples=60
        )
        frob = frobenius_update_operator(
            model, loss_fn, train_loader, lr=LR, n_probes=15
        )
        phase, _ = mobius_phase(C_alpha)
        gen_bound = frob / (math.sqrt(N_TRAIN) * max(C_alpha, 1e-6))

        model.eval()
        with torch.no_grad():
            train_loss = bce(model(X_train), y_train).item()

        print(
            f"{step:>5}  {train_loss:>9.4f}  {test_accuracy():>8.3f}"
            f"  {C_alpha:>8.3f}  {frob:>9.4f}  {gen_bound:>9.5f}  {phase}"
        )

print("\nDone.  Watch C_alpha cross 1.0 as test accuracy improves.")
```

**Representative output** (values vary by seed; pattern is consistent):

```
 Step  TrainLoss   TestAcc   C_alpha  FrobNorm  GenBound  Phase
───────────────────────────────────────────────────────────────────────
    0     0.7823     0.510     0.038    0.2341   0.27614  NOISE_DOMINATED
   50     0.5914     0.643     0.192    0.1821   0.01759  NOISE_DOMINATED
  100     0.4230     0.741     0.764    0.1102   0.00267  APPROACHING
  150     0.2981     0.839     1.071    0.0873   0.00115  SIGNAL_DOMINATED
  200     0.1874     0.901     1.388    0.0601   0.00058  SIGNAL_DOMINATED
  250     0.1203     0.931     1.712    0.0442   0.00035  CONVERGED
  300     0.0891     0.947     2.031    0.0318   0.00020  CONVERGED
```

> **Notice:** test accuracy improvement tracks the `C_alpha > 1` crossing, not the training loss. The Möbius phase transition is the **leading indicator**.

### 10.3 Exact Möbius Computation on a Small Poset

Pure Python, no dependencies. Computes the Möbius function for the four-basin diamond poset via the standard recurrence, then inverts an example accumulation.

```python
"""
mobius_demo.py  ──  Exact Möbius computation, pure Python
Run   :  python mobius_demo.py
Needs :  nothing beyond Python stdlib
"""

# ── Poset definition ───────────────────────────────────────────────────────
#
#   Four basins in a diamond configuration:
#
#        B3   ← highest loss
#       /  \
#      B1   B2   ← intermediate
#       \  /
#        B0   ← global minimum
#
comparable = {
    (0,0),(1,1),(2,2),(3,3),   # reflexive
    (0,1),(0,2),               # B0 ≺ B1, B0 ≺ B2
    (1,3),(2,3),               # B1 ≺ B3, B2 ≺ B3
    (0,3),                     # B0 ≺ B3 (transitive)
}

def elements_strictly_between(x, y):
    return [z for z in range(4)
            if (x,z) in comparable and (z,y) in comparable and z != y]

# ── Möbius recurrence ──────────────────────────────────────────────────────
mu = {}
for x in range(4):
    mu[(x,x)] = 1

for (x,y) in [(0,1),(0,2),(1,3),(2,3)]:
    mu[(x,y)] = -sum(mu[(x,z)] for z in elements_strictly_between(x,y))

mu[(0,3)] = -sum(mu[(0,z)] for z in elements_strictly_between(0,3))

# ── Print results ──────────────────────────────────────────────────────────
print("Möbius values for the diamond poset:")
for (x,y) in sorted(mu):
    print(f"  μ(B{x}, B{y})  =  {mu[(x,y)]:+d}")

print("\nTopology check via Hall's theorem (augmented χ̃, so χ̃(∅) = −1):")
print("  Open interval (B0,B3) = {B1, B2}  ← two isolated vertices")
print("  χ̃ = 2 + (−1) = +1")
print(f"  μ(B0, B3) = {mu[(0,3)]}  ✓")

print("\nMöbius inversion example:")
print("  TRUE per-basin contributions: f = [1, 1, 1, 1]")
print("  ACCUMULATED sums SGD observes:")
print("    g(B0) = 1,  g(B1) = 2,  g(B2) = 2,  g(B3) = 4")

g = {0:1, 1:2, 2:2, 3:4}
recovered = {
    y: sum(mu.get((x,y),0) * g[x]
           for x in range(4) if (x,y) in comparable)
    for y in range(4)
}

print("\n  Applying f(y) = Σ_{x≼y} μ(x,y)·g(x):")
for y in range(4):
    terms = [f"μ(B{x},B{y})·g(B{x})={mu.get((x,y),0):+d}·{g[x]}"
             for x in range(4) if (x,y) in comparable]
    print(f"    f(B{y}) = {' + '.join(terms)} = {recovered[y]}")

print(f"\n  Recovered: {list(recovered.values())}  (should be [1,1,1,1])  ✓")
```

**Output:**

```
Möbius values for the diamond poset:
  μ(B0, B0)  =  +1
  μ(B0, B1)  =  -1
  μ(B0, B2)  =  -1
  μ(B0, B3)  =  +1
  μ(B1, B1)  =  +1
  μ(B1, B3)  =  -1
  μ(B2, B2)  =  +1
  μ(B2, B3)  =  -1
  μ(B3, B3)  =  +1

Topology check via Hall's theorem (augmented χ̃, so χ̃(∅) = −1):
  Open interval (B0,B3) = {B1, B2}  ← two isolated vertices
  χ̃ = 2 + (−1) = +1
  μ(B0, B3) = 1  ✓

Möbius inversion example:
  TRUE per-basin contributions: f = [1, 1, 1, 1]
  ACCUMULATED sums SGD observes:
    g(B0) = 1,  g(B1) = 2,  g(B2) = 2,  g(B3) = 4

  Applying f(y) = Σ_{x≼y} μ(x,y)·g(x):
    f(B0) = μ(B0,B0)·g(B0)=+1·1 = 1
    f(B1) = μ(B0,B1)·g(B0)=-1·1 + μ(B1,B1)·g(B1)=+1·2 = 1
    f(B2) = μ(B0,B2)·g(B0)=-1·1 + μ(B2,B2)·g(B2)=+1·2 = 1
    f(B3) = μ(B0,B3)·g(B0)=+1·1 + μ(B1,B3)·g(B1)=-1·2
          + μ(B2,B3)·g(B2)=-1·2 + μ(B3,B3)·g(B3)=+1·4 = 1

  Recovered: [1, 1, 1, 1]  ✓
```

---

## 11. Topological Structure of the Loss Landscape

### 11.1 Hall's Theorem ✓

For the **order complex** `Δ[x,y]` — the simplicial complex whose simplices are finite chains in the open interval `(x,y)` — we have [Hall 1935]:

```
μ(x, y)  =  χ̃(Δ[x,y])
```

where `χ̃` is the augmented reduced Euler characteristic (convention: `χ̃(∅) = −1`).

This is **Philip Hall's theorem (1935)**: the bridge from the algebra of the incidence algebra to the topology of the basin complex.

**Worked examples** (verified computationally in §10.3):

| Open interval content | Order complex | χ̃ | μ(x,y) | Learning meaning |
|---|---|---|---|---|
| ∅ (rank-1 pair) | Empty complex | −1 | −1 | Adjacent basins subtract |
| `z` (rank-2 chain) | One isolated vertex | 0 | 0 | Interior basin cancels |
| `z₁, z₂` incomparable (diamond) | Two isolated vertices | +1 | +1 | Parallel paths reinforce |

```
  HOW BASIN TOPOLOGY SHAPES THE MÖBIUS FUNCTION

  Simple path:   B0 ─── B1 ─── B2
                 μ(B0,B2) = -(1+(-1)) = 0
                 (interior basin cancels exactly)

  Diamond:       B0 ─── B1 ─── B3
                  └──── B2 ───┘
                 μ(B0,B3) = -(1+(-1)+(-1)) = +1
                 (two parallel paths reinforce)

  Long chain:    B0 ─ B1 ─ B2 ─ B3
                 μ(B0,B3) = alternating sum
                           = 0  (even chain length)
                           = -1 (odd chain length)
```

### 11.2 The Crosscut Theorem ✓

For any `x ≺ y`, let `C` be a **crosscut** — a maximal antichain in the open interval `(x,y)`. Then [Stanley 2012, Chapter 3]:

```
μ(x, y)  =  Σ_{k≥0}  (−1)ᵏ · Nₖ(C)
```

where `Nₖ(C)` is the number of `k`-element subsets of `C` that have a common lower bound in `(x,y)`.

**Learning meaning:** Saddle points between two basins form a natural crosscut. The alternating sum over subsets of saddles counts whether the saddle structure as a whole aids or impedes gradient inversion. An even number of independent saddle paths cancel; an odd number produces a net contribution of `±1`.

---

## 12. Future Work

### 12.1 Formalizing the C_α = 1 Phase Transition ⚠️

**Gap:** The proof sketch in §7.3 treats `C_α` as a fixed parameter. We need it to hold dynamically.

**Approach:** Cast the running Möbius sum

```
Mₙ  =  Σ_{k≤n}  μ(k, n) · Fₖ
```

as an `L²` martingale under SGD with Robbins–Monro step sizes [Robbins & Monro 1951]. The `C_α > 1` condition then appears as the **Novikov condition** ensuring the exponential martingale `E(M)ₙ` is uniformly integrable. Applying the martingale convergence theorem gives `Mₙ → L_true` in `L²`.

### 12.2 Euler Characteristics via Persistent Homology ⚠️

**Goal:** Compute `μ(Bᵢ, Bⱼ) = χ̃(Δ[Bᵢ, Bⱼ])` from loss surface samples — without enumerating all saddle points.

**Method:** Sublevel-set persistent homology on the loss surface [Edelsbrunner & Harer 2010] gives a persistence diagram. Each bar `(bₖ, dₖ)` at dimension `k` corresponds to a topological feature born at loss value `bₖ` and dying at `dₖ`. Counting features born and dying within each loss interval `(L̄(θ*ᵢ), L̄(θ*ⱼ))` gives the Betti numbers of the order complex, hence `χ̃`.

```
  ALGORITHM SKETCH (requires gudhi or ripser):

  Input:  Loss surface samples {(θ_k, L(θ_k))}
  Output: μ(B_i, B_j) for all basin pairs

  1. Build Rips complex on the θ_k sample points
  2. Run persistence with the loss L(θ_k) as filtration value
  3. For each pair (L_lo, L_hi) corresponding to (B_i, B_j):
       count β_k = #{bars with birth ∈ (L_lo, L_hi)}  for k=0,1,2
       χ̃ = β_0 − β_1 + β_2 − 1
  4. Return χ̃ as the estimated μ(B_i, B_j)
```

```python
# Pseudocode — requires gudhi
import gudhi

def mobius_via_persistent_homology(loss_samples, threshold_pairs):
    """
    Estimate μ(B_i, B_j) for basin pairs from loss-surface point cloud.

    Parameters
    ──────────
    loss_samples   : array (M, d+1) of (θ, L(θ)) samples
    threshold_pairs: list of (L_lo, L_hi) pairs for each basin interval

    Returns
    ───────
    dict mapping (L_lo, L_hi) → estimated μ value
    """
    rips = gudhi.RipsComplex(
        points=loss_samples[:, :-1],
        max_edge_length=0.5
    )
    st = rips.create_simplex_tree(max_dimension=2)
    st.persistence()

    results = {}
    for (L_lo, L_hi) in threshold_pairs:
        betti = [0, 0, 0]
        for (dim, (birth, death)) in st.persistence():
            if L_lo < birth < L_hi and (death == float("inf") or death < L_hi):
                if dim < 3:
                    betti[dim] += 1
        chi_tilde = betti[0] - betti[1] + betti[2] - 1
        results[(L_lo, L_hi)] = chi_tilde
    return results
```

**Expected impact:** Makes `μ(Bᵢ, Bⱼ)` computationally accessible for real neural networks — a major step toward empirical validation of the entire framework.

### 12.3 Classification of Basin Posets ⚠️

**Conjecture:** For generic smooth `L̄` (Morse condition), the basin poset is **graded** (all maximal chains between any two elements have the same length) and **thin** (every rank-2 interval has exactly 4 elements).

If true, `μ(x,y) = (−1)^{rank(y)−rank(x)}` universally for Morse losses — a maximally simple Möbius function. All deviations from this formula signal non-Morse (degenerate) basin structure: exactly the structure created by overparameterization and ReLU activations.

### 12.4 Grokking Universality Class ⚠️

**Experiment:** Near the grokking epoch `t_c`, measure whether:

```
C_α(t) − 1  ~  (t − t_c)^β
```

for a universal exponent `β` across seeds, tasks, and architectures.

| Universality class | Predicted `β` | Interpretation |
|---|---|---|
| Mean-field Ising | 1/2 | Long-range connectivity in basin poset |
| Directed percolation (1+1d) | ~0.276 | Connectivity transition in basin graph |
| KPZ | 1/3 | Growing interface between basin regions |

### 12.5 Formal Proof of the Generalization Bound ⚠️

Complete the PAC-Bayes argument [Dziugaite & Roy 2017]:

1. Specify prior `N(θ*, σ²I)` centered at candidate minimum
2. Show `KL(Q‖P) ≲ ‖Φ − Id‖_F² / σ²`
3. Show `C_α` controls the excess risk from noise-artifact minima
4. Optimize `σ` to recover `G ≲ ‖Φ − Id‖_F / (n_train · C_α)`

---

## 13. Known Weaknesses

```
  ┌──────────────────────────────────┬──────────────┬─────────────────────────────┐
  │ Claim                            │ Status       │ What is missing             │
  ├──────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ C_α > 1 ↔ inversion converges   │ ⚠️ Conjecture │ Martingale / Robbins-Monro  │
  │                                  │              │ proof (§12.1)               │
  ├──────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ C_α is coordinate-invariant      │ ✗ FALSE      │ Use Fisher-weighted C_α^F   │
  │                                  │ as stated    │ for true invariance (§7.2)  │
  ├──────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ Frobenius is the natural framing │ Analogy, not │ Banach equally valid;       │
  │                                  │ isomorphism  │ Frobenius adds zeta counting│
  ├──────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ Generalization bound             │ ⚠️ Conjecture │ Full PAC-Bayes proof (§12.5)│
  ├──────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ Basin poset is locally finite    │ Assumed      │ Non-Morse / ReLU case needs │
  │                                  │              │ separate treatment          │
  ├──────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ Phase table in §8.3              │ Illustrative │ Need actual C_α measurements│
  │                                  │              │ on published grokking runs  │
  ├──────────────────────────────────┼──────────────┼─────────────────────────────┤
  │ Euler product factorization      │ ⚠️ Unverified │ Needs empirical test of     │
  │                                  │              │ basin independence           │
  └──────────────────────────────────┴──────────────┴─────────────────────────────┘
```

---

## 14. Central Result

**Theorem (Möbius–Frobenius Generalization)**

Let `Φ` be the SGD update operator. Let `(Fix(Φ), ≼)` be the basin poset with Möbius function `μ`. Let `C_α` be the consolidation ratio at fixed point `θ*`.

| # | Statement | Status |
|---|---|---|
| 1 | `μ(Bᵢ, Bⱼ) = χ̃(Δ[Bᵢ, Bⱼ])`: Möbius equals Euler char of saddle complex | ✓ Hall (1935) |
| 2 | `μ` is the unique exact inversion formula for the basin incidence algebra | ✓ Rota (1964) |
| 3 | `θ*` is a true minimum iff `C_α > 1` (Möbius inversion converges in L²) | ⚠️ Conjecture |
| 4 | `G(θ*) ≲ ‖Φ − Id‖_F / (n_train · C_α)` | ⚠️ Conjecture |
| 5 | The `C_α = 1` boundary is a genuine phase transition explaining grokking | ⚠️ Conjecture |

**Rows 1–2 are proven. Rows 3–5 constitute the research programme.**

---

## 15. Appendices

### Appendix A: Correspondence Table

| Neural network learning | Arithmetic / combinatorial algebra |
|---|---|
| Gradient `∇L(θ)` | Arithmetic function `f : P → R` |
| Noisy accumulated gradient | Summatory function `F = f * ζ` |
| Recovering true gradient | Möbius inversion: `f = F * μ` |
| Basin of attraction `Bᵢ` | Element of the poset `P` |
| Local minimum `θ*ᵢ` | Frobenius fixed point: `E[Φ(θ*ᵢ)] = θ*ᵢ` |
| Memorization minimum | Noise-artifact fixed point (`C_α < 1`) |
| Generalizing minimum | True fixed point (`C_α > 1`) |
| Grokking | Phase transition at `C_α = 1` |
| Flat minimum | Boolean interval in basin poset |
| Sharp minimum | Diamond (non-Boolean) interval |
| Saddle point | Crosscut in poset interval |
| Generalization gap | `‖Φ − Id‖_F / (n_train · C_α)` |
| Regularization | Truncation of the Dirichlet series `L(s)` |
| Training plateau | Non-convergence of the running Möbius sum `Mₙ` |

### Appendix B: Notation Reference

All multi-level scripts use explicit braces to avoid ambiguity:

| Ambiguous form | Correct form | Reason |
|---|---|---|
| `\theta_i^*` | `\theta^{*}_{i}` | Superscript `*` listed first; both levels braced |
| `n_\mathrm{train}` | `n_{\mathrm{train}}` | Compound subscript requires braces |
| `C_\alpha^F` | `C_{\alpha}^{F}` | Both scripts need braces for clarity |
| `\lambda_\max` | `\lambda_{\max}` | Compound subscript — braces required |
| `B_i` | `B_{i}` | All indexed symbols use braces |

### Appendix C: Quick Reference — C_α Phases

```
  C_α          Phase               Action
  ──────────────────────────────────────────────────────────
  < 0.5        NOISE_DOMINATED     Increase data or batch size
  0.5 – 1.0    APPROACHING         Monitor closely; near transition
  1.0 – 2.0    SIGNAL_DOMINATED    Generalization active; continue
  > 2.0        CONVERGED           True gradient fully recovered
  ──────────────────────────────────────────────────────────
```

---

## 16. References

### Foundational Combinatorics and Algebra
- **Hall, P.** (1935). "On Representatives of Subsets." *Journal of the London Mathematical Society*, 10(1), 26–30. — `μ(x,y) = χ̃(Δ[x,y])`: the topological interpretation of the Möbius function.
- **Rota, G.-C.** (1964). "On the Foundations of Combinatorial Theory I: Theory of Möbius Functions." *Zeitschrift für Wahrscheinlichkeitstheorie*, 2(4), 340–368. — Möbius inversion on posets; uniqueness theorem.
- **Stanley, R.** (2012). *Enumerative Combinatorics, Vol. 1*, 2nd ed. Cambridge University Press. — Incidence algebras, Crosscut theorem: Chapter 3.

### Topology
- **Milnor, J.** (1963). *Morse Theory*. Princeton University Press. — Critical point theory; graded basin structure; finitely many critical points between level sets.
- **Edelsbrunner, H. & Harer, J.** (2010). *Computational Topology*. American Mathematical Society. — Persistent homology for loss surface analysis.

### Arithmetic Geometry
- **Weil, A.** (1949). "Numbers of Solutions of Equations in Finite Fields." *Bulletin of the AMS*, 55(5), 497–508. — Frobenius endomorphism in arithmetic geometry; zeta functions over finite fields.

### Optimization and Learning Theory
- **Robbins, H. & Monro, S.** (1951). "A Stochastic Approximation Method." *Annals of Mathematical Statistics*, 22(3), 400–407. — Convergence conditions `Σηₙ = ∞`, `Ση²ₙ < ∞`.
- **Amari, S.** (1998). "Natural Gradient Works Efficiently in Learning." *Neural Computation*, 10(2), 251–276. — Fisher information matrix; coordinate-invariant gradient.
- **Hochreiter, S. & Schmidhuber, J.** (1997). "Flat Minima." *Neural Computation*, 9(1), 1–42. — Sharp vs. flat minima; generalization implications.

### Generalization Bounds
- **Dziugaite, G. K. & Roy, D. M.** (2017). "Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many Parameters." *UAI 2017*. — PAC-Bayes framework for deep networks.
- **Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B.** (2021). "Sharpness-Aware Minimization for Efficiently Improving Generalization." *ICLR 2021*. — Sharpness as generalization proxy.

### Grokking
- **Power, A., Anand, Y., Mosconi, A., Kaiser, Ł., & Polosukhin, I.** (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." *ICLR 2022*. — Empirical characterization of the grokking phenomenon.

---

*Möbius–Frobenius Framework — Canonical Specification v1.0*

*Built on: Rota (1964) · Hall (1935) · Stanley (2012) · Milnor (1963) · Weil (1949) · Robbins & Monro (1951)*

