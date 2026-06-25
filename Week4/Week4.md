# Week 4 — Burgers' Equation & Boundary Conditions

## Overview

Week 4 tackled Burgers' equation, the standard PINN benchmark from Raissi et al. 2019. This problem is nonlinear, develops a near-discontinuous shock at t=1, and has a known exact solution, making it ideal for accuracy evaluation. The key new concept was hard boundary condition enforcement via trial functions.

## Topics Covered

### Burgers' Equation
u_t + u·u_x = (ν/π) u_xx,   x ∈ [−1,1],   t ∈ [0,1]<br>
u(x, 0) = −sin(πx)     — initial condition<br>
u(−1, t) = 0,  u(1, t) = 0  — boundary conditions<br>
ν = 0.01/π   (small viscosity → sharp shock at t=1)<br>

### Soft vs Hard Boundary Conditions

**Soft BCs** — boundary loss added to total loss:
```python
L_bc = mean( (u(±1, t))² )
L_total = L_physics + λ_ic · L_ic + λ_bc · L_bc
```
The network is penalized for violating BCs but not mathematically forced.

**Hard BCs** — trial function enforces BCs exactly:
```python
def u_hard(model, x, t):
    raw = model(x, t)
    return (1 - x**2) * raw   # zero at x=±1, always
```
At x=±1: (1−1²)=0, so u=0 exactly. No BC loss term needed.

### Distance Function Approach (Sukumar & Srivastava 2021)
The trial function φ(x) = (1−x²) is a distance function, it measures proximity to the boundary and is exactly zero on it. Sukumar & Srivastava (2021) generalize this to complex 2D/3D geometries using R-functions constructed from CAD geometry, enabling hard BC enforcement on arbitrary domains without penalty terms.

### Network Architecture
- 4 hidden layers, 100 neurons each, tanh activation
- Input: [x, t], Output: u(x, t)
- Adam optimizer, lr=1e-3, 10,000 epochs

### Training Data
- 10,000 interior collocation points (random)
- 100 initial condition points
- 200 boundary points (soft BC model only)

### Exact Solution
Computed via Hopf-Cole transform using `scipy.integrate.quad` for accurate error evaluation.

## Assignment

### Part A — From Scratch (Soft BCs)
- Implemented Burgers' PINN in pure PyTorch (no DeepXDE)
- Produced: predicted u(x,t) heatmap, absolute error plot, training loss curve

### Part B — Hard BCs Comparison
- Modified Part A to use hard BC trial function
- Trained both soft and hard BC models for 10,000 epochs
- Produced: loss curves on same axes, L² error table

**Results:**

| Method   | L² Error |
|----------|----------|
| Soft BCs | ~0.02    |
| Hard BCs | ~0.01    |

Hard BCs performed better here because the network is mathematically forced to satisfy u(±1,t)=0, removing one source of loss conflict. However, on geometries where constructing a closed-form distance function is difficult (e.g. irregular domains), soft BCs are more practical.

### Part C — PINNs vs FEM (Written)

Based on Grossmann et al. (2023), PINNs rarely outperform FEM for standard forward problems. FEM solves smooth PDEs in milliseconds with guaranteed convergence. PINNs are slower, less accurate on shocks, and require careful tuning. However, PINNs have genuine advantages in specific scenarios:

**Scenario 1 — Inverse Problems:**
When you have sparse sensor measurements and want to recover an unknown PDE parameter (e.g. viscosity ν in Burgers' equation), PINNs handle this naturally by adding the unknown as a trainable parameter. FEM cannot do this without a separate expensive optimisation loop.

**Scenario 2 — High-Dimensional PDEs:**
FEM requires a mesh. In 10+ dimensions, meshing is computationally impossible (curse of dimensionality). PINNs sample random collocation points and scale naturally. Example: option pricing PDEs in finance (Black-Scholes in 100 dimensions).

**Scenario 3 — Data Assimilation:**
When you have noisy experimental data AND a known PDE, PINNs can fuse both into one training loss. Example: reconstructing a full velocity field from sparse PIV sensor readings in fluid mechanics.

**Honest Limitations:**
- PINNs struggle with sharp shocks, error peaks at t=1 in this problem
- Training is slow, sensitive to hyperparameters, not guaranteed to converge
- For smooth forward PDEs on simple domains, FEM is faster and more accurate

**Conclusion:** Use PINNs for inverse problems, parameter identification, high-dimensional PDEs, and data assimilation. Use FEM for forward problems on known geometries.

## Key Learnings

- Burgers' equation exposes the limits of PINNs — shock regions have highest error
- Hard BCs are strictly better when a trial function can be constructed
- PINNs are not a replacement for FEM — they solve a different class of problems well
- The founding paper (Raissi et al. 2019) results are reproducible from scratch in PyTorch

## References

- Raissi, Perdikaris & Karniadakis (2019) — Physics-informed neural networks, arXiv:1711.10561
- Grossmann, Komorowska, Latz, Schönlieb (2023) — Can PINNs beat FEM?, arXiv:2302.04107
- Sukumar & Srivastava (2021) — Exact imposition of BCs with distance functions, arXiv:2104.08426
- Karniadakis et al. (2021) — Physics-informed machine learning, Nature Reviews Physics
