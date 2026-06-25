# Week 3 — The Wave Equation & Harder Problems

## Overview

Week 3 tackled the 1D Wave Equation, a hyperbolic PDE requiring second-order derivatives in both space and time. This introduced double autograd differentiation and raised the question of how network architecture affects PINN performance.

## Topics Covered

### The 1D Wave Equation
u_tt = c² u_xx,   x ∈ [0,1],   t ∈ [0,1]<br>
u(x, 0) = sin(πx)     — initial displacement<br>
u_t(x, 0) = 0         — initial velocity<br>
u(0, t) = 0,  u(1, t) = 0  — boundary conditions<br>
Exact solution: u(x,t) = sin(πx)cos(πct)<br>

### Second-Order Derivatives via Double Autograd
- u_tt computed by differentiating u_t once more through the graph
- Requires `create_graph=True` at each differentiation step
- More expensive than first-order PDEs but fully automatic

### Network Architecture Study
- Compared shallow vs deep networks (2 vs 4 hidden layers)
- Compared narrow vs wide (32 vs 128 neurons)
- Compared tanh vs sin (Fourier feature) activations
- Deeper + wider networks performed better for oscillatory solutions

### Loss Function
L = L_physics + λ_ic1 · L_ic_u + λ_ic2 · L_ic_ut + λ_bc · L_bc<br>
L_physics = mean( (u_tt − c² u_xx)² )<br>
L_ic_u    = mean( (u(x,0) − sin(πx))² )<br>
L_ic_ut   = mean( (u_t(x,0))² )<br>
L_bc      = mean( (u(0,t))² + (u(1,t))² )<br>

## Assignment
- Implemented Wave Equation PINN in PyTorch
- Enforced both initial displacement and initial velocity conditions
- Ran architecture comparison experiments
- Produced heatmaps and loss curves for each configuration

## Key Results
- PINN successfully reproduced oscillatory wave behavior
- sin activation outperformed tanh for high-frequency solutions
- Deeper networks converged to lower error but took longer to train
- Error accumulated over time, a known limitation of PINNs for wave problems

## Key Learnings
- Second-order PDEs require careful autograd chaining
- Activation function choice matters, periodic functions suit periodic solutions
- PINNs accumulate temporal error; this is an active research problem

## Next Steps

In **Week 4**, we tackle Burgers' equation, the canonical PINN benchmark, introducing nonlinearity, shock formation, and hard boundary condition enforcement.
