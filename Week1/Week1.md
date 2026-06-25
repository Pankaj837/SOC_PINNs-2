# Week 1 — Foundations of PINNs

## Overview

Week 1 introduced the core idea behind Physics-Informed Neural Networks. The goal was to understand what makes a PINN different from a standard neural network, implement automatic differentiation in PyTorch, and solve a simple 1D ODE using a neural network for the first time.

## Topics Covered

### What is a PINN?
- A neural network that satisfies a physical law (ODE or PDE) as part of its training
- The loss function has two parts: data/boundary loss + physics residual loss
- No finite difference grid needed — the network is trained on random collocation points

### Automatic Differentiation
- PyTorch `autograd` computes exact derivatives of neural network outputs
- `torch.autograd.grad` used to compute du/dx, du/dt symbolically through the network
- Why this is different from numerical differentiation (no discretization error)

### ODE Solved: Exponential Decay

du/dt = -λu,   u(0) = 1,   λ = 1<br>
Exact solution: u(t) = e^(−t)

### Network Architecture
- Input: t (scalar time)
- 3 hidden layers, 32 neurons each, tanh activation
- Output: u(t) (scalar)

### Loss Function

L = L_physics + L_initial<br>
L_physics = mean( (du/dt + λu)² )   over collocation points<br>
L_initial = (u(0) − 1)²<br>

## Assignment
- Implemented a PINN from scratch in PyTorch to solve exponential decay
- Trained with Adam optimizer for 3,000 epochs
- Plotted predicted vs exact solution
- Observed how physics loss drives the network to satisfy the ODE even without data

## Key Results
- PINN successfully recovered u(t) = e^(−t) with high accuracy
- Physics loss and initial condition loss both converged to near zero
- The network learned the correct solution using only 100 collocation points

## Key Learnings

- PINNs embed physical knowledge directly into training, no labeled data needed
- Autograd makes computing PDE residuals exact and efficient
- The balance between physics loss and IC loss matters for convergence

## Next Steps
In **Week 2**, the focus shifts to PDEs, specifically the 1D Heat Equation, introducing spatial derivatives and the concept of collocation points in 2D (space + time).
