# SOC-26 — Physics-Informed Neural Networks

This repository documents my week-by-week progress on the SOC-26: Physics-Informed Neural Networks project. It contains concise weekly reports and solved notebooks for Weeks 1–4, along with key results and next steps.

## Project Goal

Build a solid practical understanding of Physics-Informed Neural Networks (PINNs), neural networks that encode physical laws (PDEs) directly into their loss function, and apply them to progressively harder problems, culminating in Burgers' equation with hard boundary conditions.

## Repo Layout
```
SOC-26-PINNs
├── README.md
├── Week1
│   ├── Week1.md
│   └── PINNs_Week1.ipynb
├── Week2
│   ├── Week2.md
│   └── PINNs_Week2.ipynb
├── Week3
│   ├── Week3.md
│   └── PINNs_Week3.ipynb
├── Week4
    ├── Week4.md
    └── PINNs_Week4.ipynb
```
Each `WeekX.md` contains:
- Short overview of the week
- Topics covered and why they matter
- Assignments solved
- Key results and figures
- Next steps

The `solutions/` folder holds the Jupyter notebooks implementing the solved problems for that week.

## What to Expect

**Week 1 — Foundations of PINNs**
- What is a PINN and why it works
- Automatic differentiation with PyTorch
- Solving a simple 1D ODE (exponential decay) with a neural network
- Loss function combining data loss and physics residual
- First plots: predicted vs exact solution

**Week 2 — Solving PDEs with PINNs**
- Moving from ODEs to PDEs
- Solving the 1D Heat Equation: u_t = α u_xx
- Enforcing initial and boundary conditions via soft constraints
- Collocation point sampling
- Visualizing u(x,t) as a heatmap

**Week 3 — The Wave Equation & Harder Problems**
- Solving the 1D Wave Equation: u_tt = c² u_xx
- Second-order time derivatives via double autograd
- Effect of network depth, width, and activation functions
- Hyperparameter sensitivity analysis
- Comparing tanh vs sin activations

**Week 4 — Burgers' Equation & Boundary Conditions**
- Burgers' equation: the PINN benchmark (nonlinear + shock)
- Soft vs hard boundary condition enforcement
- Hard BCs via trial functions: u(x,t) = (1−x²) · NN(x,t)
- Distance function approach (Sukumar & Srivastava 2021)
- Reproducing Raissi et al. 2019 Section 3.1 from scratch in PyTorch
- Honest PINNs vs FEM comparison

## Dependencies

Libraries used across weeks:

- `torch` — neural networks and automatic differentiation
- `numpy` — numerical computation
- `matplotlib` — visualization
- `scipy` — exact solutions for error comparison

## How to Review the Work

1. Open `WeekX/WeekX.md` to read the weekly summary and findings.
2. Check `WeekX/solutions/` for the Jupyter notebook implementing the solved problems.

## Status

This repository contains completed work for Weeks 1–4 of the SOC-26 project, submitted as the midterm report.


