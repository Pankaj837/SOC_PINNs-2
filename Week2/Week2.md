# Week 2 — Solving PDEs with PINNs

## Overview

Week 2 extended the PINN framework from ODEs to PDEs. The 1D Heat Equation was solved, requiring the network to handle both spatial and temporal derivatives simultaneously. Boundary and initial conditions were enforced as soft constraints via additional loss terms.

## Topics Covered

### The 1D Heat Equation
u_t = α u_xx,   x ∈ [0,1],   t ∈ [0,1]<br>
u(x, 0) = sin(πx) — initial condition<br>
u(0, t) = 0,  u(1, t) = 0 — boundary conditions<br>
Exact solution: u(x,t) = e^(−α π² t) sin(πx)<br>

### Collocation Points
- Interior points sampled randomly in (x, t) space, physics enforced here
- Boundary points: x=0 and x=1 at random times
- Initial condition points: t=0 at random x values
- No mesh or grid required

### Network Architecture
- Input: [x, t] (2D)
- 4 hidden layers, 64 neurons each, tanh activation
- Output: u(x, t)

### Loss Function
L = L_physics + λ_ic · L_ic + λ_bc · L_bc<br>
L_physics = mean( (u_t − α u_xx)² )<br>
L_ic      = mean( (u(x,0) − sin(πx))² )<br>
L_bc      = mean( (u(0,t))² + (u(1,t))² )<br>

### Visualization
- u(x,t) plotted as a 2D heatmap (x on y-axis, t on x-axis)
- Absolute error |u_pinn − u_exact| plotted on the same grid

## Assignment
- Implemented Heat Equation PINN in PyTorch
- Used 5,000 interior collocation points, 200 BC points, 100 IC points
- Trained with Adam optimizer for 5,000 epochs
- Produced heatmap of predicted solution and absolute error plot

## Key Results

- PINN accurately reproduced the decaying sine solution
- Error was highest near t=0 where gradients are steepest
- Loss converged smoothly with proper weighting of boundary terms

## Key Learnings

- Extending PINNs to PDEs requires handling multiple input dimensions
- Soft BCs add penalty terms to the loss, the network is encouraged but not forced to satisfy them
- Loss weighting (λ_ic, λ_bc) significantly affects convergence speed

## Next Steps

In **Week 3**, we tackle the Wave Equation, introducing second-order time derivatives and studying how network architecture choices affect solution quality.
