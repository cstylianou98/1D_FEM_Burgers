# 1D Finite Element Method (FEM) Solver - Burgers Equation

## Description
This project employs a one-dimensional Finite Element Method (FEM) solver in Python. It uses the Galerkin method with 1D Lagrange basis functions and 2nd order Gaussian Quadrature for numerical integration. The solver is capable of solving the Burgers equation using Classical Runge Kutta 4 methods for time-stepping whilst having the option of stabilizing the solution using Streamline Upwind (SU) and Streamline Upwind Petrov-Garlerkin (SUPG) techniques.

This code has been part of my work at the Barcelona Supercomputer Centre (BSC) and aims to be useful for people starting their CFD journey using FEM solvers.

## Features
-**Solver Methods**: Implements the RK4 method with the option of: 1. Not stabilizing, 2. Stabilizing with SU and 3. Stabilizing with SUPG
-**Basis Functions**: Uses 1D Lagrange basis functions for spatial discretization.
-**Gaussian Quadrature**: Employs 2nd Order Gaussian Quadrature for numerical integration.
-**User Interaction**: Allows user input for simulation time and solver method.

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- os
- Scipy

## Usage
To run the solver, execute 'main.py' script in a Python environment:

```
python main.py
```

Follow the on-screen prompts to enter the last timestep and the preferred type of stabilization.

## Output
The program computes and plots the numerical solution of the Burger Equation given initial and boundary conditions explained below.

# Problem Statement

The code focuses on the solution of the Burgers Equation given by:

$$
u_t + f(u)_x = 0, \quad where \quad f(u) = \frac{u^2}{2}
$$

With initial condition given by:

$$
u(x, 0) = \begin{cases}
    1 &  0 \leq x \leq 0.64 \\
    1-\frac{(x-0.64)}{0.20} & 0.64 \leq x \leq 0.84 \\
    0 &  0.84 \leq x \leq 1
\end{cases}
$$

and Dirichlet inflow boundary condition:

$$
u(0, t) = 1
$$

### Sample results
The solver uses a default number of elements of 100 and a timestep calculated by: 

$$
\Delta t = (Courant * h) / u_{max}
$$

Sample results can be found in the "burgers_no_stabilization", "burgers_SU_stabilization" and "burgers_SUPG_stabilization" folders of this repository.

