import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

from functions import *
from scipy.linalg import solve
from scipy.linalg import lu

#RUNGE KUTTA 4 CLASSIC
# User prompts
t_end = float(input("Input a the last timestep ------> "))
stabilization_choice = int(input("Please choose your stabilization scheme. \n1. RK4 with NO stabilization \n2. RK4 with SU \n3. RK4 with SUPG \nType your choice here -----> "))

# Ensure the input is valid
while stabilization_choice not in [1, 2, 3]:
    print("Invalid choice. Please type an appropriate integer (1, 2) for the relevant stabilization choice.")
    stabilization_choice = input("\n1. RK4 with NO stabilization \n2. RK4 with SU \n3. RK4 with SUPG \nType your choice here -----> ").upper()


# Naming based on input choice
stabilzation_graph_titles = ['(NO STAB)', '(SU)', '(SUPG)']
folder_paths = ['./burgers_no_stabilization', './burgers_SU_stabilization', './burgers_SUPG_stabilization']
file_names = ['burger_eq_RK4', 'burger_eq_RK4_SU', 'burger_eq_RK4_SUPG'] 

# Parameters
L = 1.0
numel = 100
h = L / numel
numnp = numel + 1
xnode = np.linspace(0, L, numnp)
Courant_number = 0.5
u_max = 0.5  # Assuming u_max = 0.5 based on the speed of propagation of the discontinuity.

# Time step calculation
dt = Courant_number * h / u_max
nstep = int(t_end / dt)

# Gauss points and weights for [-1, 1]
xipg = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
wpg = np.array([1, 1])

# Number of Gauss points on each element
ngaus = wpg.shape[0]

# Shape functions and their derivatives on reference element
N_mef = np.array([(1 - xipg) / 2, (1 + xipg) / 2])
Nxi_mef = np.array([[-1/2, 1/2], [-1/2, 1/2]])

u = np.zeros((numnp, nstep + 1))

# Initial condition
u[:, 0] = u_init(xnode, numnp)



if stabilization_choice == 1:
    M = assemble_mass_matrix(numel, xnode, wpg, N_mef)
    stabilzation_graph_title = stabilzation_graph_titles[0]
    
    folder_path = folder_paths[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = file_names[0]
    print('Your non-stabilized graph has been saved')

elif stabilization_choice ==2:
    M = assemble_mass_matrix(numel, xnode, wpg, N_mef)
    stabilzation_graph_title = stabilzation_graph_titles[1]
    
    folder_path = folder_paths[1]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = file_names[1]

elif stabilization_choice == 3:
    stabilzation_graph_title = stabilzation_graph_titles[2]
    folder_path = folder_paths[2]
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = file_names[2]

# Main time-stepping loop
if stabilization_choice == 1:

    for n in range(nstep):
        u_temp = u[:, n]
        
        # k1 step
        F = assemble_flux_vector(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k1 = dt * solve(M, F)

        # k2 step
        u_temp = u[:, n] + 0.5 * k1
        u_temp = apply_boundary_conditions(u_temp)
        F = assemble_flux_vector(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k2 = dt * solve(M, F)

        # k3 step
        u_temp = u[:, n] + 0.5 * k2
        u_temp = apply_boundary_conditions(u_temp)
        F = assemble_flux_vector(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k3 =  dt * solve(M, F)

        # k4 step
        u_temp = u[:, n] + k3
        u_temp = apply_boundary_conditions(u_temp)
        F = assemble_flux_vector(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k4 = dt * solve(M, F)

        # Update solution
        u[:, n + 1] = u[:, n] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Apply boundary conditions again
        u[:, n + 1] = apply_boundary_conditions(u[:, n+1])

elif stabilization_choice == 2:
     for n in range(nstep):
        u_temp = u[:, n]
        # k1 step
        F = assemble_flux_vector_SU(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k1 = dt * solve(M, F)

        # k2 step
        u_temp = u[:, n] + 0.5 * k1
        u_temp = apply_boundary_conditions(u_temp)
        F = assemble_flux_vector_SU(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k2 = dt * solve(M, F)

        # k3 step
        u_temp = u[:, n] + 0.5 * k2
        u_temp = apply_boundary_conditions(u_temp)
        F = assemble_flux_vector_SU(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k3 =  dt * solve(M, F)

        # k4 step
        u_temp = u[:, n] + k3
        u_temp = apply_boundary_conditions(u_temp)
        F = assemble_flux_vector_SU(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k4 = dt * solve(M, F)

        # Update solution
        u[:, n + 1] = u[:, n] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Apply boundary conditions again
        u[:, n + 1] = apply_boundary_conditions(u[:, n+1])

elif stabilization_choice == 3:
     for n in range(nstep):
        u_temp = u[:, n]
        
        # k1 step 
        M = assemble_mass_matrix_SUPG(u_temp, numel, xnode, wpg, N_mef, Nxi_mef)
        F = assemble_flux_vector_SUPG(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k1 = dt * solve(M, F)

        # k2 step
        u_temp = u[:, n] + 0.5 * k1
        u_temp = apply_boundary_conditions(u_temp)
        M = assemble_mass_matrix_SUPG(u_temp, numel, xnode, wpg, N_mef, Nxi_mef)
        F = assemble_flux_vector_SUPG(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k2 = dt * solve(M, F)

        # k3 step
        u_temp = u[:, n] + 0.5 * k2
        u_temp = apply_boundary_conditions(u_temp)
        M = assemble_mass_matrix_SUPG(u_temp, numel, xnode, wpg, N_mef, Nxi_mef)
        F = assemble_flux_vector_SUPG(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k3 =  dt * solve(M, F)

        # k4 step
        u_temp = u[:, n] + k3
        u_temp = apply_boundary_conditions(u_temp)
        M = assemble_mass_matrix_SUPG(u_temp, numel, xnode, wpg, N_mef, Nxi_mef)
        F = assemble_flux_vector_SUPG(u_temp, numel, xnode, N_mef, Nxi_mef, wpg)
        k4 = dt * solve(M, F)

        # Update solution
        u[:, n + 1] = u[:, n] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Apply boundary conditions again
        u[:, n + 1] = apply_boundary_conditions(u[:, n+1])

# Plot the final result at t = 0.30
plt.plot(xnode, u[:, nstep], label=f't = {t_end}')
plt.xlabel('x')
plt.xticks(np.arange(0.0, 1.1, 0.1))
plt.ylabel('u')
plt.yticks(np.arange(-0.2, 1.3, 0.1))
plt.title(f'RK4 Method for Burgers Equation {stabilzation_graph_title}')
plt.legend()
plt.savefig(f'{folder_path}/{file_name}_t={t_end}.png')

# Create an animation

# Animation setup
fig, ax = plt.subplots()
line, = ax.plot(xnode, u[:, 0], label=f't = 0.0')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title(f'RK4 Method for Burgers Equation {stabilzation_graph_title}')
ax.set_xlim(0, L)
ax.set_ylim(-0.2, 1.2)
ax.legend()

def update(frame):
    line.set_ydata(u[:, frame])
    ax.legend([f't = {frame * dt:.3f}'])
    return line,

ani = FuncAnimation(fig, update, frames=range(0, nstep + 1), blit=True)
# ani.save(f'{save_file_name}_t={t_end}.gif', writer='imagemagick')
