import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from functions import *
from scipy.linalg import solve

# Configuration Parameters
def configure_simulation():
    t_end = float(input("Input the last timestep ------> "))
    stabilization_choice = int(input(
        "Please choose your stabilization scheme. \n"
        "1. RK4 with NO stabilization \n"
        "2. RK4 with SU \n"
        "3. RK4 with SUPG \n"
        "Type your choice here -----> "
    ))

    while stabilization_choice not in [1, 2, 3]:
        print("Invalid choice. Please type an appropriate integer (1, 2, 3) for the relevant stabilization choice.")
        stabilization_choice = int(input(
            "1. RK4 with NO stabilization \n"
            "2. RK4 with SU \n"
            "3. RK4 with SUPG \n"
            "Type your choice here -----> "
        ))

    return t_end, stabilization_choice

# Setup and Initialization
def setup_simulation(t_end, stabilization_choice):
    stabilization_graph_titles = ['(NO STAB)', '(SU)', '(SUPG)']
    folder_paths = ['./burgers_no_stabilization', './burgers_SU_stabilization', './burgers_SUPG_stabilization']
    file_names = ['burger_eq_RK4', 'burger_eq_RK4_SU', 'burger_eq_RK4_SUPG']

    L = 1.0
    numel = 100
    h = L / numel
    numnp = numel + 1
    xnode = np.linspace(0, L, numnp)
    Courant_number = 0.5
    u_max = 0.5

    dt = Courant_number * h / u_max
    nstep = int(t_end / dt)

    xipg = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    wpg = np.array([1, 1])

    N_mef = np.array([(1 - xipg) / 2, (1 + xipg) / 2])
    Nxi_mef = np.array([[-1/2, 1/2], [-1/2, 1/2]])

    u = np.zeros((numnp, nstep + 1))
    u[:, 0] = u_init(xnode, numnp)

    if stabilization_choice == 1:
        M = assemble_mass_matrix(numel, xnode, wpg, N_mef)
        folder_path = folder_paths[0]
        file_name = file_names[0]

    elif stabilization_choice == 2:
        M = assemble_mass_matrix(numel, xnode, wpg, N_mef)
        folder_path = folder_paths[1]
        file_name = file_names[1]

    elif stabilization_choice == 3:
        folder_path = folder_paths[2]
        file_name = file_names[2]

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return {
        't_end': t_end,
        'stabilization_choice': stabilization_choice,
        'stabilization_graph_title': stabilization_graph_titles[stabilization_choice - 1],
        'folder_path': folder_path,
        'file_name': file_name,
        'xnode': xnode,
        'L': L,
        'numel': numel,
        'dt': dt,
        'nstep': nstep,
        'N_mef': N_mef,
        'Nxi_mef': Nxi_mef,
        'wpg': wpg,
        'u': u,
        'M': M if stabilization_choice in [1, 2] else None
    }

# Main Time-stepping Loop
def run_simulation(config):
    for n in range(config['nstep']):
        u_temp = config['u'][:, n]
        
        if config['stabilization_choice'] == 1:
            k1 = config['dt'] * solve(config['M'], assemble_flux_vector(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))

            u_temp = config['u'][:, n] + 0.5 * k1
            u_temp = apply_boundary_conditions(u_temp)
            k2 = config['dt'] * solve(config['M'], assemble_flux_vector(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))

            u_temp = config['u'][:, n] + 0.5 * k2
            u_temp = apply_boundary_conditions(u_temp)
            k3 = config['dt'] * solve(config['M'], assemble_flux_vector(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))

            u_temp = config['u'][:, n] + k3
            u_temp = apply_boundary_conditions(u_temp)
            k4 = config['dt'] * solve(config['M'], assemble_flux_vector(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))

            config['u'][:, n + 1] = config['u'][:, n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            config['u'][:, n + 1] = apply_boundary_conditions(config['u'][:, n+1])
    
        elif config['stabilization_choice'] == 2:
            k1 = config['dt'] * solve(config['M'], assemble_flux_vector_SU(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))

            u_temp = config['u'][:, n] + 0.5 * k1
            u_temp = apply_boundary_conditions(u_temp)
            k2 = config['dt'] * solve(config['M'], assemble_flux_vector_SU(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))

            u_temp = config['u'][:, n] + 0.5 * k2
            u_temp = apply_boundary_conditions(u_temp)
            k3 = config['dt'] * solve(config['M'], assemble_flux_vector_SU(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))

            u_temp = config['u'][:, n] + k3
            u_temp = apply_boundary_conditions(u_temp)
            k4 = config['dt'] * solve(config['M'], assemble_flux_vector_SU(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))

            config['u'][:, n + 1] = config['u'][:, n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            config['u'][:, n + 1] = apply_boundary_conditions(config['u'][:, n+1])

        elif config['stabilization_choice'] == 3:
            M = assemble_mass_matrix_SUPG(u_temp, config['numel'], config['xnode'], config['wpg'], config['N_mef'], config['Nxi_mef'])
            k1 = config['dt'] * solve(M, assemble_flux_vector_SUPG(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))

            u_temp = config['u'][:, n] + 0.5 * k1
            u_temp = apply_boundary_conditions(u_temp)
            M = assemble_mass_matrix_SUPG(u_temp, config['numel'], config['xnode'], config['wpg'], config['N_mef'], config['Nxi_mef'])
            k2 = config['dt'] * solve(M, assemble_flux_vector_SUPG(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))

            u_temp = config['u'][:, n] + 0.5 * k2
            u_temp = apply_boundary_conditions(u_temp)
            M = assemble_mass_matrix_SUPG(u_temp, config['numel'], config['xnode'], config['wpg'], config['N_mef'], config['Nxi_mef'])
            k3 = config['dt'] * solve(M, assemble_flux_vector_SUPG(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))

            u_temp = config['u'][:, n] + k3
            u_temp = apply_boundary_conditions(u_temp)
            M = assemble_mass_matrix_SUPG(u_temp, config['numel'], config['xnode'], config['wpg'], config['N_mef'], config['Nxi_mef'])
            k4 = config['dt'] * solve(M, assemble_flux_vector_SUPG(u_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg']))
 
            config['u'][:, n + 1] = config['u'][:, n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            config['u'][:, n + 1] = apply_boundary_conditions(config['u'][:, n+1])



# Main Execution
def main():
    t_end, stabilization_choice = configure_simulation()
    config = setup_simulation(t_end, stabilization_choice)
    run_simulation(config)
    plot_solution(config['xnode'], config['u'], t_end, config['nstep'], config['stabilization_graph_title'], config['folder_path'], config['file_name'])
    plot_animation(config['xnode'], config['u'], config['stabilization_graph_title'], config['L'], config['dt'], config['nstep'], config['t_end'], config['folder_path'], config['file_name'] )

if __name__ == "__main__":
    main()
