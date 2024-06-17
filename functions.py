import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation


def u_init(xnode, numnp):
    '''Initial condition of Burgers Equation
    (input) xnode arr: Array with x values stored 
    (input) numnp int: Number of nodes

    (output) u arr: u array stores initial condition.
    '''
    u = np.zeros(numnp)
    for i in range(numnp):
        if 0 <= xnode[i] <= 0.64:
            u[i] = 1.0
        elif 0.64 <= xnode[i] <= 0.84:
            u[i] = 1 - ((xnode[i] - 0.64) / 0.20)
        else:
            u[i] = 0
    return u

def apply_boundary_conditions(u):
    '''
    (input) u arr: Solution array at specific timestep

    (output) u arr: u array with applied boundary conditions  
    '''
    u[0] = 1  # Homogeneous inflow boundary condition at the first node
    # u[numnp - 1] = 0  # Outflow boundary condition
    return u

def assemble_mass_matrix(numel, xnode, wpg, N_mef):
    '''
    (input) numel int: Number of elements
    (input) xnode arr: Array with x values 
    (input) wpg array: Array with weights 
    (input) N_mef arr: Array with the shape function 
    (input) M: Initialized M matrix

    (output) M: Return output with assembled M matrix
    ''' 
    numnp = numel + 1
    M = np.zeros((numnp, numnp))
    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        ngaus = wpg.shape[0]
        for ig in range(ngaus):
            N = N_mef[ig, :]
            w_ig = weight[ig]
            M[np.ix_(isp, isp)] += w_ig * np.outer(N, N)
    M[0,0] = 1
    return M

def assemble_flux_vector(u_current, numel, xnode, N_mef, Nxi_mef, wpg):
    '''
    (input) u_current arr: current solution array
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 

    (output) F arr: Flux value returned by function 
    '''
    F = np.zeros(len(u_current))
    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]

        u_element = u_current[isp] # gets u at element

        ngaus = wpg.shape[0]
        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]
            u_gp = np.dot(N, u_element) # calculates u at gaussian point
            f_gp = 0.5 * u_gp**2 # calculates flux at gaussian point

            F[isp] += w_ig * f_gp * Nx
    return F

def assemble_flux_vector_SU(u_current, numel, xnode, N_mef, Nxi_mef, wpg):
    '''
    (input) u_current arr: current solution array
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 

    (output) F arr: Flux value returned by function 
    '''

    F = np.zeros(len(u_current))
    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]
        u_element = u_current[isp] # gets u at element

        # calculating inf norm of tau
        u_tau = np.linalg.norm(u_element, np.inf)
        if u_tau != 0:
            tau = h/(2*u_tau)
        else: 
            tau = 0

        ngaus = wpg.shape[0]
        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]
            u_gp = np.dot(N, u_element) # calculates u at gaussian point
            f_gp = 0.5 * u_gp**2 # calculates flux at gaussian point

            u_gpx = np.dot(Nx, u_element) 
            f_gpx = u_gp * u_gpx  # DO CHAIN RULE HERE FOR IT TO MAKE SENSE!!!


            F[isp] +=  w_ig * (f_gp * Nx - N * u_gp * Nx * f_gpx * tau) 

    return F

def assemble_flux_vector_SUPG(u_current, numel, xnode, N_mef, Nxi_mef, wpg):
    '''
    (input) u_current arr: current solution array
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 

    (output) F arr: Flux value returned by function 
    '''

    F = np.zeros(len(u_current))
    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]
        u_element = u_current[isp] # gets u at element

        # calculating inf norm of tau
        u_tau = np.linalg.norm(u_element, np.inf)
        if u_tau != 0:
            tau = h/(2*u_tau)
        else: 
            tau = 0

        ngaus = wpg.shape[0]
        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]
            u_gp = np.dot(N, u_element) # calculates u at gaussian point
            f_gp = 0.5 * u_gp**2 # calculates flux at gaussian point

            u_gpx = np.dot(Nx, u_element) 
            f_gpx = u_gp * u_gpx  # DO CHAIN RULE HERE FOR IT TO MAKE SENSE!!!

            F[isp] +=  w_ig * (f_gp * Nx - N * u_gp * Nx * f_gpx * tau) 

    return F

def assemble_mass_matrix_SUPG(u_current, numel, xnode, wpg, N_mef, Nxi_mef):
    '''
    (input) u_current: current solution array
    (input) numel int: Number of elements
    (input) xnode arr: Array with x values 
    (input) wpg array: Array with weights 
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) M: Initialized M matrix
    (input) u_max: max speed at which discontinuity propagates

    (output) M: Return output with assembled M matrix
    ''' 

    numnp = numel + 1
    M = np.zeros((numnp, numnp))
    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element


        u_element = u_current[isp] # gets u at element
        ngaus = wpg.shape[0]

        # calculating inf norm of tau
        u_tau = np.linalg.norm(u_element, np.inf)
        
        if u_tau != 0:
            tau = h/(2*u_tau)
        else: 
            tau = 0

        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]
            u_gp = np.dot(N, u_element)

            M[np.ix_(isp, isp)] += w_ig * (np.outer(N, N) +  np.outer(N,Nx) * u_gp * tau * N)

    M[0,0] = 1
    return M

def plot_solution(xnode, u, t_end, nstep, stabilization_graph_title, folder_path, file_name):
    # Plot the final result at t = 0.30
    plt.plot(xnode, u[:, nstep], label=f't = {t_end}')
    plt.xlabel('x')
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.ylabel('u')
    plt.yticks(np.arange(-0.2, 1.3, 0.1))
    plt.title(f'RK4 Method for Burgers Equation {stabilization_graph_title}')
    plt.legend()
    plt.savefig(f'{folder_path}/{file_name}_t={t_end}.png')

def plot_animation(xnode, u, stabilization_graph_title, L , dt, nstep, t_end, folder_path, file_name):
    fig, ax = plt.subplots()
    line, = ax.plot(xnode, u[:, 0], label=f't = 0.0')
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title(f'RK4 Method for Burgers Equation {stabilization_graph_title}')
    ax.set_xlim(0, L)
    ax.set_ylim(-0.2, 1.2)
    ax.legend()

    def update(frame):
        line.set_ydata(u[:, frame])
        ax.legend([f't = {frame * dt:.3f}'])
        return line,

    ani = FuncAnimation(fig, update, frames=range(0, nstep + 1), blit=True)
    ani.save(f'{folder_path}/{file_name}_t={t_end}.gif', writer='imagemagick')


