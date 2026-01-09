from mcf10amigration.cpm_initializations import initialize_cells_custom2
import numpy as np
import random
from . import hams
from .cpm import CPM
from .light import update_light
from tqdm import tqdm
from dataclasses import dataclass
import time

@dataclass
class SimulationResult:
    metadata: dict
    cell_states: list
    light_patterns: list
    event_times: list

def monte_carlo_step(cpm: CPM):
    """
    Perform a single Monte Carlo step given a Cellular Potts Models / CPM object (as defined in this package).
    
    Parameters:
        cpm : CPM - CPM object as defined by CPM class in cpm.py
    Returns:
        None (Updates CPM grid and mc_step in place.)

    """
    cpm.light_pattern[:,:] = update_light(cpm.grid_size, cpm.light_function, cpm.mc_step)

    current_hamiltonian = hams.calculate_hamiltonian(cpm)
    #print("current: ", current_hamiltonian)

    for n in range(cpm.grid_size**2):  # N random grid points
        i_x, i_y = random.randint(0, cpm.grid_size - 1), random.randint(0, cpm.grid_size - 1)
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        j_x, j_y = (i_x + dx), (i_y + dy)

        # if jx,jy is a valid grid point (no wrapping around)
        if not (0 <= j_x < cpm.grid_size) & (0 <= j_y < cpm.grid_size):
            continue

        #if xi,xj and jx,jy have different cell IDs
        if  not (cpm.grid[i_y, i_x] != cpm.grid[j_y, j_x]):
            continue

        # change j to i, calculate new hamiltonian
        old_j_value = cpm.grid[j_y, j_x]
        cpm.grid[j_y, j_x] = cpm.grid[i_y, i_x]
        new_hamiltonian = hams.calculate_hamiltonian(cpm)
        #print("new: ", new_hamiltonian)

        # deltaH
        delta_hamiltonian = new_hamiltonian - current_hamiltonian
        #print("delta: ", delta_hamiltonian)

        if (delta_hamiltonian <= 0) or (random.random() < np.exp(-delta_hamiltonian / cpm.temperature)):
            current_hamiltonian = new_hamiltonian
            #print(f"move {n} accepted")
            #print("-----move over------")

        else:
            cpm.grid[j_y, j_x] = old_j_value  # reject j -> i
            #print(f"move {n} rejected")
            #print("-----move over------")

    #print(f"-----mc step {cpm.mc_step} over------")
    cpm.mc_step += 1  # increment time by 1 every time one full monte carlo step is complete (all N events have been attempted)


    # old implementation: pick N random grid points, attempt copy for each    
    """
    for _ in range(cpm.grid_size**2):  # N random grid points
        i_x, i_y = random.randint(0, cpm.grid_size - 1), random.randint(0, cpm.grid_size - 1)
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        j_x, j_y = (i_x + dx), (i_y + dy)

        # if jx,jy is a valid grid point (no wrapping around)
        if (0 <= j_x < cpm.grid_size) & (0 <= j_y < cpm.grid_size):
            #if xi,xj and jx,jy have different cell IDs
            if (cpm.grid[i_y, i_x] != cpm.grid[j_y, j_x]):

                #old hamiltonian with old j ID
                old_j_value = cpm.grid[j_y, j_x]
                old_hamiltonian = hams.calculate_hamiltonian(cpm)

                # change j to i, calculate new hamiltonian
                cpm.grid[j_y, j_x] = cpm.grid[i_y, i_x]
                new_hamiltonian = hams.calculate_hamiltonian(cpm)

                # deltaH
                delta_hamiltonian = new_hamiltonian - old_hamiltonian

                if (delta_hamiltonian <= 0) or (random.random() < np.exp(-delta_hamiltonian / cpm.temperature)):
                    pass  # accept j -> i
                else:
                    cpm.grid[j_y, j_x] = old_j_value  # reject j -> i
                    
    cpm.mc_step += 1 #increment time by 1 every time one full monte carlo step is complete (all N events have been attempted)
    """
        
def mc_sim(cpm, num_steps) -> SimulationResult:
    """
    Run a Monte Carlo simulation on grid of a CPM object until specified number of steps is reached.

    Parameters
        cpm : CPM - Cellular Potts Model / CPM object to simulate with
        num_steps : float - number of Monte Carlo steps to simulate

    Returns
        SimulationResult data class with:
            metadata : dict - of cpm input values
            cell_states : list of np.ndarray - list of grid (2D NumPy arrays) at/after each Monte Carlo event
            light_patterns: list of np.ndarray - list of light grid (2D NumPy arrays) at/after each Monte Carlo event
            event_times: list of int - list of each  Monte Carlo event step, 1:1 corresponds to frames_for_plot
                (a bit trivial, since it's just 0, 1, 2, ..., num_steps, but corresponds to gillespie simulation output format)
    """    

    start_time = time.time()

    # initialize
    cpm.initialization(cpm)
    
    cell_states = [cpm.grid.copy()] #initialize
    light_patterns = [cpm.light_pattern.copy()] #initialize
    event_times = [cpm.mc_step] # initialize
    
    #while cpm.mc_step <= num_steps:
    for _ in tqdm(range(cpm.mc_step, num_steps), desc="Monte Carlo Simulation"):
        #prev_time = cpm.mc_step
        monte_carlo_step(cpm)
        event_times.append(cpm.mc_step)
        #print(f"Time: {cpm.mc_step}")
        cell_states.append(cpm.grid.copy())
        light_patterns.append(cpm.light_pattern.copy())
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    metadata = {
    "grid_size": cpm.grid_size,
    "num_cells": cpm.num_cells,
    "target_area": cpm.target_area,
    "target_perimeter": cpm.target_perimeter,
    "k": cpm.k,
    "temperature": cpm.temperature,
    "tissue_size": cpm.tissue_size,
    "margin": cpm.margin,
    "light_function": cpm.light_function,
    "simulation_runtime" : elapsed_time,
    }
    
    return SimulationResult(metadata, cell_states, light_patterns, event_times)