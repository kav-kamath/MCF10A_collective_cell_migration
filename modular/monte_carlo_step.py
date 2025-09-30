import numpy as np
import random
from . import hams
from .cpm import CPM

def monte_carlo_step(self: CPM):
    """
    Perform a single Monte Carlo step given a Cellular Potts Models / CPM object (as defined in this package).
    
    Parameters:
        cpm : CPM - CPM object as defined by CPM class in cpm.py
    Returns:
        None (Updates CPM grid and mc_step in place.)

    """
    
    for _ in range(self.grid_size**2):  # N random grid points
        i_x, i_y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        j_x, j_y = (i_x + dx), (i_y + dy)

        # if jx,jy is a valid grid point (no wrapping around)
        if (0 <= j_x < self.grid_size) & (0 <= j_y < self.grid_size):
            #if xi,xj and jx,jy have different cell IDs
            if (self.grid[i_y, i_x] != self.grid[j_y, j_x]):

                #old hamiltonian with old j ID
                old_j_value = self.grid[j_y, j_x]
                old_hamiltonian = hams.calculate_hamiltonian(self)

                # change j to i, calculate new hamiltonian
                self.grid[j_y, j_x] = self.grid[i_y, i_x]
                new_hamiltonian = hams.calculate_hamiltonian(self)

                # deltaH
                delta_hamiltonian = new_hamiltonian - old_hamiltonian

                if (delta_hamiltonian <= 0) or (random.random() < np.exp(-delta_hamiltonian / self.temperature)):
                    pass  # accept j -> i
                else:
                    self.grid[j_y, j_x] = old_j_value  # reject j -> i
    self.mc_step += 1 #increment time by 1 every time one full monte carlo step is complete (all N events have been attempted)
        
def mc_sim(cpm, num_steps):
    """
    Run a Monte Carlo simulation on grid of a CPM object until specified number of steps is reached.

    Parameters
        cpm : CPM - Cellular Potts Model / CPM object to simulate with
        num_steps : float - number of Monte Carlo steps to simulate

    Returns
        frames_for_plot : list of np.ndarray - list of grid (2D NumPy arrays) at/after each Monte Carlo event
        event_times : list of int - list of each  Monte Carlo event step, 1:1 corresponds to frames_for_plot
            (a bit trivial, since it's just 0, 1, 2, ..., num_steps, but corresponds to gillespie simulation output format)

    """    
    
    frames_for_plot = [cpm.grid.copy()] #initialize
    event_times = [cpm.mc_step] # initialize
    
    while cpm.mc_step <= num_steps:
        prev_time = cpm.mc_step
        monte_carlo_step(cpm)
        event_times.append(cpm.mc_step)
        print(f"Time: {cpm.mc_step}")
        frames_for_plot.append(cpm.grid.copy())
    
    return frames_for_plot, event_times