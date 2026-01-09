
##### OUTDATED - STOPPED UPDATING TO MATCH W REST OF CPDE #####

import numpy as np

from . import hams
from .cpm import CPM
from .light import update_light
from tqdm.notebook import tqdm
from IPython.display import display

def gillespie_step(cpm: CPM):
    """
    Perform a single Gillespie step given a Cellular Potts Models / CPM object (as defined in this package).
    
    Parameters:
        cpm : CPM - CPM object as defined by CPM class in cpm.py
    Returns:
        None (Updates  CPM grid and gill_time in place.)
    """    
    
    cpm.light_pattern[:,:] = update_light(cpm.grid_size, cpm.light_function, cpm.gill_time)
    
    events = []
    rates = []

    old_hamiltonian = hams.calculate_hamiltonian(cpm)
    
    # all possible copy events and their rates (probability of occuring)
    for i_y in range(cpm.grid_size):
        for i_x in range(cpm.grid_size):
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                j_x, j_y = i_x + dx, i_y + dy
                if 0 <= j_x < cpm.grid_size and 0 <= j_y < cpm.grid_size:
                    if cpm.grid[i_y, i_x] != cpm.grid[j_y, j_x]:
                        # calculate deltaH for this event
                        old_j_value = cpm.grid[j_y, j_x]
                        cpm.grid[j_y, j_x] = cpm.grid[i_y, i_x]
                        new_hamiltonian = hams.calculate_hamiltonian(cpm)
                        cpm.grid[j_y, j_x] = old_j_value  # revert

                        deltaH = new_hamiltonian - old_hamiltonian
                        # rate: exp(-deltaH/T) if deltaH > 0, else 1
                        if not np.isnan(deltaH):
                            rate = np.exp(-deltaH / cpm.temperature) #1.0
                            print("deltaH calc-ed")
                        else:
                            print("deltaH is nan")
                        events.append(((i_x, i_y), (j_x, j_y)))
                        rates.append(rate)
    
    # cell empty evenst
    for y in range(cpm.grid_size):
        for x in range(cpm.grid_size):
            if cpm.grid[y, x] != 0:
                original_id = cpm.grid[y, x]
                cpm.grid[y, x] = 0
                new_hamiltonian = hams.calculate_hamiltonian(cpm)
                cpm.grid[y, x] = original_id  # revert

                deltaH = new_hamiltonian - old_hamiltonian
                rate = np.exp(-deltaH / cpm.temperature) if deltaH > 0 else 1.0
                events.append(((x, y), "EMPTY"))
                rates.append(rate)                    
                                    
    total_rate = np.sum(rates)
    if total_rate < 1e-12:  # event veryyyy unlikely (make it total_rate==0 causing NAN error)
        print("minimal total rate")
        return  # no possible events

    #print(events)
    #print(rates)
    
    # proportionally choose which event occurs  
    chosen_index = np.random.choice(len(events), p=np.array(rates)/total_rate)
    chosen_event = events[chosen_index]        
    
    if chosen_event[1] == "EMPTY":
        x, y = chosen_event[0]
        cpm.grid[y, x] = 0
    else:
        (i_x, i_y), (j_x, j_y) = chosen_event
        cpm.grid[j_y, j_x] = cpm.grid[i_y, i_x]
    

    # move forward in time, probability of any event occuring (like aggregated poisson)
    # inverse transform sampling method
    U = np.random.uniform() #choose random number from uniform dist [0, 1) 
    delta_t = -np.log(U) / total_rate # waiting time for next event is expential; adds up to poisson process over many events
    cpm.gill_time += delta_t
    
    
def gillespie_sim(cpm: CPM, max_time):
    """
    Run a Gillespie simulation on grid of a CPM object until specified simulation time is reached.
    
    Parameters
        cpm : CPM - Cellular Potts Model / CPM object to simulate with
        max_time : float - amount of time simulation will run until (arbitrary time units / units contrived from model)

    Returns
        frames_for_plot : list of np.ndarray - list of grid (2D NumPy arrays) at/after each Gillespie event
        event_times : list of float - list each time a gillespie event occured, 1:1 corresponds to frames_for_plot
            (arbitrary time units / units contrived from model)
    """
    
    #initializations
    frames_for_plot = [cpm.grid.copy()]
    light_patterns = [cpm.light_pattern.copy()]
    event_times = [cpm.gill_time]
    
    progress_bar = tqdm(total=max_time, desc="Gillespie Simulation", unit_scale=True)
    display(progress_bar)
    # run sim
    while cpm.gill_time < max_time:
        prev_time = cpm.gill_time
        
        gillespie_step(cpm)
        event_times.append(cpm.gill_time)
        #print(f"Time: {cpm.gill_time}")
        
        frames_for_plot.append(cpm.grid.copy())
        light_patterns.append(cpm.light_pattern.copy())
        
        tc = cpm.gill_time - prev_time
        if tc > 0:
            progress_bar.update(tc)
    progress_bar.close()

    return frames_for_plot, light_patterns, event_times