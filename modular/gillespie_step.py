import numpy as np

from .cpm import CPM

def gillespie_step(cpm: CPM):
    events = []
    rates = []

    old_hamiltonian = cpm.calculate_hamiltonian()
    
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
                        new_hamiltonian = cpm.calculate_hamiltonian()
                        cpm.grid[j_y, j_x] = old_j_value  # revert

                        deltaH = new_hamiltonian - old_hamiltonian
                        # rate: exp(-deltaH/T) if deltaH > 0, else 1
                        if not np.isnan(deltaH):
                            rate = np.exp(-deltaH / cpm.temperature) #1.0
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
                new_hamiltonian = cpm.calculate_hamiltonian()
                cpm.grid[y, x] = original_id  # revert

                deltaH = new_hamiltonian - old_hamiltonian
                rate = np.exp(-deltaH / cpm.temperature) if deltaH > 0 else 1.0
                events.append(((x, y), "EMPTY"))
                rates.append(rate)                    
                                    
    total_rate = np.sum(rates)
    if total_rate < 1e-12:  # event veryyyy unlikely (make it total_rate==0 causing NAN error)
        return  # no possible events

    #print(events)
    #print(rates)
    #print(total_rate)
    
    
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
    
    
def gillespie_sim(cpm, max_time,):
    frames_for_plot = [cpm.grid.copy()]
    event_times = [cpm.gill_time]
    
    while cpm.gill_time < max_time:
        prev_time = cpm.gill_time
        gillespie_step(cpm)
        event_times.append(cpm.gill_time)
        #print(f"Time: {cpm.gill_time}")
        frames_for_plot.append(cpm.grid.copy())
    
    return frames_for_plot, event_times