
# python imports
import numpy as np
import random
import os

from mcf10amigration.cpm_initializations import *

# my functions

from mcf10amigration import *

grid_size = 56
num_cells = 5 #relevant if using initialize_cells_random or initiliaze_cells_voronoi when constructing CPM
target_area = 37 #49 #37
target_perimeter = 20.992 #20.992 #0.00
target_ratio = 0.2534 #0.2917 #0.2534 #sqrt(area)/perimeter
temperature = 3

light_function = light.multiple_moving_bars_light

#output_folder = "trials_2cell_perim0_no_light"

cpm = CPM(grid_size, num_cells, target_area, target_perimeter, target_ratio, temperature, initialize_cells_ideal, light_function)

#frames_for_plot, event_times = gillespie_sim(cpm, max_time=0.008)
frames_for_plot, light_patterns, event_times = mc_sim(cpm, num_steps=50)

#output_path = os.path.join(output_folder, f"calc_simulation.mp4")
animate_simulation(frames_for_plot, event_times, output_filename="current_simulation.mp4")

#animate_simulation(frames_for_plot, event_times, output_filename = f"trial{i}_2cell_perim0_no_light.mp4")
#light_patterns = [lp.astype(int) for lp in light_patterns]
visualize_dynamic_light_pattern(light_patterns, event_times)
