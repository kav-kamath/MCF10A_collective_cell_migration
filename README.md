# Full README coming soom :)
 
***Directory of functions:***

**cpm.py**
class CPM

**cpm_initializations.py**
initialize_cells_random(cpm: CPM)
initialize_cells_ideal(cpm: CPM)
initialize_cells_space_filling(cpm: CPM)
initialize_cells_voronoi(cpm: CPM)
initialize_cells_tissue_sparse(cpm: CPM)
initialize_cells_tissue_dense(cpm: CPM)
initialize_cells_wound(cpm: CPM)
initialize_cells_custom_centers(cpm: CPM)
initialize_cells_custom_grid(cpm: CPM)

**light.py**
static_circle_light(y, x, t, cpm: CPM)
static_left_light(y, x, t, cpm: CPM)
static_right_light(y, x, t, cpm: CPM)
no_light (y, x, t, cpm: CPM)
light_spreading_from_corner(y, x, t, cpm: CPM)
shrinking_circle_light(y, x, t, cpm: CPM)
growing_circle_light(y, x, t, cpm: CPM)	
outward_circle_wave_light(y, x, t, cpm: CPM)
multiple_outward_circle_waves_light(y, x, t, cpm: CPM)
inward_circle_wave_light(y, x, t, cpm: CPM)
multiple_inward_circle_waves_light(y, x, t, cpm: CPM):
moving_bar_light(y, x , t, cpm: CPM)
multiple_moving_bars_light(y, x , t, cpm: CPM)
update_light(grid_size, light_function, time_step, cpm: CPM)

**hams.py**
calculate_perimeter(cpm: CPM, cell_id)	
fraction_illuminated(cpm: CPM, cell_id)
cell_contains_holes(cpm: CPM, cell_id)
calculate_hamiltonian(cpm: CPM)

**monte_carlo_step.py**
dataclass SimulationResult
monte_carlo_step(cpm: CPM)
mc_sim(cpm, num_steps)

**simulation_analysis.py**
radial_density(frame: np.ndarray, bin_width:int = 1, method:str = "cell_area")
inside_circle_count(frame, cy, cx, radius)
inside_square_count(frame, cy, cx, halfwidth):
avg_displacement(start_frame, end_frame, direction=None)
avg_distance_from_point(frames, point)
visualize_displacement(start_frame, end_frame, title="individual displacement", output_filename="fig.png", save_fig=True)
cosine_similarity(start_frame, end_frame, target_point)
middle_zero_region_size(frame)

**visuals.py**
plot_static_light_pattern(light_pattern, save_boolean=False, output_filename="static_light_pattern.png")
animate_light_pattern(light_patterns, times, background_color=(1, 1, 1), save_boolean=True, output_filename="dynamic_light_pattern.mp4", fps=5)
_animate_cell_simulation_old(frames, times, background_color=(1, 1, 1), save_boolean=True, output_filename="current_simulation.mp4", fps=5)
animate_cell_simulation(frames, times, background_color=(1, 1, 1), save_boolean=True, output_filename="current_simulation.mp4", fps=5)
plot_one_frame(frame, title = None, output_filename = "fig.png", dpi=600, save_fig=True)
plot_frames(frames, cmap, rows: int, columns: int, figsize, subplot_titles = None, title=None, dpi = 600, save_fig=True, filename="fig.png")
_plot_5x5(frames, cmap = None)
plot_event_times(event_times, sim_type="Gillespie")

**gillespie_step.py**
gillespie_step(cpm: CPM)
gillespie_sim(cpm: CPM, max_time)
