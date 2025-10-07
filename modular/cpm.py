# python functions
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.measure import perimeter

# my files
from modular.cpm_initializations import *


# full prelim CPM (Hamiltonian with deltaH_area & deltaH_perimeter & prelim deltaH_lum)

class CPM:
    """
    Initialize a Cellular Potts Model (CPM) simulation grid with specified parameters.

    The CPM class models cells on a 2D grid with a Hamiltonian that includes area, perimeter, 
    and illumination-related energy terms. The constructor sets up the grid, initializes 
    cells according to a chosen method, and applies a light pattern mask if provided.

    Parameters:
        grid_size : int - width and height of the square simulation grid.
        num_cells : int - number of cells to initialize on the grid
        target_area : float - ideal cell area to be used in the Hamiltonian energy calculations
        target_ratio : float - ideal ratio of cell area to perimeter to be used in the Hamiltonian energy calculations
        temperature : float - simulation temperature controlling stochastic acceptance in mc/gillespie step
        initialization : str - name of method for initializing cells on the grid. Valid options are:
            "random", "ideal", "space_filling", "voronoi", "custom1", or "custom2".
        light_pattern : array-like or None - binary 2D array indicating regions of illumination.
    Returns:
        None (Initializes the CPM grid and parameters.)
    """
    
    def __init__(self, grid_size, num_cells, target_area, target_ratio, temperature, initialization, light_function, light_pattern=None):
        self.grid_size = grid_size
        self.num_cells = num_cells
        self.target_area = target_area
        self.target_ratio = target_ratio
        self.temperature = temperature
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.mc_step = 0
        self.gill_time = 0.0
        self.light_function = light_function
        
        # initialize light pattern
        if light_pattern is not None:
            light_pattern = np.array(light_pattern, dtype=int) # make a numpy array, necessary for masking later on
            assert light_pattern.shape == (grid_size, grid_size), "light_pattern must match grid size"
            assert np.all(np.isin(light_pattern, [0, 1])), "light_pattern must be binary (0s and 1s)"
            self.light_pattern = light_pattern
        else:
            self.light_pattern = np.zeros((grid_size, grid_size), dtype=int)  # default: all dark

        init_methods = {
            "random": initialize_cells_random,
            "ideal": initialize_cells_ideal,
            "space_filling": initialize_cells_space_filling,
            "voronoi": initialize_cells_voronoi,
            "custom1": initialize_cells_custom1,
            "custom2": initialize_cells_custom2
        }

        if initialization in init_methods:
            init_methods[initialization](self)
        else:
            raise ValueError(f"Invalid initialization method: {initialization}")

        existing_cell_ids = np.unique(self.grid)
        existing_cell_ids = existing_cell_ids[existing_cell_ids != 0]
        self.num_cells = len(existing_cell_ids)