# python functions
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.measure import perimeter

# my files
from .cpm_initializations import *


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
    
    
    def __init__(self, grid_size, num_cells, target_area, target_ratio, temperature, initialization, light_pattern):
        self.grid_size = grid_size
        self.num_cells = num_cells
        self.target_area = target_area
        self.target_ratio = target_ratio
        self.temperature = temperature
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.mc_step = 0
        self.gill_time = 0.0
        
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
        
    
    
    
    
    
    
    
    # where hamiltonian function + helper functions used to be
    """
    ###### ACTUAL FUNCTIONS ###### ==> FEATURES OF CELL

    # updated to be skimage.measure.regionprops() perimeter
    def calculate_perimeter(self, cell_id):

        # skimage.measure.regionprops() perimeter
        binary_grid = (self.grid == cell_id)
        perimeter_value = perimeter(binary_grid, neighborhood = 8)

        return perimeter_value

    def fraction_illuminated(self, cell_id):

        cell_mask = (self.grid == cell_id) # t/f mask of cell location
        light_mask = (self.light_pattern == 1) # t/f mask of light location
        
        overlap = cell_mask & light_mask # AND of both masks, where both true

        area_in_light = np.sum(overlap)
        total_area = np.sum(cell_mask)
        
        if total_area == 0:
            return 0.0
        return area_in_light / total_area

    def cell_contains_holes(self, cell_id):
        
        cell_mask = (self.grid == cell_id) # binary mask for the cell
        filled_mask = binary_fill_holes(cell_mask) # fill holes in the cell mask

        # compare original and filled masks - if equal, there are no holes
        contains_holes = not np.array_equal(cell_mask, filled_mask)
        return contains_holes

    ###### ACTUAL FUNCTIONS ###### ==> HAMILTONIAN & STEP METHODS
    
    def calculate_hamiltonian(self):
        hamiltonian = 0
        cell_ids = np.unique(self.grid)
        cell_ids = cell_ids[cell_ids != 0]
        
        for cell_id in cell_ids:
            # deltaH_ground: check for disjoint parts
            labeled_array, num_features = label(self.grid == cell_id)
            if num_features > 1:
               return np.inf  # positive infinity for disjoint parts
            
            #deltaH_ground: check for holes
            if self.cell_contains_holes(cell_id):
                return np.inf
            
            # calc area & perimeter
            area = np.sum(self.grid == cell_id)
            perimeter = self.calculate_perimeter(cell_id)

            # Energy terms for area and perimeter/area ratio
            hamiltonian += 0.2*np.abs(area - self.target_area) # deltaH_area
            hamiltonian += 0.8*(np.abs(((area**(1/2)) / perimeter) - self.target_ratio)) # deltaH_area/perimeter_ratio
            hamiltonian -= self.fraction_illuminated(cell_id)  # no specific deltaH term as outlined in JP, but deltaH_lum for now

        return hamiltonian
    """

    