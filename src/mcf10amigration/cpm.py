# python functions
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.measure import perimeter


# my files
#from .cpm_initializations import init_methods

# full prelim CPM (Hamiltonian with deltaH_area & deltaH_perimeter & prelim deltaH_lum)

allowed_light_functions = (
    "static_circle",
    "static_left_half",
    "static_right_half",
    "no_light",
    "spreading_from_corner",
    "shrinking_circle",
    "growing_circle",
    "outward_circle_wave",
    "inward_circle_wave",
    "moving_bar",
    "multiple_moving_bars"
)

allowed_initializations = (
    "random",
    "ideal",
    "space_filling",
    "voronoi",
    "tissue_sparse",
    "tissue_dense",
    "wound",
    "custom_centers",
    "custom_grid"
)

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

    def __init__(
        self, 
        grid_size=50, 
        num_cells=1, 
        target_area=37, 
        k=0, # light-responsiveness parameter
        lambda_area = 1,
        lambda_roundness = 1,
        lambda_adhesion = 0,
        temperature = 1,
        initialization="initialize_cells_random", 
        light_function="no_light", 
        light_speed = 0.0,
        light_pattern=None, 
        tissue_size = None, 
        margin = None,
        wound_size = None, 
        cell_centers = None,
        custom_grid = None
    ):
        
        if initialization not in allowed_initializations:
            raise ValueError(f"Invalid initialization function: {initialization}. Must be one of {allowed_initializations}")
        assert initialization in allowed_initializations, f"invalid initialization function: {initialization}. must be one of {allowed_initializations}"

        if light_function not in allowed_light_functions:
            raise ValueError(f"Invalid light function: {light_function}. Must be one of {allowed_light_functions}")
        assert light_function in allowed_light_functions, f"invalid light function: {light_function}. must be one of {allowed_light_functions}"
        
        assert (temperature>0), "t should be > 0 for cells attracted to light"
        
        self.grid_size = grid_size
        self.num_cells = num_cells
        self.target_area = target_area
        self.k = k
        self.temperature = temperature
        self.lambda_area = lambda_area
        self.lambda_roundness = lambda_roundness
        self.lambda_adhesion = lambda_adhesion
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.initialization = initialization
        self.light_function = light_function
        self.light_speed = light_speed
        self.tissue_size = tissue_size
        self.margin = margin
        self.wound_size = wound_size
        self.cell_centers = cell_centers
        self.custom_grid = custom_grid
        
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

        #if initialization in init_methods:
            #init_methods[initialization](self)
        #    pass
        #else:
        #    raise ValueError(f"Invalid initialization method: {initialization}")

        #existing_cell_ids = np.unique(self.grid)
        #existing_cell_ids = existing_cell_ids[existing_cell_ids != 0]
        #self.num_cells = len(existing_cell_ids)