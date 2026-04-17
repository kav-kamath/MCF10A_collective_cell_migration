# python functions
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.measure import perimeter


# my files
#from .cpm_initializations import init_methods

# full prelim CPM (Hamiltonian with deltaH_area & deltaH_perimeter & prelim deltaH_lum)

allowed_light_functions = (
    "static_circle",
    "static_left",
    "static_right",
    "no_light",
    "spreading_from_corner",
    "shrinking_circle",
    "growing_circle",
    "outward_circle_wave",
    "multiple_outward_circle_waves",
    "inward_circle_wave",
    "multiple_inward_circle_waves",
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
    adhesion, and illumination-related energy terms. The constructor intializes the cell state 
    grid and the illumination grid, and holds model/simulation parameters.

    Parameters:
        grid_size : int - width and height of the square simulation grid.
        num_cells : int - number of cells to initialize on the grid
        target_area : float - ideal cell area to be used in the Hamiltonian energy calculations
        k : int - light responsiveness parameter
        lambda_area : float - area penalty parameter
        lambda_roundness : float - perimeter penalty parameter
        lambda_adhesion : int - (lack of) adhesion penalty parameter
        temperature : float - simulation temperature controlling stochastic acceptance of potential events
        initialization : str - name of method for initializing cells on the grid. Valid options are:
            "random", "ideal", "space_filling", "voronoi", "tissue_sparse", "tissue_dense", "wound", "custom_centers", "custom_grid"
        light_function : str - name of method for initializing illumination function over grid. Valid options are:
            "no_light", "static_circle", "static_left", "static_right", 
            "spreading_from_corner", "moving_bar", "multiple_moving_bars", "shrinking_circle", "growing_circle",
            "outward_circle_wave", "multiple_outward_circle_waves", "inward_circle_wave", "multiple_inward_circle_waves"         
        light_speed : float or None - speed of light movement in light function, used when light function is dynamic
        light_pattern : Numpy ndarray or None - starting binary 2D array indicating regions of illumination
        tissue_size : int or None - width of tissue for "tissue_sparse", "tissue_dense", and "wound" intializations
        margin : int or None - margin of empty space around tissue for "tissue_sparse", "tissue_dense", and "wound" intializations
        wound_size : int or None - radius of wound for "wound" initializations
        cell_centers : list of tuples or None - list of cell centers for "custom_centers" initialization
        custom_grid : Numpy ndarray or None - custom starting cell state grid for "custom_grid" initialization
        light_center : tuple of ints or None - center of illumination pattern for "static_circle", "shrinking_circle", "growing_circle", 
            "outward_circle_wave", "multiple_outward_circle_waves", "inward_circle_wave", and "multiple_inward_circle_waves" illumination functions
        light_radius : int or None - radius of (starting) illumination pattern for "static_circle", "shrinking_circle", "growing_circle", 
            "outward_circle_wave", and "inward_circle_wave" illumination functions
        light_boundary : int or None - illumination boundary for "static_left" and "static_right" illumination functions
        light_width : int or None - width of wave in "outward_circle_wave", "inward_circle_wave", and "moving_bar" illumination functions
        light_spatial_period : int or None - spatial period for "multiple_outward_circle_waves", "multiple_inward_circle_waves", 
            and "multiple_moving_bars" illumination functions
        light_duty_cycle : float or None - fraction of sptial period that is illuminated for "multiple_outward_circle_waves", 
            "multiple_inward_circle_waves", and "multiple_moving_bars" illumination functions
        
    Returns:
        None (Initializes the CPM grid and parameters.)
    """

    def __init__(
        self, 
        grid_size=50, 
        num_cells=1, 
        target_area=37, 
        k=0, # light-responsiveness parameter
        lambda_area = 1.0,
        lambda_roundness = 1.0,
        lambda_adhesion = 0,
        temperature = 1,
        initialization="initialize_cells_random", 
        light_function="no_light", 
        light_speed = None,
        light_pattern=None, 
        tissue_size = None, 
        margin = None,
        wound_size = None, 
        cell_centers = None,
        custom_grid = None,
        light_center = None,
        light_radius = None,
        light_boundary = None,
        light_width = None,
        light_spatial_period = None,
        light_duty_cycle = None
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
        self.light_center = light_center
        self.light_radius = light_radius
        self.light_boundary = light_boundary
        self.light_width = light_width
        self.light_spatial_period = light_spatial_period
        self.light_duty_cycle = light_duty_cycle
        
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