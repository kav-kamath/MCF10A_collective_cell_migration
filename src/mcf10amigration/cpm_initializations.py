import random
import numpy as np
from .cpm import CPM

# all cpm initializations
# figure out a better (non-hard coded way) to create cell of approx. circular shape to start with

###### ALL INITILIZATIONS ######

## RANDOM ##
def initialize_cells_random(cpm: CPM): #choose cell centers randomly
    """
    Initialize cells by placing a center randomly on the grid, then filling in a roughly cirular shape around the center.
    Repeated for each center.
    
    Each cell is initially placed as an approximately circular cluster of pixels, but cells may overwrite previously placed cells they overlap with.  

    Parameters:
        cpm : CPM
    Returns:
        None (Updates the CPM grid and updates the number of cells in case it changed.)
    """
    
    cell_ids = range(1, cpm.num_cells + 1)

    # new implementation: randomly choose cell centers, can totally overwrite previous cell if a cell center is
    # repeatedly chosen, code should still work but num_cells value may be lower than highest cell ID
    for cell_id in cell_ids:
        y, x = random.randint(3, cpm.grid_size - 4), random.randint(3, cpm.grid_size - 4)
        # main square
        cpm.grid[y-2:y+3, x-2:x+3] = cell_id #[inclusive, exclusive)
        #sides
        cpm.grid[y-1:y+2, x-3] = cell_id # left
        cpm.grid[y-3, x-1:x+2] = cell_id # top
        cpm.grid[y-1:y+2, x+3] = cell_id # right
        cpm.grid[y+3, x-1:x+2] = cell_id # bottom

    #get new number of cells
    cpm.num_cells = np.unique(cpm.grid).size - 1 #-1 to account for background id of 0
    

    # old implementation: always places centers on whitespace but can get stuck in while loop if no whitespace left
    """
    for cell_id in cell_ids:
        while True:
            y, x = random.randint(3, self.grid_size - 4), random.randint(3, self.grid_size - 4)
            if self.grid[y, x] == 0:
                # main square
                self.grid[y-2:y+3, x-2:x+3] = cell_id #[inclusive, exclusive)
                #sides
                self.grid[y-1:y+2, x-3] = cell_id # left
                self.grid[y-3, x-1:x+2] = cell_id # top
                self.grid[y-1:y+2, x+3] = cell_id # right
                self.grid[y+3, x-1:x+2] = cell_id # bottom
                break
    """

## IDEAL ##
def initialize_cells_ideal(cpm: CPM): #choose cell centers such as to uniformly place cells across space
    """
    Initialize cells to be non-overlapping and uniformly spaced across the grid, with grid filled (whitespace allowed).

    Parameters:
        cpm : CPM
    Returns:
        None (Updates the CPM grid.)
    """
    
    cell_id = 1
    for y in range(3, cpm.grid_size - 3, 7):
        for x in range(3, cpm.grid_size - 3, 7):
            # main sqaure
            cpm.grid[y-2:y+3, x-2:x+3] = cell_id #[unclusive, exclusive]
            #sides
            cpm.grid[y-1:y+2, x-3] = cell_id # left
            cpm.grid[y-3, x-1:x+2] = cell_id # top
            cpm.grid[y-1:y+2, x+3] = cell_id # right
            cpm.grid[y+3, x-1:x+2] = cell_id # bottom

            cell_id += 1
    
    cpm.num_cells = cell_id - 1 #correct the number of cells from the default value to the actual number in the simulation

## SPACE_FILLING ##            
def initialize_cells_space_filling(cpm: CPM):
    """
    Initialize cells to be non-overlapping and uniformly spaced across the grid, with grid filled (whitespace NOT allowed).

    Starts with the ideal uniform placement of cells, then fills empty spaces by assigning each empty pixel the ID of a randomly chosen, directly neighboring cell.


    Parameters:
        cpm : CPM
    Returns:
        None (Updates the CPM grid.)
    """
    
    initialize_cells_ideal(cpm) # build upon above, so cells are vaguely circular
    
    # iterate through all spaces, looking for empty ones
    while 0 in cpm.grid:
        for y in range(0, cpm.grid_size):
            for x in range(0, cpm.grid_size):
                #if empty
                if cpm.grid[y, x] == 0:
                    new_cell_id = [0] # start empty list
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]: #look at IDs of all neighbors
                        nx, ny = (x + dx) , (y + dy)
                        if 0 <= nx < cpm.grid_size and 0 <= ny < cpm.grid_size: #if neighbor has valid indices
                                if cpm.grid[ny, nx] != 0:
                                    new_cell_id.append(cpm.grid[ny, nx])
                    random_id = random.choice(new_cell_id) #choose new ID randomly
                    cpm.grid[y, x] = random_id
    
    cpm.num_cells = (np.unique(cpm.grid).size - 1) # -1 to account for background id of 0

## VORONOI ##
def initialize_cells_voronoi(cpm: CPM):
    """
    Initialize cells by assigning pixels based on closest cell center (Voronoi tessellation).
    
    Cell centers are chosen either uniformly or randomly (currently hardcoded). Each pixel is assigned to the closest cell center.

    Parameters:
        cpm : CPM
    Returns:
        None (Updates the CPM grid.)
    """

    center_method = "random"

    if center_method == "uniform": # Generate cell centers with uniform spacing
        # Calculate spacing for uniform distribution
        num_rows = int(np.sqrt(cpm.num_cells))
        num_cols = int(np.ceil(cpm.num_cells / num_rows))
        spacing_y = int(cpm.grid_size / num_rows)  # Spacing in y direction
        spacing_x = int(cpm.grid_size / num_cols)  # Spacing in x direction

        # calulcate centers
        cell_centers = []
        cell_id = 1  # Start cell ID from 1

        for i in range(num_rows):
            for j in range(num_cols):
                y = spacing_y // 2 + i * spacing_y
                x = spacing_x // 2 + j * spacing_x
                cell_centers.append((y, x))
                cell_id += 1
                if cell_id > cpm.num_cells:
                    break  # stop if reached the desired number of cells
            if cell_id > cpm.num_cells:
                break

    if center_method == "random": #Randomly choose cell centers:
        cell_ids = range(1, cpm.num_cells + 1)
        cell_centers = []
        for cell_id in cell_ids:
            y, x = random.randint(0, cpm.grid_size - 1), random.randint(0, cpm.grid_size - 1) #random.randint [inclusive, inclusive]
            cell_centers.append((y, x))

    #Assign pixels to cell center
    for y in range(cpm.grid_size):
        for x in range(cpm.grid_size):
            min_dist = float('inf')
            closest_cell_id = 0
            for cell_id, (cy, cx) in enumerate(cell_centers, 1):  # Start cell_id from 1
                dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_cell_id = cell_id
            cpm.grid[y, x] = closest_cell_id

def initialize_cells_tissue_sparse(cpm: CPM):
    """
    Initialize cells by placing initializing with ideal logic (see above), but reducing grid space for cells by margin amount.

    Parameters:
        cpm : CPM
        margin : int or None, specified margin space
        tissue_size : int or None, pecified tissue size
    Returns:
        None (Updates the CPM grid.)
    """
    
    N = cpm.grid_size
    margin = cpm.margin
    tissue_size = cpm.tissue_size
    
    # checks
    if (margin is None and tissue_size is None) or (margin is not None and tissue_size is not None):
        raise ValueError("specify exactly ONE of: margin OR tissue_size in CPM object")

    # margin given, compute tissue_size
    if margin is not None:
        assert (2*margin < N), "margin too large, no space left for tissue, margin must be < (N/2)"
        tissue_size = N - 2 * margin

    elif tissue_size is not None:
        assert 0 < tissue_size <= N, "tissue_size incompatible. 0 < tissue_size <= N"
        margin = (N - tissue_size) // 2
        assert tissue_size + (2*margin) == N, (
            "error: tissue_size + 2*margin =/= grid_size"
        )
    
    # for type-checker
    assert margin is not None
    assert tissue_size is not None

    # fill in grid        
    cell_id = 1
    for y in range(3+margin, cpm.grid_size - 3 - margin, 7):
        for x in range(3+margin, cpm.grid_size - 3 - margin, 7):
            # main sqaure
            cpm.grid[y-2:y+3, x-2:x+3] = cell_id #[unclusive, exclusive]
            #sides
            cpm.grid[y-1:y+2, x-3] = cell_id # left
            cpm.grid[y-3, x-1:x+2] = cell_id # top
            cpm.grid[y-1:y+2, x+3] = cell_id # right
            cpm.grid[y+3, x-1:x+2] = cell_id # bottom

            cell_id += 1
    cpm.num_cells = cell_id - 1


def initialize_cells_tissue_dense(cpm: CPM):
    """
    Initialize cells by placing initializing with ideal logic (see above), but reducing grid space for cells by margin amount.

    Parameters:
        cpm : CPM
        margin : int or None, specified margin space
        tissue_size : int or None, pecified tissue size
    Returns:
        None (Updates the CPM grid.)
    """
    
    N = cpm.grid_size
    margin = cpm.margin
    tissue_size = cpm.tissue_size
    
    # checks
    if (margin is None and tissue_size is None) or (margin is not None and tissue_size is not None):
        raise ValueError("specify exactly ONE of: margin OR tissue_size in CPM object")

    # margin given, compute tissue_size
    if margin is not None:
        assert (2*margin < N), "margin too large, no space left for tissue, margin must be < (N/2)"
        tissue_size = N - 2 * margin

    elif tissue_size is not None:
        assert 0 < tissue_size <= N, "tissue_size incompatible. 0 < tissue_size <= N"
        margin = (N - tissue_size) // 2
        assert tissue_size + (2*margin) == N, (
            "error: tissue_size + 2*margin =/= grid_size"
        )
    
    # for type-checker
    assert margin is not None
    assert tissue_size is not None

    # fill in small grid        
    cell_id = 1
    for y in range(3+margin, N - 3 - margin, 7):
        for x in range(3+margin, N - 3 - margin, 7):
            # main sqaure
            cpm.grid[y-2:y+3, x-2:x+3] = cell_id #[unclusive, exclusive]
            #sides
            cpm.grid[y-1:y+2, x-3] = cell_id # left
            cpm.grid[y-3, x-1:x+2] = cell_id # top
            cpm.grid[y-1:y+2, x+3] = cell_id # right
            cpm.grid[y+3, x-1:x+2] = cell_id # bottom

            cell_id += 1
    cpm.num_cells = cell_id - 1
            
    # fill in empty space in small grid
    # iterate through all spaces in small grid, looking for empty ones
    while 0 in cpm.grid[margin:N-margin, margin:N-margin]:
        for y in range(margin, N-margin):
            for x in range(margin, N-margin):
                #if empty
                if cpm.grid[y, x] == 0:
                    new_cell_id = [0] # start empty list
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]: #look at IDs of all neighbors
                        nx, ny = (x + dx) , (y + dy)
                        if 0 <= nx < cpm.grid_size and 0 <= ny < cpm.grid_size: #if neighbor has valid indices
                                if cpm.grid[ny, nx] != 0:
                                    new_cell_id.append(cpm.grid[ny, nx])
                    random_id = random.choice(new_cell_id) #choose new ID randomly
                    cpm.grid[y, x] = random_id


## CUSTOM (customize cell centers) ##
def initialize_cells_custom_centers(cpm: CPM):        
    """
    Initialize cells with specified cell centers. Later listed cells may overlap with and overwrite formerly listed cells.

    Parameters:
        self : CPM
    Returns:
        None (Updates the CPM grid and num_cells in case it changed.)
    """

    if cpm.cell_centers is None:
        raise ValueError("please specify cell_centers")
    assert cpm.cell_centers is not None

    assert isinstance(cpm.cell_centers, list), "cell_centers argument must be a list"
    for index, coordinate in enumerate(cpm.cell_centers, 1):
        assert isinstance(coordinate, tuple), f"cell_center #{index} must be a tuple"
        assert all(isinstance(x, int) for x in coordinate), f"cell_center #{index} must be a tuple of 2 ints"
    
    for y, x in cpm.cell_centers:
        assert 0 <= y < cpm.grid_size and 0 <= x < cpm.grid_size, (
            f"Cell center {(y, x)} is out of bounds: must be within "
            f"(0 ≤ y < {cpm.grid_size}, 0 ≤ x < {cpm.grid_size})"
        )
    
    # repeatedly chosen, code should still work but num_cells value may be lower than highest cell ID
    for cell_id, (y,x) in enumerate(cpm.cell_centers, 1):
        
        # to account for cells that may be partially out of bounds
        y_min = max(y-2, 0)
        y_max = min(y+3, cpm.grid_size)
        x_min = max(x-2, 0)
        x_max = min(x+3, cpm.grid_size)
        
        # main square
        cpm.grid[y_min:y_max, x_min:x_max] = cell_id #[inclusive, exclusive)
        
        #sides
        x_left = max(x-3, 0)
        cpm.grid[max(y-1,0):min(y+2, cpm.grid_size), x_left] = cell_id
        
        x_right = min(x+3, cpm.grid_size-1)
        cpm.grid[max(y-1,0):min(y+2, cpm.grid_size), x_right] = cell_id
        
        y_top = max(y-3, 0)
        cpm.grid[y_top, max(x-1,0):min(x+2, cpm.grid_size)] = cell_id
        
        y_bottom = min(y+3, cpm.grid_size-1)
        cpm.grid[y_bottom, max(x-1,0):min(x+2, cpm.grid_size)] = cell_id

        #old, doesn't prevent wrapping:
        #cpm.grid[y-1:y+2, x-3] = cell_id # left
        #cpm.grid[y-1:y+2, x+3] = cell_id # right
        #cpm.grid[y-3, x-1:x+2] = cell_id # top
        #cpm.grid[y+3, x-1:x+2] = cell_id # bottom

    #get new number of cells
    cpm.num_cells = np.unique(cpm.grid).size - 1

## CUSTOM (customize entire grid) ##
def initialize_cells_custom_grid(cpm: CPM):
    """
    Initialize cells with hardcoded layout.

    Parameters:
        self : CPM
    Returns:
        None (Updates the CPM grid.)
    """
    if cpm.custom_grid is None:
        raise ValueError(f"please specify custom_grid argument")
    if not isinstance(cpm.custom_grid, np.ndarray):
        raise ValueError(f"custom_grid argument should be a numpy array of ints")
    if not np.issubdtype(cpm.custom_grid.dtype, np.integer):
        raise ValueError("all values in the custom_grid array must be integers")
    if cpm.custom_grid.shape[0] != cpm.custom_grid.shape[1]:
        raise ValueError(f"custom_grid array must be square, got shape {cpm.custom_grid.shape}")
    
    cpm.grid_size = cpm.custom_grid.shape[0]
    
    # can't directly assign self.grid (self.grid = np.array(custom_grid, dtype=int)) so need to overwrite matrix values instead
    cpm.grid[0:cpm.grid_size, 0:cpm.grid_size] = np.array(cpm.custom_grid, dtype=int)
    
    cpm.num_cells = np.unique(cpm.grid).size - 1
    
init_methods = {
    "random": initialize_cells_random,
    "ideal": initialize_cells_ideal,
    "space_filling": initialize_cells_space_filling,
    "voronoi": initialize_cells_voronoi,
    "tissue_sparse": initialize_cells_tissue_sparse,
    "tissue_dense": initialize_cells_tissue_dense,
    "custom1": initialize_cells_custom_centers,
    "custom2": initialize_cells_custom_grid
}