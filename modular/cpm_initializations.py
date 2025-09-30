import random
import numpy as np
from .cpm import CPM

# all cpm initializations
# figure out a better (non-hard coded way) to create cell of approx. circular shape to start with

###### ALL INITILIZATIONS ######

## RANDOM ##
def initialize_cells_random(self: CPM): #choose cell centers randomly
    
    """
    Initialize cells by placing a center randomly on the grid, then filling in a roughly cirular shape around the center.
    Repeated for each center.
    
    Each cell is initially placed as an approximately circular cluster of pixels, but cells may overwrite previously placed cells they overlap with.  

    Parameters:
        self : CPM
    Returns:
        None (Updates the CPM grid and updates the number of cells in case it changed.)
    """
    
    cell_ids = range(1, self.num_cells + 1)

    # new implementation: randomly choose cell centers, can totally overwrite previous cell if a cell center is
    # repeatedly chosen, code should still work but num_cells value may be lower than highest cell ID
    for cell_id in cell_ids:
        y, x = random.randint(3, self.grid_size - 4), random.randint(3, self.grid_size - 4)
        # main square
        self.grid[y-2:y+3, x-2:x+3] = cell_id #[inclusive, exclusive)
        #sides
        self.grid[y-1:y+2, x-3] = cell_id # left
        self.grid[y-3, x-1:x+2] = cell_id # top
        self.grid[y-1:y+2, x+3] = cell_id # right
        self.grid[y+3, x-1:x+2] = cell_id # bottom

    #get new number of cells
    existing_cell_ids = np.unique(self.grid)
    existing_cell_ids = existing_cell_ids[existing_cell_ids != 0]
    self.num_cells = len(existing_cell_ids)
    

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
def initialize_cells_ideal(self: CPM): #choose cell centers such as to uniformly place cells across space
        
    """
    Initialize cells to be non-overlapping and uniformly spaced across the grid, with grid filled (whitespace allowed).

    Parameters:
        self : CPM
    Returns:
        None (Updates the CPM grid.)
    """
    
    cell_id = 1
    for y in range(3, self.grid_size - 3, 7):
        for x in range(3, self.grid_size - 3, 7):
            # main sqaure
            self.grid[y-2:y+3, x-2:x+3] = cell_id #[unclusive, exclusive]
            #sides
            self.grid[y-1:y+2, x-3] = cell_id # left
            self.grid[y-3, x-1:x+2] = cell_id # top
            self.grid[y-1:y+2, x+3] = cell_id # right
            self.grid[y+3, x-1:x+2] = cell_id # bottom

            cell_id += 1

## SPACE_FILLING ##            
def initialize_cells_space_filling(self: CPM):
    
    """
    Initialize cells to be non-overlapping and uniformly spaced across the grid, with grid filled (whitespace NOT allowed).

    Starts with the ideal uniform placement of cells, then fills empty spaces by assigning each empty pixel the ID of a randomly chosen, directly neighboring cell.


    Parameters:
        self : CPM
    Returns:
        None (Updates the CPM grid.)
    """
    
    initialize_cells_ideal(self) # build upon above, so cells are vaguely circular
    
    # iterate through all spaces, looking for empty ones
    while 0 in self.grid:
        for y in range(0, self.grid_size):
            for x in range(0, self.grid_size):
                #if empty
                if self.grid[y, x] == 0:
                    new_cell_id = [0] # start empty list
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]: #look at IDs of all neighbors
                        nx, ny = (x + dx) , (y + dy)
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size: #if neighbor has valid indices
                                if self.grid[ny, nx] != 0:
                                    new_cell_id.append(self.grid[ny, nx])
                    random_id = random.choice(new_cell_id) #choose new ID randomly
                    self.grid[y, x] = random_id

## VORONOI ##
def initialize_cells_voronoi(self: CPM):

    """
    Initialize cells by assigning pixels based on closest cell center (Voronoi tessellation).
    
    Cell centers are chosen either uniformly or randomly (currently hardcoded). Each pixel is assigned to the closest cell center.

    Parameters:
        self : CPM
    Returns:
        None (Updates the CPM grid.)
    """

    center_method = "uniform"

    if center_method == "uniform": # Generate cell centers with uniform spacing
        # Calculate spacing for uniform distribution
        num_rows = int(np.sqrt(self.num_cells))
        num_cols = int(np.ceil(self.num_cells / num_rows))
        spacing_y = int(self.grid_size / num_rows)  # Spacing in y direction
        spacing_x = int(self.grid_size / num_cols)  # Spacing in x direction

        # calulcate centers
        cell_centers = []
        cell_id = 1  # Start cell ID from 1

        for i in range(num_rows):
            for j in range(num_cols):
                y = spacing_y // 2 + i * spacing_y
                x = spacing_x // 2 + j * spacing_x
                cell_centers.append((y, x))
                cell_id += 1
                if cell_id > self.num_cells:
                    break  # Stop if we've reached the desired number of cells
            if cell_id > self.num_cells:
                break

    if center_method == "random": #Randomly choose cell centers:
        cell_ids = range(1, self.num_cells + 1)
        cell_centers = []
        for cell_id in cell_ids:
            y, x = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1) #random.randint [inclusive, inclusive]
            cell_centers.append((y, x))

    #Assign pixels to cell center
    for y in range(self.grid_size):
        for x in range(self.grid_size):
            min_dist = float('inf')
            closest_cell_id = 0
            for cell_id, (cy, cx) in enumerate(cell_centers, 1):  # Start cell_id from 1
                dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_cell_id = cell_id
            self.grid[y, x] = closest_cell_id

## CUSTOM (customize cell centers) ##
def initialize_cells_custom1(self: CPM, cell_centers: list[tuple[int, int]]):        
    
    """
    Initialize cells with specified cell centers. Later listed cells may overlap with and overwrite formerly listed cells.

    Parameters:
        self : CPM
    Returns:
        None (Updates the CPM grid and num_cells in case it changed.)
    """
    
    assert len(cell_centers) == self.num_cells, (
        f"{self.num_cells} cell centers expected, {len(cell_centers)} cell centers input"
    )
    for y, x in cell_centers:
        assert 0 <= y < self.grid_size and 0 <= x < self.grid_size, (
            f"Cell center {(y, x)} is out of bounds: must be within "
            f"(4 ≤ y < {self.grid_size - 3}, 4 ≤ x < {self.grid_size - 3})"
        )
    
    cell_ids = range(1, self.num_cells + 1)
    
    # specify cell centers
    cell_centers = [(1,1), (8,1), (1,8), (8,8)]
    
    # repeatedly chosen, code should still work but num_cells value may be lower than highest cell ID
    for cell_id, (y,x) in enumerate(cell_centers, 1):
        # y, x = random.randint(3, self.grid_size - 4), random.randint(3, self.grid_size - 4)
        # main square
        self.grid[y-2:y+3, x-2:x+3] = cell_id #[inclusive, exclusive)
        #sides
        self.grid[y-1:y+2, x-3] = cell_id # left
        self.grid[y-3, x-1:x+2] = cell_id # top
        self.grid[y-1:y+2, x+3] = cell_id # right
        self.grid[y+3, x-1:x+2] = cell_id # bottom

    #get new number of cells
    existing_cell_ids = np.unique(self.grid)
    existing_cell_ids = existing_cell_ids[existing_cell_ids != 0]
    self.num_cells = len(existing_cell_ids)

## CUSTOM (customize entire grid) ##
def initialize_cells_custom2(self: CPM):
    
    """
    Initialize cells with hardcoded layout.

    Parameters:
        self : CPM
    Returns:
        None (Updates the CPM grid.)
    """
    
    custom_grid = [
        [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 3, 3, 3, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    
    # can't directly assign self.grid (self.grid = np.array(custom_grid, dtype=int)) so need to overwrite matrix values instead
    self.grid[0:self.grid_size, 0:self.grid_size] = np.array(custom_grid, dtype=int)