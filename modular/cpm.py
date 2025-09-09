import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from scipy.ndimage import label, binary_fill_holes
from skimage.measure import perimeter

# full prelim CPM (Hamiltonian with deltaH_area & deltaH_perimeter & prelim deltaH_lum)

class CPM:
    def __init__(self, grid_size, num_cells, target_area, target_ratio, temperature, initialization, light_pattern):
        self.grid_size = grid_size
        self.num_cells = num_cells
        self.target_area = target_area
        self.target_ratio = target_ratio
        self.temperature = temperature
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.mc_time = 0
        self.gill_time = 0.0
        
        # initialize light pattern
        if light_pattern is not None:
            light_pattern = np.array(light_pattern, dtype=int) # make a numpy array, necessary for masking later on
            assert light_pattern.shape == (grid_size, grid_size), "light_pattern must match grid size"
            assert np.all(np.isin(light_pattern, [0, 1])), "light_pattern must be binary (0s and 1s)"
            self.light_pattern = light_pattern
        else:
            self.light_pattern = np.zeros((grid_size, grid_size), dtype=int)  # default: all dark

        # initialize cells on grid (random, ideal, space_filling)
        if initialization == "random":
          self.initialize_cells_random()
        elif initialization == "ideal":
          self.initialize_cells_ideal()
        elif initialization == "space_filling":
          self.initialize_cells_space_filling()
        elif initialization == "voronoi":
          self.initialize_cells_voronoi()
        elif initialization == "custom1":
          self.initialize_cells_custom1()
        elif initialization == "custom2":
          self.initialize_cells_custom2()
        else:
          print("invalid initialization")

    # figure out a better (non-hard coded way) to create cell of approx. circular shape to start with

    ###### ALL INITILIZATIONS ######
    
    ## RANDOM ##
    def initialize_cells_random(self): #choose cell centers randomly
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
    def initialize_cells_ideal(self): #choose cell centers such as to uniformly place cells across space
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
    def initialize_cells_space_filling(self):
      self.initialize_cells_ideal() # build upon above, so cells are vaguely circular
      # iterate through all spaces, looking for empty ones
      while 0 in self.grid:
        for y in range(0, self.grid_size):
          for x in range(0, self.grid_size):
            #if empty
            if self.grid[y, x] == 0:
              new_cell_id = [0] # start empty lsit
              for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]: #look at IDs of all neighbors
                  nx, ny = (x + dx) , (y + dy)
                  if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size: #if neighbor has valid indices
                    if self.grid[ny, nx] != 0:
                      new_cell_id.append(self.grid[ny, nx])
              random_id = random.choice(new_cell_id) #choose new ID randomly
              self.grid[y, x] = random_id
    
    ## VORONOI ##
    def initialize_cells_voronoi(self):

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
    def initialize_cells_custom1(self):        
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
    def initialize_cells_custom2(self):
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

        self.grid = np.array(custom_grid, dtype=int)

        existing_cell_ids = np.unique(self.grid)
        existing_cell_ids = existing_cell_ids[existing_cell_ids != 0]
        self.num_cells = len(existing_cell_ids)
        


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


    def monte_carlo_step(self):
        for _ in range(self.grid_size**2):  # N random grid points
            i_x, i_y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            j_x, j_y = (i_x + dx), (i_y + dy)

            # if jx,jy is a valid grid point (no wrapping around)
            if (0 <= j_x < self.grid_size) & (0 <= j_y < self.grid_size):
                #if xi,xj and jx,jy have different cell IDs
                if (self.grid[i_y, i_x] != self.grid[j_y, j_x]):

                  #old hamiltonian with old j ID
                  old_j_value = self.grid[j_y, j_x]
                  old_hamiltonian = self.calculate_hamiltonian()

                  # change j to i, calculate new hamiltonian
                  self.grid[j_y, j_x] = self.grid[i_y, i_x]
                  new_hamiltonian = self.calculate_hamiltonian()

                  # deltaH
                  delta_hamiltonian = new_hamiltonian - old_hamiltonian

                  if (delta_hamiltonian <= 0) or (random.random() < np.exp(-delta_hamiltonian / self.temperature)):
                      pass  # accept j -> i
                  else:
                      self.grid[j_y, j_x] = old_j_value  # reject j -> i
        self.mc_time += 1 #increment time by 1 every time one full monte carlo step is complete (all N events have been attempted)
    
    def gillespie_step(self):
        events = []
        rates = []

        old_hamiltonian = self.calculate_hamiltonian()
        
        # all possible copy events and their rates (probability of occuring)
        for i_y in range(self.grid_size):
            for i_x in range(self.grid_size):
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    j_x, j_y = i_x + dx, i_y + dy
                    if 0 <= j_x < self.grid_size and 0 <= j_y < self.grid_size:
                        if self.grid[i_y, i_x] != self.grid[j_y, j_x]:
                            # calculate deltaH for this event
                            old_j_value = self.grid[j_y, j_x]
                            self.grid[j_y, j_x] = self.grid[i_y, i_x]
                            new_hamiltonian = self.calculate_hamiltonian()
                            self.grid[j_y, j_x] = old_j_value  # revert

                            deltaH = new_hamiltonian - old_hamiltonian
                            # rate: exp(-deltaH/T) if deltaH > 0, else 1
                            if not np.isnan(deltaH):
                                rate = np.exp(-deltaH / self.temperature) #1.0
                            else:
                                print("deltaH is nan")
                            events.append(((i_x, i_y), (j_x, j_y)))
                            rates.append(rate)
                            
        # cell empty evenst
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y, x] != 0:
                    original_id = self.grid[y, x]
                    self.grid[y, x] = 0
                    new_hamiltonian = self.calculate_hamiltonian()
                    self.grid[y, x] = original_id  # revert

                    deltaH = new_hamiltonian - old_hamiltonian
                    rate = np.exp(-deltaH / self.temperature) if deltaH > 0 else 1.0
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
            self.grid[y, x] = 0
        else:
            (i_x, i_y), (j_x, j_y) = chosen_event
            self.grid[j_y, j_x] = self.grid[i_y, i_x]
        

        # move forward in time, probability of any event occuring (like aggregated poisson)
        # inverse transform sampling method
        U = np.random.uniform() #choose random number from uniform dist [0, 1) 
        delta_t = -np.log(U) / total_rate # waiting time for next event is expential; adds up to poisson process over many events
        self.gill_time += delta_t
      