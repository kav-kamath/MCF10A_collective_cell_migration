import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.measure import perimeter
from .cpm import CPM


###### HELPER FUNCTIONS ###### ==> FEATURES OF CELL

# updated to be skimage.measure.regionprops() perimeter
def _calculate_perimeter(self: CPM, cell_id):

    # skimage.measure.regionprops() perimeter
    binary_grid = (self.grid == cell_id)
    perimeter_value = perimeter(binary_grid, neighborhood = 8)

    return perimeter_value

def _fraction_illuminated(self: CPM, cell_id):

    cell_mask = (self.grid == cell_id) # t/f mask of cell location
    light_mask = (self.light_pattern == 1) # t/f mask of light location
        
    overlap = cell_mask & light_mask # AND of both masks, where both true

    area_in_light = np.sum(overlap)
    total_area = np.sum(cell_mask)
        
    if total_area == 0:
        return 0.0
    return area_in_light / total_area

def _cell_contains_holes(self: CPM, cell_id):
        
    cell_mask = (self.grid == cell_id) # binary mask for the cell
    filled_mask = binary_fill_holes(cell_mask) # fill holes in the cell mask

    # compare original and filled masks - if equal, there are no holes
    contains_holes = not np.array_equal(cell_mask, filled_mask) # intepreter misreading the type of filled_mask, totally fine at runtime
    return contains_holes



###### HAMILTONIAN FUNCTION ###### ==> DIFFERENT HAMILTONIANS
    
def calculate_hamiltonian(self: CPM):
    """
    Compute the total Hamiltonian energy of the current CPM grid state.

    This Hamiltonian includes the following energy contributions for each cell:
    - Area deviation from the target area.
    - Deviation of the perimeter-to-area ratio from the target ratio.
    - Light consideration; energy of the cell is reduced if in an illuminated regions.

    Parameters
        self : CPM - CPM object as defined in cpm.py

    Returns
        hamiltonian : float - computed Hamiltonian energy value
            Returns np.inf if a cell is disjoint or contains holes.
    """
    
    hamiltonian = 0
    cell_ids = np.unique(self.grid)
    cell_ids = cell_ids[cell_ids != 0]
        
    for cell_id in cell_ids:
        # deltaH_ground: check for disjoint parts
        labeled_array, num_features = label(self.grid == cell_id) # intepreter misreading the type of self.grid, totally fine at runtime
        if num_features > 1:
            return np.inf  # positive infinity for disjoint parts
            
        #deltaH_ground: check for holes
        if _cell_contains_holes(self, cell_id):
            return np.inf
            
        # calc area & perimeter
        area = np.sum(self.grid == cell_id)
        perimeter = _calculate_perimeter(self, cell_id)

        # Energy terms for area and perimeter/area ratio
        hamiltonian += 0.2*np.abs(area - self.target_area) # deltaH_area
        hamiltonian += 0.8*(np.abs(((area**(1/2)) / perimeter) - self.target_ratio)) # deltaH_area/perimeter_ratio
        hamiltonian -= _fraction_illuminated(self, cell_id)  # no specific deltaH term as outlined in JP, but deltaH_lum for now

    return hamiltonian

    