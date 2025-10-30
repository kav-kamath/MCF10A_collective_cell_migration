import numpy as np
from scipy.ndimage import label, binary_fill_holes
from skimage.measure import perimeter, perimeter_crofton, regionprops
from modular.cpm import CPM

###### HELPER FUNCTIONS ###### ==> FEATURES OF CELL

# updated to be skimage.measure.regionprops() perimeter
def _calculate_perimeter(cpm: CPM, cell_id):

    # skimage.measure.regionprops() perimeter
    binary_grid = (cpm.grid == cell_id)
    perimeter_value = perimeter_crofton(binary_grid, directions=4)

    return perimeter_value

def _fraction_illuminated(cpm: CPM, cell_id):
    
    props = regionprops(label_image=cpm.grid, intensity_image=cpm.light_pattern)
    region = next((r for r in props if r.label == cell_id), None)
    
    if region is not None:
        return region.mean_intensity
    else:
        return 0.0
    
    # old implementation:
    """
    cell_mask = (cpm.grid == cell_id) # t/f mask of cell location
    light_mask = (cpm.light_pattern == 1) # t/f mask of light location
        
    overlap = cell_mask & light_mask # AND of both masks, where both true

    area_in_light = np.sum(overlap)
    total_area = np.sum(cell_mask)
        
    if total_area == 0:
        return 0.0
    return area_in_light / total_area
    """

def _cell_contains_holes(cpm: CPM, cell_id):
    
    cell_mask = (cpm.grid == cell_id).astype(np.uint8)  # binary mask for the cell
    
    # check that the cell has only one connected component
    labeled_array, num_features = label(cell_mask) # possibly: intepreter misreading the type of self.grid, totally fine at runtime
    assert num_features == 1, f"Expected exactly 1 connnected component, found {num_features}"

    # get region properties (of interest: euler number)
    props = regionprops(cell_mask)
    region = props[0]  # only one region
    
    return region.euler_number != 1 # if no holes, euler_number = 1 

    # old implementation
    """    
    cell_mask = (cpm.grid == cell_id) # binary mask for the cell
    filled_mask = binary_fill_holes(cell_mask) # fill holes in the cell mask

    # compare original and filled masks - if equal, there are no holes
    contains_holes = not np.array_equal(cell_mask, filled_mask) # intepreter misreading the type of filled_mask, totally fine at runtime
    return contains_holes
    """

###### HAMILTONIAN FUNCTION ###### ==> DIFFERENT HAMILTONIANS
    
    
def calculate_hamiltonian(cpm: CPM):
    """
    Compute the total Hamiltonian energy of the current CPM grid state.

    This Hamiltonian includes the following energy contributions for each cell:
    - Area deviation from the target area.
    - Deviation of the perimeter-to-area ratio from the target ratio.
    - Light consideration; energy of the cell is reduced if in an illuminated regions.

    Parameters
        cpm : CPM - CPM object as defined in cpm.py

    Returns
        hamiltonian : float - computed Hamiltonian energy value
            Returns np.inf if a cell is disjoint or contains holes.
    """
    
    hamiltonian = 0
    cell_ids = np.unique(cpm.grid)
    cell_ids = cell_ids[cell_ids != 0]
        
    for cell_id in cell_ids:
        # deltaH_ground: check for disjoint parts
        labeled_array, num_features = label(cpm.grid == cell_id) # intepreter misreading the type of self.grid, totally fine at runtime
        if num_features > 1:
            return np.inf  # positive infinity for disjoint parts
            
        #deltaH_ground: check for holes
        if _cell_contains_holes(cpm, cell_id):
            return np.inf
            
        # calc area & perimeter
        area = np.sum(cpm.grid == cell_id)
        perimeter = _calculate_perimeter(cpm, cell_id)

        # Energy terms for area and perimeter/area ratio
        hamiltonian += 10*np.power(np.abs(area - cpm.target_area), 2) # deltaH_area
        hamiltonian += np.power(np.abs(perimeter-cpm.target_perimeter), 4) # deltaH_perimeter
        #hamiltonian += 0.8*(np.abs(((area**(1/2)) / perimeter) - cpm.target_ratio)) # deltaH_area/perimeter_ratio
        hamiltonian -= np.power(100*_fraction_illuminated(cpm, cell_id), 1.75) # no specific deltaH term as outlined in JP, but deltaH_lum for now
        
        # print statements
        #print("Cell ID: ", cell_id)
        #print("area delta: ", (area - cpm.target_area))
        #print("perimeter, perimeter delta: ", perimeter, (perimeter - cpm.target_perimeter))
        

    return hamiltonian

    