import random
import numpy as np
from .cpm import CPM

def radial_density(frame: np.ndarray, bin_width:int = 1):
        
    l, w = frame.shape
    cy = l/2
    cx = w/2
    
    # coordinate arrays
    y_coords, x_coords = np.indices(frame.shape)
    dist_array = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2) #euclidian distance
    
    # bins
    r_max = dist_array.max()
    bin_edges = np.arange(0, r_max + bin_width, bin_width)

    density = []
    bin_centers = []

    for r0, r1 in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (dist_array >= r0) & (dist_array < r1)

        labels_in_ring = frame[mask]

        # count unique non-zero cell IDs
        unique_cells = np.unique(labels_in_ring)
        unique_cells = unique_cells[unique_cells != 0] # remove 0

        density.append(len(unique_cells))
        bin_centers.append((r0 + r1) / 2)

    return [bin_centers, bin_edges, density]

def inside_circle_count(frame, cy, cx, radius):

    n = frame.shape[0]

    # bounding box to prevent wrapping
    y_min = max(0, cy-radius)
    y_max = min(n, cy+radius)
    x_min = max(0, cx-radius)
    x_max = min(n, cx+radius)
    
    # local region
    subset = frame[y_min:y_max, x_min:x_max]
    # local coordinates
    y_coords, x_coords = np.indices(subset.shape)
    # convert to global coordinates
    y_coords = y_coords + y_min
    x_coords = x_coords + x_min

    # mask of if inside circle
    inside_mask = (y_coords - cy)**2 + (x_coords - cx)**2 <= radius**2 #t/f boolean mask
    values_inside = subset[inside_mask] 

    # Count unique non-zero labels
    unique_cells = np.unique(values_inside)
    unique_cells = unique_cells[unique_cells != 0]

    return len(unique_cells)
    
    
def inside_square_count(frame, cy, cx, width):

    n = frame.shape[0]

    # bounding box to prevent wrapping
    y_min = max(0, cy-width)
    y_max = min(n, cy+width)
    x_min = max(0, cx-width)
    x_max = min(n, cx+width)
    
    # local region
    values_inside = frame[y_min:y_max, x_min:x_max]
    
    # Count unique non-zero labels
    unique_cells = np.unique(values_inside)
    unique_cells = unique_cells[unique_cells != 0]

    return len(unique_cells)
    