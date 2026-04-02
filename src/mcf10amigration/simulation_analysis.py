import random
import numpy as np
from .cpm import CPM
from skimage.measure import regionprops, regionprops_table
import matplotlib.pyplot as plt
from matplotlib import colors

def radial_density(frame: np.ndarray, bin_width:int = 1, method:str = "cell_area"):
        
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
        ring_area = mask.sum()
        inside_ring = frame[mask]

        if method == "cell_count":
            # count unique non-zero cell IDs
            unique_cells = np.unique(inside_ring)
            unique_cells = unique_cells[unique_cells != 0] # remove 0
            return_val = len(unique_cells)/ring_area
        elif method == "cell_area":
            return_val = np.count_nonzero(inside_ring)/ring_area
        elif method == "raw_count":
            return_val = len(unique_cells)
        else:
            raise ValueError("method argument must be 'cell_count' 'cell_area' or 'raw_count'")

        density.append(return_val)
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

    # count unique non-zero labels
    unique_cells = np.unique(values_inside)
    unique_cells = unique_cells[unique_cells != 0]

    return len(unique_cells)
    
    
def inside_square_count(frame, cy, cx, halfwidth):

    n = frame.shape[0]

    # bounding box to prevent wrapping
    y_min = max(0, cy-halfwidth)
    y_max = min(n, cy+halfwidth)
    x_min = max(0, cx-halfwidth)
    x_max = min(n, cx+halfwidth)
    
    # local region
    values_inside = frame[y_min:y_max, x_min:x_max]
    
    # Count unique non-zero labels
    unique_cells = np.unique(values_inside)
    unique_cells = unique_cells[unique_cells != 0]

    return len(unique_cells)
    
def avg_displacement(start_frame, end_frame, direction=None):

    centroids_start = regionprops_table(label_image=start_frame, properties=['centroid'])
    centroids_start = np.column_stack((centroids_start['centroid-0'], centroids_start['centroid-1']))
    
    centroids_end = regionprops_table(label_image=end_frame, properties=['centroid'])
    centroids_end = np.column_stack((centroids_end['centroid-0'], centroids_end['centroid-1']))
    
    twoD_displacements = centroids_end - centroids_start
    
    if direction == "both":
        return np.linalg.norm(twoD_displacements, axis=1) # gives disp of each cell
    elif direction == "x":
        return np.mean(twoD_displacements[:,1]) # gets avg x disp
    elif direction == "y":
        return np.mean(twoD_displacements[:,0]) # gets avg y disp
    else:
        return centroids_end, centroids_start, twoD_displacements

def avg_distance_from_point(frames, point):
    assert(type(point) == tuple) # should be (y,x) / (row,col)
    
    if(type(frames) == np.ndarray): # if given one frame
        
        centroids_sep = regionprops_table(frames, properties=['centroid'])

        centroids = np.column_stack((
            centroids_sep['centroid-0'],   # y coordinate
            centroids_sep['centroid-1']    # x coordinate
        ))

        distances = np.linalg.norm(centroids - np.array(point), axis=1)

        return np.mean(distances)

    if (type(frames) == list): # if given list of frames
        all_distances = []
        
        for frame in frames:
            centroids_sep = regionprops_table(frame, properties=['centroid'])

            centroids = np.column_stack((
                centroids_sep['centroid-0'],   # y coordinate
                centroids_sep['centroid-1']    # x coordinate
            ))

            distances = np.linalg.norm(centroids - np.array(point), axis=1)
            all_distances.append(np.mean(distances))
        
        return all_distances
        

def visualize_displacement(start_frame, end_frame, title="individual displacement", output_filename="fig.png", save_fig=True):
    
    centroids_start = regionprops_table(label_image=start_frame, properties=['centroid'])
    centroids_start = np.column_stack((centroids_start['centroid-0'], centroids_start['centroid-1']))
    
    centroids_end = regionprops_table(label_image=end_frame, properties=['centroid'])
    centroids_end = np.column_stack((centroids_end['centroid-0'], centroids_end['centroid-1']))
    
    individual_displacements = centroids_end - centroids_start
    
    # centroid are (row,column) which is like (y,x)
    y0 = centroids_start[:, 0]
    x0 = centroids_start[:, 1]
    
    dy = individual_displacements[:, 0]
    dx = individual_displacements[:, 1]

    # create a viridis colormap with 0 mapped to white
    viridis = plt.colormaps['viridis']
    newcolors = viridis(np.linspace(0,1,256))
    newcolors[0] = np.array([1,1,1,1])  # first color (lowest) is white
    viridis_white0 = colors.ListedColormap(newcolors)

    plt.figure()
    plt.imshow(start_frame, cmap=viridis_white0)

    plt.quiver(
        x0, y0,
        dx, dy,
        angles='xy',
        scale_units='xy',
        scale=1,
        color='red'
    )

    plt.title(title)
    if save_fig:
        plt.savefig(output_filename, dpi=600)
    plt.show()
    
def cosine_similarity(start_frame, end_frame, target_point):
    
    centroids_start = regionprops_table(label_image=start_frame, properties=['centroid'])
    centroids_start = np.column_stack((centroids_start['centroid-0'], centroids_start['centroid-1']))
    
    centroids_end = regionprops_table(label_image=end_frame, properties=['centroid'])
    centroids_end = np.column_stack((centroids_end['centroid-0'], centroids_end['centroid-1']))
    
    target_point = np.array(target_point)
    expected_displacements_direction = target_point - centroids_start
    individual_displacements = centroids_end - centroids_start
    
    # cos(θ) = (v dot u) / (||v|| * ||u||)  
    v_dot_u = np.sum(expected_displacements_direction * individual_displacements, axis=1)
    v_norm = np.linalg.norm(expected_displacements_direction, axis=1)
    u_norm = np.linalg.norm(individual_displacements, axis=1)
    cos_theta_vector = v_dot_u / (v_norm*u_norm)
    
    return cos_theta_vector, u_norm