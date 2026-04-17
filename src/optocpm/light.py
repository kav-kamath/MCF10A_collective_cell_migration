import numpy as np
from .cpm import CPM

# STATIC LIGHT PATTERN FUNCTIONS

def static_circle_light(y, x, t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_center : tuple of ints - center of static light circle in (y,x) or (row,col) format
        light_radius : int - radius of static circle light
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_center is not None), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[0]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[1]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    
    assert (cpm.light_radius is not None), "Please specify an light_radius value"
    assert (type(cpm.light_radius) is int), "Please specify a light_radius value as an integer"
    
    center = cpm.light_center
    radius = cpm.light_radius
    return ((y - center[0])**2 + (x - center[1])**2) <= radius**2

def static_left_light(y, x, t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_boundary : int or float - all grid columns including and to the left of this value will be illuminated
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_boundary is not None), "Please specify a light_boundary value"
    
    return x <= cpm.light_boundary

def static_right_light(y, x, t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_boundary : int or float - all grid columns including and to the right of this value will be illuminated
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_boundary is not None), "Please specify a light_boundary value"
    
    return x >= cpm.light_boundary

def no_light (y, x, t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        None
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    return x*0

# DYNAMIC LIGHT PATTERN FUNCTIONS

def light_spreading_from_corner(y, x, t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_speed : float - speed with which illuminated region expands, implemented as light_speed*t
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_speed is not None), "Please specify a light_speed value"
    
    return ((y+x) <= cpm.light_speed*t)

def shrinking_circle_light(y, x, t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_center : tuple of ints - center of initial light circle in (y,x) or (row,col) format
        light_radius : int - radius of intial circle light
        light_speed : float - speed with which illuminated region shrinks, implemented as light_speed*t
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_center is not None), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[0]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[1]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    
    assert (cpm.light_radius is not None), "Please specify an initial light_radius value"
    assert (type(cpm.light_radius) is int), "Please specify a initial light_radius value as an integer"
    
    assert (cpm.light_speed is not None), "Please specify a light_speed value"
    
    center = cpm.light_center
    radius = cpm.light_radius - cpm.light_speed*t
    if (radius >= 0):
        return ((y - center[0])**2 + (x - center[1])**2) <= radius**2
    else:
        return x*0

def growing_circle_light(y, x, t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_center : tuple of ints - center of initial light circle in (y,x) or (row,col) format
        light_radius : int - radius of intial circle light
        light_speed : float - speed with which illuminated region expands, implemented as light_speed*t
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_center is not None), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[0]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[1]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    
    assert (cpm.light_radius is not None), "Please specify an initial light_radius value"
    assert (type(cpm.light_radius) is int), "Please specify a initial light_radius value as an integer"
    
    assert (cpm.light_speed is not None), "Please specify a light_speed value"
    
    center = cpm.light_center
    radius = cpm.light_radius + cpm.light_speed*t
    return ((y - center[0])**2 + (x - center[1])**2) <= radius**2

def outward_circle_wave_light(y, x, t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_center : tuple of ints - center of initial circular band of light in (y,x) or (row,col) format
        light_radius : int - inner radius of intial circlular band of light
        light_width : int - width of light wave, implemented as light_radius+width
        light_speed : float - speed with which illuminated region expands, implemented as light_speed*t
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_center is not None), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[0]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[1]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    
    assert (cpm.light_radius is not None), "Please specify an initial inner light_radius value"
    assert (type(cpm.light_radius) is int), "Please specify a initial inner light_radius value as an integer"
    
    assert (cpm.light_width is not None), "Please specify a light_width value"
    assert (type(cpm.light_width) is int), "Please specify a light_width value as an integer"
    
    assert (cpm.light_speed is not None), "Please specify a light_speed value"
    
    center = cpm.light_center
    initial_inner_radius = cpm.light_radius
    initial_outer_radius = initial_inner_radius + cpm.light_width
    
    max_radius = max(np.sqrt((0 - center[0])**2 + (0 - center[1])**2),                          # dist from center to top left
                     np.sqrt((0 - center[0])**2 + (x.shape[1]-1 - center[1])**2),               # dist from center to top right
                     np.sqrt((y.shape[0]-1 - center[0])**2 + (0 - center[1])**2),               # dist from center to bottom left
                     np.sqrt((y.shape[0]-1 - center[0])**2 + (x.shape[1]-1 - center[1])**2))    # dist from center to bottom right
    
    inner_radius = ((initial_inner_radius + cpm.light_speed*t) % max_radius)
    outer_radius = ((initial_outer_radius + cpm.light_speed*t) % max_radius)
    
    return (((y - center[0])**2 + (x - center[1])**2) >= inner_radius**2) & (((y - center[0])**2 + (x - center[1])**2) <= outer_radius**2)
    

def multiple_outward_circle_waves_light(y, x, t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_center : tuple of ints - center of initial circular band of light in (y,x) or (row,col) format
        light_spatial_period : int - spatial period of illumination pattern
        light_duty_cycle : float - proportion of spatial period that is illuminated
        light_speed : float - speed with which illuminated region expands, implemented as light_speed*t
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_center is not None), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[0]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[1]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    
    assert (cpm.light_spatial_period is not None), "Please specify a light_spatial_period value"
    assert (type(cpm.light_spatial_period) is int), "Please specify an integer light_spatial_period value"
    
    assert (cpm.light_duty_cycle is not None), "Please specify a light_duty_cycle value"   
    assert ((type(cpm.light_duty_cycle) is float) and (cpm.light_duty_cycle >= 0.0) and (cpm.light_duty_cycle <= 1.0)), "light_duty_cycle value should be a float between 0.0 and 1.0"
    assert (cpm.light_speed is not None), "Please specify a light_speed value"
    
    center = cpm.light_center
    spatial_period = cpm.light_spatial_period                  # width + distance between waves
    duty_cycle = cpm.light_duty_cycle                          # width / spatial_period (fraction of period that is lit up)
    width = spatial_period * duty_cycle
    
    r = np.sqrt((y - center[0])**2 + (x - center[1])**2)       # radial distance from center
    wave_coord = (r - (cpm.light_speed * t)) % spatial_period     # moving wave coordinate
    light_mask = wave_coord < width                            # light where inside wave band

    return light_mask
    
    
def inward_circle_wave_light(y, x, t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_center : tuple of ints - center of initial circular band of light in (y,x) or (row,col) format
        light_radius : int - outer radius of intial circlular band of light
        light_width : int - width of light wave, implemented as light_radius-width
        light_speed : float - speed with which illuminated region shrinks, implemented as light_speed*t
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_center is not None), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[0]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[1]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    
    assert (cpm.light_radius is not None), "Please specify an initial outer light_radius value"
    assert (type(cpm.light_radius) is int), "Please specify a initial outer light_radius value as an integer"
    
    assert (cpm.light_width is not None), "Please specify a light_width value"
    assert (type(cpm.light_width) is int), "Please specify a light_width value as an integer"
    
    assert (cpm.light_speed is not None), "Please specify a light_speed value"
    
    center = cpm.light_center
    initial_outer_radius = cpm.light_radius
    initial_inner_radius = initial_outer_radius - cpm.light_width
    
    max_radius = max(np.sqrt((0 - center[0])**2 + (0 - center[1])**2),                          # dist from center to top left
                     np.sqrt((0 - center[0])**2 + (x.shape[1]-1 - center[1])**2),               # dist from center to top right
                     np.sqrt((y.shape[0]-1 - center[0])**2 + (0 - center[1])**2),               # dist from center to bottom left
                     np.sqrt((y.shape[0]-1 - center[0])**2 + (x.shape[1]-1 - center[1])**2))    # dist from center to bottom right
    
    inner_radius = ((initial_inner_radius - cpm.light_speed*t) % max_radius)
    outer_radius = ((initial_outer_radius - cpm.light_speed*t) % max_radius)
    
    return (((y - center[0])**2 + (x - center[1])**2) >= inner_radius**2) & (((y - center[0])**2 + (x - center[1])**2) <= outer_radius**2)
    
    
def multiple_inward_circle_waves_light(y, x, t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_center : tuple of ints - center of initial circular band of light in (y,x) or (row,col) format
        light_spatial_period : int - spatial period of illumination pattern
        light_duty_cycle : float - proportion of spatial period that is illuminated
        light_speed : float - speed with which illuminated region shrinks, implemented as light_speed*t
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_center is not None), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[0]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    assert (type(cpm.light_center[1]) is int), "Please specify a light_center value as an integer tuple of format (row,column)"
    
    assert (cpm.light_spatial_period is not None), "Please specify a light_spatial_period value"
    assert (type(cpm.light_spatial_period) is int), "Please specify an integer light_spatial_period value"
    
    assert (cpm.light_duty_cycle is not None), "Please specify a light_duty_cycle value"    
    assert ((type(cpm.light_duty_cycle) is float) and (cpm.light_duty_cycle >= 0.0) and (cpm.light_duty_cycle <= 1.0)), "light_duty_cycle value should be a float between 0.0 and 1.0"
    assert (cpm.light_speed is not None), "Please specify a light_speed value"
    
    center = cpm.light_center
    spatial_period = cpm.light_spatial_period                  # width + distance between waves
    duty_cycle = cpm.light_duty_cycle                          # width / spatial_period (fraction of period that is lit up)
    width = spatial_period * duty_cycle
    
    r = np.sqrt((y - center[0])**2 + (x - center[1])**2)       # radial distance from center
    wave_coord = (r + (cpm.light_speed * t)) % spatial_period     # moving wave coordinate
    light_mask = wave_coord < width                            # light where inside wave band

    return light_mask

def moving_bar_light(y, x , t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_width : int - width of moving bar
        light_speed : float - speed of moving bar, implemented as light_speed*t
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_width is not None), "Please specify a light_width value"
    assert (cpm.light_speed is not None), "Please specify a light_speed value"
    
    width = cpm.light_width

    top = (int(t * cpm.light_speed)) % y.shape[0]
    bottom = (top + width) % y.shape[0]

    if top < bottom:
        return (y >= top) & (y <= bottom)
    else: # wrap around (bottom < top)
        return (y >= top) | (y <= bottom)

def multiple_moving_bars_light(y, x , t, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        y : np.ndarray - grid of y coordinates
        x : np.ndarray - grid of x coordinates
        t : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Parameters cpm must have:
        light_spatial_period : int - spatial period of illumination pattern
        light_duty_cycle : float - proprotion of spatial period that is illuminated
        light_speed : float - speed of moving bars, implemented as light_speed*t
    Returns:
        boolean np.ndarray with same dimensions as CPM grid where True indicates illminated regions and False indicated unlit regions
    """
    
    assert (cpm.light_spatial_period is not None), "Please specify a light_spatial_period value"
    assert (type(cpm.light_spatial_period) is int), "Please specify an integer light_spatial_period value"
    
    assert (cpm.light_duty_cycle is not None), "Please specify a light_duty_cycle value"
    assert ((type(cpm.light_duty_cycle) is float) and (cpm.light_duty_cycle >= 0.0) and (cpm.light_duty_cycle <= 1.0)), "light_duty_cycle value should be a float between 0.0 and 1.0"    
    assert (cpm.light_speed is not None), "Please specify a light_speed value"

    spatial_period = cpm.light_spatial_period                      # width + distance_between_bars
    duty_cycle = cpm.light_duty_cycle                              # width / spatial_period (fraction of period that is lit up)
    width = int(spatial_period * duty_cycle)
    
    y_shifted = (y - (t*cpm.light_speed)) % spatial_period # modulo over sptial period will give where y is in respect to spatial period

    light_mask = y_shifted < width # light where inside the bar width

    return light_mask
    
    # old implementation
    """
    num_bars = int(y.shape[0] / spatial_period)
    
    # mask for y true(1)/false(0)
    light_mask = np.zeros(y.shape, dtype=bool)
    
    for i in range(num_bars):
        top = int((t*cpm.light_speed + i*spatial_period) % y.shape[0]) # start taking remainder once top > size of grid -> creates wrap around
        bottom = int((top + width) % y.shape[0])
    
        if bottom < top: # specific case when wrapping around when top is at bottom of grid & bottom is at top
            light_mask |= (y >= top) | (y <= bottom)
        else: # normal case
            light_mask |= (y >= top) & (y <= bottom)
    
    return light_mask
    """
    

# LIGHTING UPDATING FUNCTION

light_methods = {
    # Static light patterns
    "static_circle": static_circle_light,
    "static_left": static_left_light,
    "static_right": static_right_light,
    "no_light": no_light,

    # Dynamic light patterns
    "spreading_from_corner": light_spreading_from_corner,
    "shrinking_circle": shrinking_circle_light,
    "growing_circle": growing_circle_light,
    "outward_circle_wave": outward_circle_wave_light,
    "multiple_outward_circle_waves": multiple_outward_circle_waves_light,
    "inward_circle_wave": inward_circle_wave_light,
    "multiple_inward_circle_waves": multiple_inward_circle_waves_light,
    "moving_bar": moving_bar_light,
    "multiple_moving_bars": multiple_moving_bars_light,
}

def update_light(grid_size, light_function, time_step, cpm: CPM):
    """
    Initialize illumination 

    Parameters:
        grid_size : int - width or length of grid
        light_function : string - string that corresponds to desired light_function, see light_methods dict
        time_step : int - current time step
        cpm : CPM - current CPM with updated grid, light_pattern, and time_step
    Returns:
        binary np.ndarray with same dimensions as CPM grid where 1 indicates illminated regions and 0 indicated unlit regions
    """

    x_s, y_s = np.meshgrid(np.arange(grid_size), np.arange(grid_size))  # create a grid of x,y coordinates
    
    return (light_methods[light_function](y = y_s, x = x_s, t = time_step, cpm = cpm)).astype(int)
    
    """
    light_pattern = np.zeros((grid_size, grid_size), dtype=int)

    for y in range(grid_size):
        for x in range(grid_size):
            #print(light_function(y, x, time_step))
            light_pattern[y, x] = light_function(y, x, time_step)
            
            
    return light_pattern
    """
