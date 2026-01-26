import numpy as np
#from .cpm import CPM

# STATIC LIGHT PATTERN FUNCTIONS

def static_circle_light(y, x, t):
    center = 37
    radius = 17
    return ((y - center)**2 + (x - center)**2) <= radius**2

def static_left_half_light(y, x, t):
    return x <= 17

def static_right_half_light(y, x, t):
    return x > 17

def no_light (y, x, t):
    return x*0

# DYNAMIC LIGHT PATTERN FUNCTIONS

def light_spreading_from_corner(y, x, t):
    return ((y+x) <= t)

def shrinking_circle_light(y, x, t):
    center = 17
    radius = 15 - 0.75*t
    return ((y - center)**2 + (x - center)**2) <= radius**2

def growing_circle_light(y, x, t):
    center = 17
    radius = 0 + 0.75*t
    return ((y - center)**2 + (x - center)**2) <= radius**2

def outward_circle_wave_light(y, x, t):
    
    # change these values
    center = 37
    initial_inner_radius = 0
    initial_outer_radius = 4
    
    # DO NOT CHANGE BELOW
    
    inner_radius = ((initial_inner_radius + 2*t) % y.shape[0]/2)
    outer_radius = ((initial_outer_radius + 2*t) % y.shape[0]/2)
    
    #if inner_radius > outer_radius : # specific case when wrapping around when inside is edge of grid and outside is at center
        #return (((y - center)**2 + (x - center)**2) <= inner_radius**2) & (((y - center)**2 + (x - center)**2) >= outer_radius**2)
    #else: # normal case
    return (((y - center)**2 + (x - center)**2) >= inner_radius**2) & (((y - center)**2 + (x - center)**2) <= outer_radius**2)
    
    #return (((y - center)**2 + (x - center)**2) >= inner_radius**2) & (((y - center)**2 + (x - center)**2) <= outer_radius**2)

def inward_circle_wave_light(y, x, t):
    center = 17
    initial_inner_radius = 31
    initial_outer_radius = 35
    
    inner_radius = ((initial_inner_radius - 2*t) % y.shape[0]/2)
    outer_radius = ((initial_outer_radius - 2*t) % y.shape[0]/2)
    
    #if inner_radius > outer_radius : # specific case when wrapping around when inside is edge of grid and outside is at center
        #return (((y - center)**2 + (x - center)**2) <= inner_radius**2) & (((y - center)**2 + (x - center)**2) >= outer_radius**2)
    #else: # normal case
    return (((y - center)**2 + (x - center)**2) >= inner_radius**2) & (((y - center)**2 + (x - center)**2) <= outer_radius**2)
    
    #return (((y - center)**2 + (x - center)**2) >= inner_radius**2) & (((y - center)**2 + (x - center)**2) <= outer_radius**2)

def moving_bar_light(y, x , t, light_speed):
    width = 7 # number of pixels
    #light_speed = 2 #some scaler with respect to time

    top = (int(t * light_speed)) % y.shape[0]
    bottom = (top + width) % y.shape[0]

    if top < bottom:
        return (y >= top) & (y <= bottom)
    else: # wrap around (bottom < top)
        return (y >= top) | (y <= bottom)

def multiple_moving_bars_light(y, x , t):
    
    #num_bars = 5
    #width = 3 # number of pixels
    #distance_between_bars = 2
    #speed = 1 #some scaler with respect to time

    spatial_period = 14                              # width + distance_between_bars
    duty_cycle = 0.28                                # width / spatial_period (fraction of period that is lit up)
    width = spatial_period * duty_cycle
    num_bars = int(y.shape[0] / spatial_period)
    speed = 1
    
    # mask for y true(1)/false(0)
    light_mask = np.zeros(y.shape, dtype=bool)
    
    for i in range(num_bars):
        top = int((t*speed + i*spatial_period) % y.shape[0]) # start taking remainder once top > size of grid -> creates wrap around
        bottom = int((top + width) % y.shape[0])
    
        if bottom < top: # specific case when wrapping around when top is at bottom of grid & bottom is at top
            light_mask |= (y >= top) | (y <= bottom)
        else: # normal case
            light_mask |= (y >= top) & (y <= bottom)
    
    return light_mask

# LIGHTING UPDATING FUNCTION

light_methods = {
    # Static light patterns
    "static_circle": static_circle_light,
    "static_left_half": static_left_half_light,
    "static_right_half": static_right_half_light,
    "no_light": no_light,

    # Dynamic light patterns
    "spreading_from_corner": light_spreading_from_corner,
    "shrinking_circle": shrinking_circle_light,
    "growing_circle": growing_circle_light,
    "outward_circle_wave": outward_circle_wave_light,
    "inward_circle_wave": inward_circle_wave_light,
    "moving_bar": moving_bar_light,
    "multiple_moving_bars": multiple_moving_bars_light,
}

def update_light(grid_size, light_function, time_step, light_speed):

    x_s, y_s = np.meshgrid(np.arange(grid_size), np.arange(grid_size))  # create a grid of x,y coordinates
    
    return (light_methods[light_function](y_s, x_s, time_step, light_speed)).astype(int)
    
    """
    light_pattern = np.zeros((grid_size, grid_size), dtype=int)

    for y in range(grid_size):
        for x in range(grid_size):
            #print(light_function(y, x, time_step))
            light_pattern[y, x] = light_function(y, x, time_step)
            
            
    return light_pattern
    """




# hardcoded light pattern arrays for 21x21 testing

zero_light = [
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

half_light = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

corner_light = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
