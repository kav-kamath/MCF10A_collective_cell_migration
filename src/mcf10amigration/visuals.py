import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import ffmpeg
from IPython.display import HTML

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

from .cpm import *

# visualization to show whats going on

# putting together the animation
# plot of how often an event occurs
# 2x2 & 5x5 static plots

# visualize light pattern

def plot_static_light_pattern(cpm, save_boolean=False, output_filename="static_light_pattern.png"):
    """
    Plot the binary light pattern applied to the simulation grid. Currently only works for static light pattern.
    
    Parameters
        self : CPM - CPM object as defined in cpm.py
        save_boolean : bool, optional - if True, saves the plot to a file.
            Default is False.
        output_filename : str, optional - file path for saving the plot if `save_boolean` is True.
            Default is "light_pattern.png".

    Returns
        None (Displays the plot and optionally saves it.)    
    """
    
    plt.figure(figsize=(5, 5))
    plt.imshow(cpm.light_pattern, cmap='hot', interpolation='nearest')
    plt.title("Light Pattern")
    plt.colorbar(label="Light (0=off, 1=on)")
    
    if save_boolean:
        plt.savefig(output_filename)
    
    plt.show()


def visualize_dynamic_light_pattern(light_patterns, times, background_color=(1, 1, 1), save_boolean=True, output_filename="dynamic_light_pattern.mp4"):
    
    """
    Create (and optionally save to file) animation of dynamic light pattern over the course of a simulation.

    Parameters
        light_patterns : list of np.ndarray - list of 2D arrays representing light patterns at each time point
        times : list of float - corresponding simulation times(gillespie)/steps(monte carlo) for each frame
        background_color : tuple, optional - RGB values for background
            Default is white: (1, 1, 1)
        save_boolean : bool, optional - if True, the animation is saved to a file
            Default is True.
        output_filename : str, optional - Name of the file to save the animation to.
            Default is "dynamic_light_pattern.mp4".
            
    Returns
        None (Creates and optionally saves the animation to file. Does not display it.)
    """

    # custom colormap
    cmap = ListedColormap([background_color, (0, 1, 1)])  # background color and green/blue for light

    fig, ax = plt.subplots()
    image = ax.imshow(light_patterns[0], cmap=cmap, interpolation='nearest', vmax=1)

    def update(frame_idx):
        image.set_array(light_patterns[frame_idx])
        ax.set_title(f"Time: {times[frame_idx]:.5f}")
        return image,

    ani = FuncAnimation(fig, update, frames=len(light_patterns), interval=100, blit=True)

    if save_boolean:
        ani.save(output_filename, writer=animation.FFMpegWriter(fps=5))

    plt.close(fig)
    #return HTML(ani.to_jshtml())

def animate_simulation(frames, times, background_color=(1, 1, 1), save_boolean=True, output_filename="current_simulation.mp4"):
    """
    Create (and optionally save to file) animation of CPM grid over the course of a simulation.

    Parameters
        frames : list of np.ndarray - list of 2D arrays representing CPM grid states at each time point
        times : list of float - corresponding simulation times(gillespie)/steps(monte carlo) for each frame
        background_color : tuple, optional - RGB values for background
            Default is white: (1, 1, 1)
        save_boolean : bool, optional - if True, the animation is saved to a file
            Default is True.
        output_filename : str, optional - Name of the file to save the animation to.
            Default is "current_simulation.mp4".
            
    Returns
        None (Creates and optionally saves the animation to file. Does not display it.)
    """

    # custom colormap
    num_colors = np.max([np.max(frame) for frame in frames])    # old color method (doesn't allow for change # of cells): num_colors = np.max(cpm.grid)
    random_colors = np.random.rand(num_colors + 1, 3) # random rbg values
    random_colors[0, :] = background_color  # set background color to white

    cmap = ListedColormap(random_colors)

    fig, ax = plt.subplots()
    image = ax.imshow(frames[0], cmap=cmap, interpolation='nearest')

    def update(frame_idx):
        image.set_array(frames[frame_idx])
        ax.set_title(f"Time: {times[frame_idx]:.5f}")
        return image,

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)

    if save_boolean:
        ani.save(output_filename, writer=animation.FFMpegWriter(fps=10))

    plt.close(fig)
    #return HTML(ani.to_jshtml())

def plot_one_frame(frame, title = None):
    """
    Display only the first CPM simulation frame.

    Parameters
        frames : list of np.ndarray - list of grid states at each time point
        cmap : matplotlib.colors.Colormap - colormap for display

    Returns:
        None (Displays a static 2x2 figure of plots of the first four frames.)
    """
    
    num_colors = np.max([np.max(frame) for frame in frame])    # old color method (doesn't allow for change # of cells): num_colors = np.max(cpm.grid)
    random_colors = np.random.rand(num_colors + 1, 3) # random rbg values
    random_colors[0, :] = (1, 1, 1) # set background color to white

    cmap = ListedColormap(random_colors)
    
    fig, ax = plt.subplots()
    image = ax.imshow(frame, cmap=cmap, interpolation='nearest')
    
    if title is not None:
        ax.set_title(title)

    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def plot_2x2(frames, cmap):
    """
    Display a 2x2 grid of the first four CPM simulation frames.

    Parameters
        frames : list of np.ndarray - list of 2D arrays representing CPM grid states at each time point
        cmap : matplotlib.colors.Colormap - Colormap to use for displaying the frames.

    Returns
        None (Displays a static 2x2 figure of plots of the first four frames.)
    """
    
    # Create a 2x2 plot of the first four frames
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(4):
        axes[i].imshow(frames[i], cmap=cmap, interpolation='nearest')
        axes[i].set_title(f"Frame {i + 1}")

    plt.tight_layout()
    plt.show()


def plot_5x5(frames, cmap = None):
    """
    Display a 5x5 grid of the first 25 CPM simulation frames.

    Parameters
        frames : list of np.ndarray - list of 2D arrays representing CPM grid states at each time point
        cmap : matplotlib.colors.Colormap - Colormap to use for displaying the frames.

    Returns
        None (Displays a static 5x5 figure of plots of the first 25 frames.)
    """
    
    # custom colormap
    if cmap is None:
        background_color=(1, 1, 1)
        num_colors = np.max([np.max(frame) for frame in frames])    # old color method (doesn't allow for change # of cells): num_colors = np.max(cpm.grid)
        random_colors = np.random.rand(num_colors + 1, 3) # random rbg values
        random_colors[0, :] = background_color  # set background color to white
        cmap = ListedColormap(random_colors)
    
    
    # Create a 5x5 plot of the first 16 frames
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(16):
        axes[i].imshow(frames[i], cmap=cmap, interpolation='nearest')
        axes[i].set_title(f"Time: {i}")

    plt.tight_layout()
    plt.show()


def plot_event_times(event_times, sim_type="Gillespie"):
    """
    Plot a timeline of event occurrences in the simulation. More intended for gillespie simulation, but can be used for monte carlo too.

    Parameters
        event_times : list of float - times(gillespie)/steps(monte carlo) at which each event occurred during the simulation.

    sim_type : str, optional - label for the simulation type shown in the plot title.
        Default is "Gillespie".

    Returns
        None - displays a timeline plot of event timings and average waiting time.
    """
    
    if (sim_type == "Monte Carlo"):
        print("Notes:")
        print("Average waiting time is trivial for Monte Carlo simulations, as events occur at each integer step.")
        print("Simulation time corresponds to step number, not time.")
    
    #get avg waiting time
    waiting_times = np.diff(event_times)
    avg_waiting_time = np.mean(waiting_times)

    # plot event times (like timeline, every dot represents 1 event)
    plt.figure(figsize=(10, 2))
    plt.hlines(1, event_times[0], event_times[-1], color='lightgray', linewidth=2)  # timeline line
    plt.eventplot(event_times, lineoffsets=1, colors='tab:blue', linelengths=0.3)
    plt.scatter(event_times, [1]*len(event_times), color='tab:blue', zorder=3)
    plt.yticks([])
    plt.xlabel("Simulation time")
    plt.title(f"Timeline of {sim_type} events")

    plt.text(
    x=event_times[0], 
    y=1.1, 
    s=f"Avg waiting time: {avg_waiting_time:.4f}", # put avg waiting time on top left
    fontsize=12,
    color='black')
    
    plt.tight_layout()
    plt.show()