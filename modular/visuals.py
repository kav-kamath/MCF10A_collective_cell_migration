import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

# fun visualization to show whats going on

# putting together the animation
# plot of how often an event occurs
# 2x2 & 5x5 static plots

# visualize light pattern

def plot_light_pattern(self):
    plt.figure(figsize=(5, 5))
    plt.imshow(self.light_pattern, cmap='hot', interpolation='nearest')
    plt.title("Light Pattern")
    plt.colorbar(label="Light (0=off, 1=on)")
    plt.show()


def animate_simulation(frames, times, background_color=(1, 1, 1)):

    num_colors = np.max([np.max(frame) for frame in frames])
    random_colors = np.random.rand(num_colors + 1, 3)
    random_colors[0, :] = background_color  # set background

    cmap = ListedColormap(random_colors)

    fig, ax = plt.subplots()
    image = ax.imshow(frames[0], cmap=cmap, interpolation='nearest')

    def update(frame_idx):
        image.set_array(frames[frame_idx])
        ax.set_title(f"Time: {times[frame_idx]:.5f}")
        return image,

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def save_simulation(frames, times, output_path="simulation.gif", fps=10):

    num_colors = np.max([np.max(frame) for frame in frames])
    random_colors = np.random.rand(num_colors + 1, 3)
    random_colors[0, :] = [1, 1, 1]

    cmap = ListedColormap(random_colors)
    fig, ax = plt.subplots()
    image = ax.imshow(frames[0], cmap=cmap, interpolation='nearest')

    def update(frame_idx):
        image.set_array(frames[frame_idx])
        ax.set_title(f"Time: {times[frame_idx]:.5f}")
        return image,

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    ani.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)