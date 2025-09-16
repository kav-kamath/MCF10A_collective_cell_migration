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

# visualization to show whats going on

# putting together the animation
# plot of how often an event occurs
# 2x2 & 5x5 static plots

# visualize light pattern

def plot_light_pattern(self, save_boolean=False, output_filename="light_pattern.png"):
    plt.figure(figsize=(5, 5))
    plt.imshow(self.light_pattern, cmap='hot', interpolation='nearest')
    plt.title("Light Pattern")
    plt.colorbar(label="Light (0=off, 1=on)")
    
    if save_boolean:
        plt.savefig(output_filename)
    
    plt.show()


def animate_simulation(frames, times, background_color=(1, 1, 1), save_boolean=True, output_filename="current_simulation.mp4"):

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

    ani = FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)

    if save_boolean:
        ani.save(output_filename, writer=animation.FFMpegWriter(fps=5))

    plt.close(fig)
    return HTML(ani.to_jshtml())




# Create a 2x2 plot of the first four frames
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for i in range(4):
    axes[i].imshow(frames_for_plot[i], cmap=cmap, interpolation='nearest')
    axes[i].set_title(f"Frame {i + 1}")

plt.tight_layout()
plt.show()


    
# Create a 5x5 plot of the first 16 frames
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()

for i in range(16):
    axes[i].imshow(frames_for_plot[i], cmap=cmap, interpolation='nearest')
    axes[i].set_title(f"Frame {i + 1}")

plt.tight_layout()
plt.show()



# plot event times (like timeline, every dot represents 1 event)
plt.figure(figsize=(10, 2))
plt.hlines(1, event_times[0], event_times[-1], color='lightgray', linewidth=2)  # timeline line
plt.eventplot(event_times, lineoffsets=1, colors='tab:blue', linelengths=0.3)
plt.scatter(event_times, [1]*len(event_times), color='tab:blue', zorder=3)
plt.yticks([])
plt.xlabel("Simulation time")
plt.title("Timeline of Gillespie events")
plt.tight_layout()
plt.show()

#get avg waiting time
waiting_times = np.diff(event_times)
avg_waiting_time = np.mean(waiting_times)
print(f"Average waiting time between events: {avg_waiting_time:.4f}") # put avg waiting time on the graphs