import numpy as np
import random

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