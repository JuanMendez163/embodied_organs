"""
"""

import numpy as np
import random
from enum import IntEnum



class Cell(IntEnum):
    """
    Maps interpretable env object names to integers.
    """

    EMPTY = 0
    FOOD = 1


class GridWorld:
    """
    """

    def __init__(self, grid_size=8, glucose_target=50.0, glucose_max=100.0,
                 n_food=5, max_steps=200):
        self.grid_size = grid_size
        self.glucose_target = glucose_target # homeostatic glucose target
        self.glucose_max = glucose_max # stomach capacity
        self.n_food = n_food # amount of food in environment
        self.max_steps = max_steps

        self.metabolism_rate = 5 # glucose lost per step
        self.intake_amount = 10 # glucose gained per eat

        self.reset()


    def reset(self):
        """
        Reset to starting state and return initial observation.
        """

        # Initializing grid and agent position
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8) # initialize grid; holds env objects only, not agent
        self.agent_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)) # random starting position

        # Placing food randomly
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) != self.agent_pos] # we don't want to place food on top of the agent
        flat_indices = np.random.choice(len(empty_cells), size=self.n_food, replace=False) # replace=FALSE avoids duplicate placements
        food_indices = [empty_cells[i] for i in flat_indices]
        for r, c in food_indices:
            self.grid[r, c] = Cell.FOOD

        # Initial glucose level
        # TODO: I feel like there is a better way to do this
        while True:
            self.glucose_start = random.randint(self.glucose_target-self.metabolism_rate*8, self.glucose_target-self.metabolism_rate*2) # start with varying levels of hunger
            if self.glucose_start >= 0:
                break

        self.step_count = 0

        return self._get_obs()

   
if __name__ == "__main__":
    env = GridWorld()