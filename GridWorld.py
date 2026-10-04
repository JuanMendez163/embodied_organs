"""
"""

import numpy as np
from enum import IntEnum
#from QLearningAgent import QLearningAgent

# ----------------------------------------------------------
# Interpretable constants used throughout the environment
# ----------------------------------------------------------

class Cell(IntEnum):
    """
    Maps interpretable env object names to integers.
    """

    EMPTY = 0
    FOOD = 1

class Action(IntEnum):
    """
    Maps interpretable action names to integers.
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    EAT = 4

DELTAS = {
    Action.UP:      (-1, 0), # negative means moving up because you are reducing index
    Action.DOWN:    (1, 0),
    Action.LEFT:    (0, -1),
    Action.RIGHT:   (0, 1)
}

# ----------------------------------------------------------


class GridWorld:
    """
    """

    def __init__(self, grid_size=8, glucose_target=50, glucose_max=100,
                 n_food=20, max_steps=200, metabolism_rate=5, intake_amount=10, 
                 drive_n=2, drive_m=1, seed=None):
        self.np_random = np.random.default_rng(seed)
        
        self.grid_size = grid_size
        self.glucose_target = glucose_target # homeostatic glucose target
        self.glucose_max = glucose_max # stomach capacity
        self.n_food = n_food # amount of food in environment
        self.max_steps = max_steps

        self.metabolism_rate = metabolism_rate # glucose lost per step
        self.intake_amount = intake_amount # glucose gained per eat

        self.n_actions = len(Action)

        self.drive_n = drive_n # exponent for drive calculation
        self.drive_m = drive_m # exponent for drive calculation

        self.reset()


    def reset(self, seed=None):
        """
        Reset to starting state and return initial observation.
        Only re-seeds the RNG if a seed is explicitly provided; otherwise
        continues drawing from the existing random stream.
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Initializing grid and agent position
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8) # initialize grid; holds env objects only, not agent
        self.agent_pos = (int(self.np_random.integers(0, self.grid_size)), int(self.np_random.integers(0, self.grid_size))) # random starting position

        # Placing food randomly
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) != self.agent_pos] # we don't want to place food on top of the agent
        flat_indices = self.np_random.choice(len(empty_cells), size=self.n_food, replace=False) # replace=FALSE avoids duplicate placements
        self.food_indices = [empty_cells[i] for i in flat_indices]
        for r, c in self.food_indices:
            self.grid[r, c] = Cell.FOOD

        # Initial glucose level
        self.glucose_start = self.glucose_target
        # while True:
        #     self.glucose_start = int(self.np_random.integers(self.glucose_target-self.metabolism_rate*8, self.glucose_target-self.metabolism_rate*2, endpoint=True)) # start with varying levels of hunger
        #     if self.glucose_start >= 0:
        #         break

        self.step_count = 0

        return self._get_obs()


    def _get_obs(self):
        """
        Environment state observation:
         - current position (x,y)
         - current glucose level
         - current drive (distance from homeostasis)
         - position of food items (list of (x,y) positions)
        Observation is stored as a dictionary.
        """
        if self.step_count == 0:
            self.current_glucose = self.glucose_start
        obs = {
            "agent_pos": self.agent_pos,
            "glucose_level": self.current_glucose,
            "drive": self._drive(self.drive_n, self.drive_m),
            "food_positions": self.food_indices
        }
        return obs


    def _drive(self, n=None, m=None):
        """
        Computes distance from homeostasis.
        Look at drive definition from Gutkin paper.
        n,m have important impact on shape of reward, drive curve
        """
        drive = ((abs(self.current_glucose - self.glucose_target))**n) ** (1/m)
        return drive

    def in_bounds(self, pos):
        """
        Checks if the position pos is within the bounds of the grid.
        """
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def step(self, action=None):
        """
        Advance the environment by one timestep.
        If action is None, an action is sampled randomly (for testing/demos);
        otherwise the given Action is taken (for RL agent integration).
        Returns (obs, reward, done, info), where done detetermines if episode is over,
        and info explains the current state of the agent.
        """
        cur_drive = self._drive(self.drive_n, self.drive_m)

        # Metabolism - how much glucose we lose each step
        self.current_glucose = max(0, self.current_glucose-self.metabolism_rate) # min glucose value is 0

        if action is None:
            action = self.np_random.choice(list(Action)) # we take a random action
        self.action = Action(action)
        if self.action not in DELTAS: # non-movement actions; right now only is EAT
            grid_item = self.grid[self.agent_pos[0], self.agent_pos[1]]

            # Eating
            if grid_item == Cell.FOOD: # check if we are standing on food
                self.current_glucose = min(self.current_glucose+self.intake_amount, self.glucose_max) # eating food; increase glucose (cap at max)
                self.grid[self.agent_pos[0], self.agent_pos[1]] = Cell.EMPTY # remove food item after eating
                self.food_indices.remove(self.agent_pos) # remove food from list of food positions
        else: # a movement action
            dr, dc = DELTAS[self.action]
            new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if self.in_bounds(new_pos): # checking if proposed position is within bounds of grid
                self.agent_pos = new_pos


        new_drive = self._drive(self.drive_n, self.drive_m) # we calculate a new drive after action + env impact (metabolism)
        # TODO: consider more complex starvation behavior
        if self.current_glucose == 0:
            self.reward -= 50 # reward steadily draining as you starve
        else:
            self.reward = cur_drive - new_drive # as defined in Gutkin paper

        self.step_count += 1
        if self.step_count == self.max_steps:
            done = True
            info = "Agent reached max steps."
        elif self.current_glucose == 0:
            done = True
            info = "Agent starved to death."
        else:
            done = False
            info = "Agent still exploring."
        return (self._get_obs(), self.reward, done, info)


    def render(self):
        """
        Renders the current state of the agent on a grid, including food locations and current glucose.
        """
        render_grid = np.full((self.grid_size, self.grid_size), "", dtype="<U10")
        food_positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if self.grid[r, c] == Cell.FOOD]
        r, c = self.agent_pos
        if self.agent_pos in food_positions: # when the agent is on a food tile
            render_grid[r, c] = 'AF'
        else:
            render_grid[r, c] = 'A' # agent position
        for food in food_positions:
            r, c = food
            if (r, c) != self.agent_pos:
                render_grid[r, c] = 'F' # food positions
        print(render_grid)
        print(f"GLUCOSE: {self.current_glucose}")
        print(f"COMPLETED ACTION: {self.action.name}")
        print(f"REWARD: {self.reward}")



   
if __name__ == "__main__":
    env = GridWorld()
    #agent = QLearningAgent(n_actions=4)

    print("Initial obs:", env._get_obs())

    for t in range(15):
        print("--------------------------------------")
        print(env.step()[0])
        env.render()
        obs = env._get_obs()
        #state = agent.discretize_state(obs)
        #print("Discretized state:", state)
        print("--------------------------------------")