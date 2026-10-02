"""
"""

import numpy as np
from collections import defaultdict

#from GridWorld import GridWorld, Action

class QLearningAgent:
    """
    Q-Learning agent for discrete action spaces.
    """

    def __init__(self, n_actions, alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 seed=None):
        self.n_actions = n_actions
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor for future rewards
        self.epsilon = epsilon_start 
        self.epsilon_end = epsilon_end # epsilon is for exploration / exploitation
        self.epsilon_decay = epsilon_decay
        self.np_random = np.random.default_rng(seed)

        # Q-table: maps a discretized state -> array of Q-values, one per action.
        # defaultdict initializes unseen states to zero for all actions
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))


    def discretize_state(self, obs):
        """
        Converts a raw environment observation into a hashable,
        discrete state you can use as a Q-table key.
        Hashable: basically immutable, i.e. a tuple of immutable values

        obs: the specific environment observation

        State vector:
        - Manhattan distance to nearest food item (int, range 0-14, & None if no food remains)
        - cardinal direction to nearest food item (string, 'NSEWO', 'O' for standing on food, & None if no food remains)
        - current glucose, binned (10 bins of size 10), 0-10
        -- TODO: think carefully about binning scheme
        """
        state = [None, None, None]

        # distance
        agent_pos = obs["agent_pos"]
        all_food_positions = obs["food_positions"]

        all_distances = [
            sum(abs(a - b) for a, b in zip(agent_pos, t)) # computes the Manhattan distance between agent and every food position
            for t in all_food_positions
        ]
        # the nearest food item
        state[0] = min(all_distances) if all_distances else None # the else statement catches when there's no more food in environment


        # direction
        if all_distances and min(all_distances) is not None: # checking that there is a remaining food item
            if state[0] == 0: # check if we are standing on a food tile
                state[1] = 'O'
            else: # we are not standing on a food tile
                nearest_food_index = all_distances.index(min(all_distances))
                nearest_food_pos = all_food_positions[nearest_food_index]
                dx = nearest_food_pos[1] - agent_pos[1] # cols are x pos (list of lists)
                dy = nearest_food_pos[0] - agent_pos[0] # rows are y pos
                if abs(dx) > abs(dy):
                    state[1] = 'E' if dx > 0 else 'W'
                else: # if equal arbitrarily choose y direction
                    state[1] = 'S' if dy > 0 else 'N'
        else: # no more food
            state[1] = None
        

        # binned glucose
        cur_glucose = obs["glucose_level"]
        state[2] = cur_glucose // 10 # binning glucose into 10-unit bins

        return tuple(state) # return as a hashable tuple for use as Q-table key



    def choose_action(self, state):
        """
        Chooses an action based on the current state using epsilon-greedy strategy.

        state: the discretized state (tuple)

        Returns: the chosen action (int)
        """
        if self.np_random.random() < self.epsilon:
            return self.np_random.integers(self.n_actions) # explore: random action
        else: # exploit: choose best action based on Q-table
            q_values = self.q_table[state]
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return self.np_random.choice(best_actions) # break ties randomly



    

