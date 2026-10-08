"""
"""

import numpy as np
from collections import defaultdict

from GridWorld import GridWorld

class QLearningAgent:
    """
    Q-Learning agent for discrete action spaces.
    """

    def __init__(self, n_actions, alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 seed=None):
        self.n_actions = n_actions
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor for future rewards
        self.epsilon = epsilon_start # epsilon is for exploration / exploitation
        self.epsilon_min = epsilon_min 
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
            q_values = self.q_table[state] # array of Q-values for state
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q] # i represents action index
            return self.np_random.choice(best_actions) # break ties randomly


    def update(self, state, action, reward, next_state):
        """
        Updates the Q-value for the given state-action pair using the Q-learning update rule.

        state: the current state (tuple)
        action: the action taken (int)
        reward: the reward received (float)
        next_state: the next state (tuple)
        done: whether the episode has ended (bool)
        """  
        old_q = self.q_table[state][action]
        next_q_values = self.q_table[next_state]
        max_next_q = max(next_q_values)
        # Q-learning update rule  
        self.q_table[state][action] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)


    def decay_epsilon(self):
        """
        Decreases the epsilon value for epsilon-greedy strategy to balance exploration and exploitation.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)




def train(env, agent, n_episodes=1000):
    """
    Main training loop for the Q-learning agent.

    env: the environment to interact with
    agent: the Q-learning agent instance
    n_episodes: number of episodes to train the agent

    Returns: a list of total rewards received per episode
    """
    episode_total_rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        state = agent.discretize_state(obs)
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state) # choose action based off state and epsilon-greedy
            next_obs, reward, done, info = env.step(action) # perform chosen action
            next_state = agent.discretize_state(next_obs)
            agent.update(state, action, reward, next_state) # update Q-table
            state = next_state
            #print(state)
            #print(f"Glucose Level: {next_obs['glucose_level']}")
            #print(f"Next Obs Drive: {next_obs['drive']}")
            #print(f"Reward: {reward}")
            total_reward += reward
            #print(f"Total Reward: {total_reward}")
        agent.decay_epsilon() # decay epsilon per episode
        episode_total_rewards.append(total_reward)
    return episode_total_rewards


if __name__ == "__main__":
    # debugging: first put code into Claude to see suggestions for debugging
    # also want to have some sort of renderer perhaps so I can get visual intuition of what it's doing
    # first want to see if I can get learning behavior in this env; then introduce further complexities to env
    # consider glucose bin size for debugging

    env = GridWorld(grid_size=6, glucose_target=50, glucose_max=100,
                 n_food=20, max_steps=200, metabolism_rate=5, intake_amount=10, 
                 drive_n=2, drive_m=1, seed=None)
    agent = QLearningAgent(n_actions=env.n_actions, alpha=0.3, gamma=0.95,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.7,
                 seed=None)
    rewards = train(env, agent, n_episodes=5000)
    print(rewards)