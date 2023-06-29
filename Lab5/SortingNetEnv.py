import random

import gym
import plotly.graph_objects as go
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from sympy.abc import epsilon
from sorting_network import SortingNetwork


def get_num_out_of_order_pairs(state):
    return np.sum(state[:-1] > state[1:])
class SortingNetworkEnv(gym.Env):
    def __init__(self, num_elements):
        super(SortingNetworkEnv, self).__init__()

        self.current_step=0
        self.added_comp=0
        self.num_elements = num_elements
        self.observation_space = spaces.Box(low=0, high=1, shape=(100,num_elements))  # State represents the current order of elements
        self.network = SortingNetwork(self.num_elements)
        # self.action_space = spaces.Discrete(self.network.num_comparators-1)  # Actions correspond to indices of elements
        self.state = np.random.rand(100, self.num_elements)  # Initialize the state with random order of elements
        # Update the action space to include swap, add, and remove operations
        self.action_space = spaces.Discrete(3)  # Three possible actions: swap, add, remove



    def reset(self):
        self.state = np.random.rand(100,self.num_elements)  # Initialize the state with random order of elements
        self.network = SortingNetwork(self.num_elements)
        self.current_step = 0
        self.added_comp=0
        return self.state.copy()
    def step(self, action):
        if self._is_sorted():
            return self.state.copy(),0,True,{}

        if action == 0:
            # Swap operation
            self.apply_swap_action()
            self.state=self.network.sort(self.state)
        elif action == 1:
            # Add operation
            self.apply_add_action()
            # if len(self.network.comparators) > 0:
            #     self.network.comparators[np.random.randint(0, len(self.network.comparators))]=[np.random.randint(0, len(self.state)),np.random.randint(0, len(self.state))]
            self.state=self.network.sort(self.state)
        elif action == 2:
            # Remove operation
            self.apply_remove_action()
            self.state=self.network.sort(self.state)
        self.optimize()
        reward = -self._get_num_out_of_order_pairs() - self.added_comp
        self.current_step+=1
        # Check if the sorting is complete
        done = self._is_sorted() #or self.current_step>10000

        # Return the next state, reward, and done flag
        return self.state.copy(), reward, done, {}




    def apply_swap_action(self):
        # Swap elements in the sorting network
        if len(self.network.comparators)>0:
            i, j = np.random.randint(0, len(self.network.comparators)),np.random.randint(0, len(self.network.comparators))
            self.network.comparators[i], self.network.comparators[j] = self.network.comparators[j], self.network.comparators[i]

    def apply_add_action(self):
        self.added_comp+=1
        self.network.add_comp()

    def apply_remove_action(self):
        if len(self.network.comparators)>0:
            self.network.delete_comp(np.random.randint(0, len(self.network.comparators)))
            self.added_comp-=1
    def optimize(self):
        index = 0
        while index < len(self.network.comparators) - 1:
            current_element = self.network.comparators[index]
            next_element = self.network.comparators[index + 1]

            if current_element[0] == next_element[0] and current_element[1]==next_element[1]:
                self.network.comparators = np.delete(self.network.comparators, index + 1, axis=0)
                self.added_comp-=1
            else:
                index += 1

    def _is_sorted(self):
        for arr in self.state:
            if not ( np.all(arr[:-1]<=arr[1:]) ):
                return False
        return True

        #return np.all(self.state[:-1] <= self.state[1:])

    def _get_num_out_of_order_pairs(self):
        sum=0
        for arr in self.state:
            sum+= np.sum(arr[:-1] > arr[1:])
        return sum/100




# Create the SortingNetworkEnv
env = SortingNetworkEnv(num_elements=6)

# # Create the DQN agent
# model = DQN('MlpPolicy', env, learning_rate=0.1, buffer_size=100000, learning_starts=1000,
#             batch_size=64, gamma=0.6, target_update_interval=500, exploration_fraction=0.4,
#             exploration_final_eps=0.01, verbose=1)

model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.1,gamma=0.95,exploration_fraction=0.3)

# Train the agent
model.learn(total_timesteps=10000)

# # Save the trained model
# model.save("dqn_sorting_network")


# Evaluate the agent
num_episodes = 1
total_rewards = 0
max_episode_length = 100  # Set a maximum episode length
# Initialize a list to store the rewards per iteration
reward_progress = []

env.network.draw()
print("net size ",len(env.network.comparators) )
for _ in range(num_episodes):
    obs = env.reset()
    episode_reward = 0
    done = False
    step_count = 0  # Track the number of steps taken in the current episode


    while not done :
        action, a = model.predict(obs)
        action = action.item()  # Convert action to scalar integer
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        step_count += 1
        # reward_progress.append(reward)  # Append the reward to the progress list
        # # Plot the reward progress
        # plt.plot(reward_progress)
        # plt.xlabel('Iteration')
        # plt.ylabel('Reward')
        # plt.title('Reward Progress')
        # plt.draw()
        # plt.pause(0.001)  # Pause to allow the plot to update
        # plt.clf()  # Clear the plot for the next iteration


    total_rewards += episode_reward
# env.network.draw()
average_reward = total_rewards / step_count
print(f"Average reward over {num_episodes} episodes: {average_reward}")
print("sorted in : ",step_count)

# Extract the network parameters
# params = model.policy.get_parameter()
# Close the environment
env.close()
