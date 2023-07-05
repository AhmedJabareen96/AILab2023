import random
import time

import gym
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from sorting_network import SortingNetwork


def get_num_out_of_order_pairs(state):
    return np.sum(state[:-1] > state[1:])
class SortingNetworkEnv(gym.Env):
    def __init__(self, num_elements):
        super(SortingNetworkEnv, self).__init__()
        self.episodes=0
        self.current_step=0
        self.added_comp=0
        self.num_elements = num_elements
        self.observation_space = spaces.Box(low=0, high=1, shape=(100,num_elements))  # State represents the current order of elements
        self.network = SortingNetwork(self.num_elements)

        self.state = np.random.rand(100, self.num_elements)  # Initialize the state with random order of elements
        # Update the action space to include swap, add, and remove operations
        self.action_space = spaces.Discrete(3)  # Three possible actions: swap, add, remove



    def reset(self):
        self.state = np.random.rand(100,self.num_elements)  # Initialize the state with random order of elements
        self.network = SortingNetwork(self.num_elements)
        self.current_step = 0
        self.added_comp=0
        self.episodes+=1
        return self.state.copy()
    def step(self, action):
        self.state = self.network.sort(self.state)
        if self._is_sorted():
            return self.state.copy(),0,True,{}

        if action == 0:
            # Swap operation
            self.apply_swap_action()
            self.state = self.network.sort(self.state)

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
        for i in range(len(self.state)):
            arr=self.state[i]
            if not ( np.all(arr[:-1]<=arr[1:]) ):
                return False
            else:
                self.state[i]= np.random.rand(self.num_elements)

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

model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.1,gamma=0.95,exploration_fraction=0.3,buffer_size=1000)


# Start the timer
start_time = time.time()
# Train the agent
model.learn(total_timesteps=10000)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the elapsed time
print("Elapsed time:", elapsed_time, "seconds")
print(f"took {env.episodes} to learn the model")


# Evaluate the agent
num_episodes = 10
total_rewards = 0
max_episode_length = 100  # Set a maximum episode length
# Initialize a list to store the rewards per iteration
reward_progress = []

env.network.draw()

reward_per_step = []
reward_per_episode = []
episode_lengths = []
for _ in range(num_episodes):
    obs = env.reset()

    episode_reward = 0
    done = False
    step_count = 0  # Track the number of steps taken in the current episode


    while not done :
        action, q = model.predict(obs)
        action = action.item()  # Convert action to scalar integer
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        step_count += 1
        reward_per_step.append(reward )



    episode_lengths.append(step_count)
    reward_per_episode.append(episode_reward/step_count)


    total_rewards += episode_reward
# env.network.draw()
average_reward = total_rewards / step_count

average_reward_per_step = sum(reward_per_step) / len(reward_per_step)
average_reward_per_episode = sum(reward_per_episode) / len(reward_per_episode)
average_episode_length = sum(episode_lengths) / len(episode_lengths)
print(f"Average episode length : {average_episode_length} steps ")

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(reward_per_step)
plt.xlabel("Step")
plt.ylabel("Average Reward per Agent Step")
plt.title("Average Reward per Agent Step")

plt.subplot(1, 3, 2)
plt.plot(reward_per_episode)
plt.xlabel("Episode")
plt.ylabel("Average Reward per Episode")
plt.title("Average Reward per Episode")

plt.subplot(1, 3, 3)
plt.plot(episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Average Episode Length (Steps)")
plt.title("Average Episode Length (Steps)")

plt.tight_layout()
plt.show()

# Extract the network parameters
# params = model.policy.get_parameter()
# Close the environment
env.close()
