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

        self.num_elements = num_elements
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_elements,))  # State represents the current order of elements
        self.network = SortingNetwork(self.num_elements)
        self.action_space = spaces.Discrete(self.network.num_comparators-1)  # Actions correspond to indices of elements


    def reset(self):
        self.state = np.random.rand(self.num_elements)  # Initialize the state with random order of elements
        self.network = SortingNetwork(self.num_elements)
        return self.state.copy()

    def step(self, action):

        # swap
        tmp1=self.network.copy()
        tmp_arr1=self.state.copy()
        # Perform the selected action (swap elements at the given indices)
        tmp1.comparators[action],tmp1.comparators[action+1]=tmp1.comparators[action+1],tmp1.comparators[action]
        tmp_arr1=tmp1.sort(tmp_arr1)

        # Calculate the reward based on the sorting quality (e.g., number of out-of-order pairs)
        reward1 = -get_num_out_of_order_pairs(tmp_arr1)

        # delete comprator
        tmp2=self.network.copy()
        tmp_arr2=self.state.copy()
        # Perform the selected action (swap elements at the given indices)
        tmp2.delete_comp(action)
        tmp_arr2=tmp2.sort(tmp_arr2)

        # Calculate the reward based on the sorting quality (e.g., number of out-of-order pairs)
        reward2 = -get_num_out_of_order_pairs(tmp_arr2)+2

        # add comperator
        tmp3=self.network.copy()
        tmp_arr3=self.state.copy()
        # Perform the selected action (swap elements at the given indices)
        tmp3.add_comp(action)
        tmp_arr3=tmp3.sort(tmp_arr3)

        # Calculate the reward based on the sorting quality (e.g., number of out-of-order pairs)
        reward3 = -get_num_out_of_order_pairs(tmp_arr3)-2

        if reward1> reward2 and reward1 > reward3:
            self.state=tmp_arr1
            self.network=tmp1
            reward=reward1
        else:
            if reward2>reward1 and reward2>reward3:
                self.state = tmp_arr2
                self.network = tmp2
                reward = reward2
            else:
                self.state = tmp_arr3
                self.network = tmp3
                reward = reward3


        # Check if the sorting is complete
        done = self._is_sorted()

        # Return the next state, reward, and done flag
        return self.state.copy(), reward, done, {}

    def _is_sorted(self):
        return np.all(self.state[:-1] <= self.state[1:])

    def _get_num_out_of_order_pairs(self):
        return np.sum(self.state[:-1] > self.state[1:])




# Create the SortingNetworkEnv
env = SortingNetworkEnv(num_elements=6)

# # Create the DQN agent
# model = DQN('MlpPolicy', env, learning_rate=0.1, buffer_size=100000, learning_starts=1000,
#             batch_size=64, gamma=0.6, target_update_interval=500, exploration_fraction=0.4,
#             exploration_final_eps=0.01, verbose=1)

model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.1,gamma=0.5,exploration_fraction=0.7)

# Train the agent
model.learn(total_timesteps=10000 )

# # Save the trained model
# model.save("dqn_sorting_network")


# Evaluate the agent
num_episodes = 1
total_rewards = 0
max_episode_length = 100  # Set a maximum episode length
# Initialize a list to store the rewards per iteration
reward_progress = []
fig = go.Figure()
plt.ion()  # Enable interactive mode
for _ in range(num_episodes):
    obs = env.reset()
    episode_reward = 0
    done = False
    step_count = 0  # Track the number of steps taken in the current episode


    while not done and step_count<1000 :
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


    print(1)

    total_rewards += episode_reward
env.network.draw()
plt.ioff()  # Turn off interactive mode
average_reward = total_rewards / step_count
print(f"Average reward over {num_episodes} episodes: {average_reward}")
print("net size ",env.network.num_comparators)
# Extract the network parameters
# params = model.policy.get_parameter()
# Close the environment
env.close()
