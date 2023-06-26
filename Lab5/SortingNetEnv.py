import gym
import plotly.graph_objects as go
import numpy as np
from gym import spaces
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from sympy.abc import epsilon


class SortingNetworkEnv(gym.Env):
    def __init__(self, num_elements):
        super(SortingNetworkEnv, self).__init__()

        self.num_elements = num_elements
        self.action_space = spaces.Discrete(num_elements-1)  # Actions correspond to indices of elements
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_elements,))  # State represents the current order of elements

    def reset(self):
        self.state = np.random.rand(self.num_elements)  # Initialize the state with random order of elements
        return self.state.copy()

    def step(self, action):
        # Perform the selected action (swap elements at the given indices)
        i, j = action, action + 1
        self.state[i], self.state[j] = self.state[j], self.state[i]

        # Calculate the reward based on the sorting quality (e.g., number of out-of-order pairs)
        reward = -self._get_num_out_of_order_pairs()

        # Check if the sorting is complete
        done = self._is_sorted()

        # Return the next state, reward, and done flag
        return self.state.copy(), reward, done, {}

    def _is_sorted(self):
        return np.all(self.state[:-1] <= self.state[1:])

    def _get_num_out_of_order_pairs(self):
        return np.sum(self.state[:-1] > self.state[1:])

# Create the SortingNetworkEnv
env = SortingNetworkEnv(num_elements=10)

# # Create the DQN agent
# model = DQN('MlpPolicy', env, learning_rate=0.1, buffer_size=100000, learning_starts=1000,
#             batch_size=64, gamma=0.6, target_update_interval=500, exploration_fraction=0.4,
#             exploration_final_eps=0.01, verbose=1)

model = DQN('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000 )

# Save the trained model
model.save("dqn_sorting_network")

# Load the trained model
model = DQN.load("dqn_sorting_network")

# Extract the network parameters
params = model.policy.parameters()
# Get the sorted order using the network parameters
sorted_order = np.argsort(params)

# Plot the sorted order
plt.bar(range(len(sorted_order)), sorted_order)
plt.xlabel('Index')
plt.ylabel('Element Value')
plt.title('Optimal Sorting Network')
plt.show()

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


    while not done:
        # if np.random.uniform() < 0.5:
        #     # Explore: Choose a random action
        #     action = env.action_space.sample()
        # else:
        #     # Exploit: Choose the action with the highest predicted value
        #     action, _ = model.predict(obs)
        action, a = model.predict(obs)
        action = action.item()  # Convert action to scalar integer
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        step_count += 1
        # if step_count >= max_episode_length or reward >= 0:
        #     # Break out of the loop if the maximum episode length is reached
        #     # or if the environment is not making any progress in sorting
        #     break
        reward_progress.append(reward)  # Append the reward to the progress list
        # Plot the reward progress
        plt.plot(reward_progress)
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.title('Reward Progress')
        plt.draw()
        plt.pause(0.001)  # Pause to allow the plot to update
        plt.clf()  # Clear the plot for the next iteration
        # Update the plot with the current reward progress
        # fig.add_trace(go.Scatter(x=list(range(len(reward_progress))),
        #                          y=reward_progress,
        #                          mode='lines',
        #                          name='Reward Progress'))
        #
        # fig.update_layout(title='Reward Progress',
        #                   xaxis_title='Step',
        #                   yaxis_title='Reward')
        #
        # fig.show()

    print(1)

    total_rewards += episode_reward
plt.ioff()  # Turn off interactive mode
average_reward = total_rewards / num_episodes
print(f"Average reward over {num_episodes} episodes: {average_reward}")
# Extract the network parameters
# params = model.policy.get_parameter()
# Close the environment
env.close()
