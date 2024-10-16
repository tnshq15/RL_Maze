import numpy as np
import random
import matplotlib.pyplot as plt
from maze_setup import maze  # Importing the maze from the other file

# Initialize reward system
reward_system = np.full(maze.shape, -1)  # Default reward for each step
reward_system[maze == 1] = -10  # Walls have a high negative reward
reward_system[maze == 3] = 10   # Sub-goal has a positive reward
reward_system[maze == 4] = 100  # End-goal has the highest reward

# Initialize Q-table
q_table = np.zeros((maze.shape[0], maze.shape[1], 4))  # 4 actions: up, down, left, right

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000  # Number of training episodes
max_steps_per_episode = 100  # Maximum number of steps per episode

# Actions: up, down, left, right
actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# Function to get the next state based on action
def get_next_state(state, action):
    new_state = (state[0] + action[0], state[1] + action[1])
    if maze[new_state] == 1:  # If the next state is a wall, stay in the same state
        return state
    return new_state

# Q-learning algorithm
for episode in range(num_episodes):
    state = (1, 1)  # Starting position (S)
    done = False
    step = 0

    while not done and step < max_steps_per_episode:
        # Choose an action using epsilon-greedy strategy
        if random.uniform(0, 1) < epsilon:
            action_idx = random.randint(0, 3)  # Explore
        else:
            action_idx = np.argmax(q_table[state[0], state[1]])  # Exploit best action

        action = actions[action_idx]
        next_state = get_next_state(state, action)
        reward = reward_system[next_state]

        # Update Q-value using the Q-learning formula
        best_next_action = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action_idx] += alpha * (reward + gamma * best_next_action - q_table[state[0], state[1], action_idx])

        # Transition to the next state
        state = next_state
        step += 1

        # Check if the agent has reached the end goal
        if maze[state] == 4:
            done = True

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{num_episodes} completed")

print("Training completed.")

# Function to visualize the agent's learned path
def visualize_path(q_table, maze):
    path = np.copy(maze)
    state = (1, 1)  # Start position
    while maze[state] != 4:  # Until the agent reaches the end goal
        action_idx = np.argmax(q_table[state[0], state[1]])
        action = actions[action_idx]
        next_state = get_next_state(state, action)
        if next_state == state:  # Break if the agent is stuck
            break
        path[next_state] = 5  # Mark the agent's path with a unique value
        state = next_state

    plt.imshow(path, cmap="coolwarm", origin="upper")
    plt.xticks([]), plt.yticks([])  # Remove axis ticks
    plt.title("Learned Path of the Agent")
    plt.show()

# Visualize the learned path
visualize_path(q_table, maze)