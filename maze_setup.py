import numpy as np
import matplotlib.pyplot as plt

# Define the maze layout as a 10x10 grid
# 1 represents a wall, 0 represents a path
# S = Start (2), G = Sub-goal (3), E = End-goal (4)

maze = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 3, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 4, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Visualizing the maze using Matplotlib
def plot_maze(maze):
    plt.imshow(maze, cmap="coolwarm", origin="upper")
    plt.xticks([]), plt.yticks([])  # Remove axis ticks
    plt.title("Maze Layout")
    plt.show()

plot_maze(maze)