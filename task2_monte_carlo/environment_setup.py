"""
Maze environment setup for Monte Carlo task.
Creates mazes of varying difficulty for exploration-exploitation analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, List


class MazeEnv:
    """
    Grid-based maze environment for Monte Carlo learning.
    Agent navigates from start to goal, optimizing path with exploration.
    
    Discrete state space: position (x, y) → unique state index
    Discrete actions: 0=up, 1=down, 2=left, 3=right
    """
    
    def __init__(self, maze_size: int = 8, difficulty: str = 'medium'):
        """
        Initialize maze environment.
        
        Args:
            maze_size: Size of square maze (e.g., 8 = 8x8 grid)
            difficulty: 'easy', 'medium', or 'hard' (affects wall density)
        """
        self.maze_size = maze_size
        self.difficulty = difficulty
        self.num_actions = 4  # up, down, left, right
        self.num_states = maze_size * maze_size
        
        # Generate maze
        self.maze = self._generate_maze(difficulty)
        
        # Positions
        self.start_pos = (0, 0)
        self.goal_pos = (maze_size - 1, maze_size - 1)
        self.agent_pos = self.start_pos
        
        # Ensure start and goal are not walls
        self.maze[self.start_pos] = 0
        self.maze[self.goal_pos] = 0
        
        self.steps = 0
        self.max_steps = maze_size * maze_size * 3
        
        # Visualization
        self.fig = None
        self.ax = None
        self.render_delay = 0.1  # Delay between renders in seconds
        
    def _generate_maze(self, difficulty: str) -> np.ndarray:
        """
        Generate a random maze with guaranteed paths.
        
        Args:
            difficulty: Controls wall density
            
        Returns:
            maze: 2D array where 0=free, 1=wall
        """
        if difficulty == 'easy':
            wall_density = 0.10
        elif difficulty == 'medium':
            wall_density = 0.20
        else:  # hard
            wall_density = 0.30
        
        # Create random maze
        maze = (np.random.random((self.maze_size, self.maze_size)) < wall_density).astype(int)
        
        # Ensure connected paths
        maze[self.maze_size // 2, :] = 0  # Horizontal path
        maze[:, self.maze_size // 2] = 0  # Vertical path
        
        return maze
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (x, y) position to state index."""
        x, y = pos
        return x * self.maze_size + y
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to (x, y) position."""
        x = state // self.maze_size
        y = state % self.maze_size
        return (x, y)
    
    def reset(self) -> int:
        """Reset environment to start position."""
        self.agent_pos = self.start_pos
        self.steps = 0
        return self._pos_to_state(self.agent_pos)
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """
        Execute one action in the maze.
        
        Args:
            action: 0=up, 1=down, 2=left, 3=right
            
        Returns:
            state: New state as integer
            reward: -1 for step, +10 for reaching goal, -2 for hitting wall
            terminated: Whether goal reached
            truncated: Whether max steps exceeded
            info: Additional information
        """
        self.steps += 1
        
        # Calculate new position
        x, y = self.agent_pos
        
        if action == 0:      # up
            new_x, new_y = x - 1, y
        elif action == 1:    # down
            new_x, new_y = x + 1, y
        elif action == 2:    # left
            new_x, new_y = x, y - 1
        else:                # right
            new_x, new_y = x, y + 1
        
        # Check bounds
        if new_x < 0 or new_x >= self.maze_size or new_y < 0 or new_y >= self.maze_size:
            reward = -2  # Boundary penalty
            new_x, new_y = x, y
        # Check wall
        elif self.maze[new_x, new_y] == 1:
            reward = -2  # Wall penalty
            new_x, new_y = x, y
        else:
            reward = -1  # Normal step
        
        self.agent_pos = (new_x, new_y)
        
        # Check goal
        terminated = (self.agent_pos == self.goal_pos)
        if terminated:
            reward = +10  # Goal reward
        
        # Check max steps
        truncated = self.steps >= self.max_steps
        
        return self._pos_to_state(self.agent_pos), reward, terminated, truncated, {'steps': self.steps}
    
    def get_valid_actions(self, state: int) -> List[int]:
        """Get list of valid actions from a state."""
        x, y = self._state_to_pos(state)
        valid_actions = []
        
        # Check each direction: up, down, left, right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for action, (dx, dy) in enumerate(moves):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.maze_size and 0 <= new_y < self.maze_size:
                if self.maze[new_x, new_y] == 0:
                    valid_actions.append(action)
        
        return valid_actions if valid_actions else [0]  # Return at least one action
    
    def get_state_info(self, state: int) -> dict:
        """Get information about a state."""
        pos = self._state_to_pos(state)
        return {
            'position': pos,
            'is_goal': pos == self.goal_pos,
            'is_start': pos == self.start_pos
        }
    
    def render(self):
        """
        Visualize the maze using matplotlib.
        Shows the current state with walls, free spaces, agent, and goal.
        """
        if self.fig is None:
            # Create figure and axis
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.fig.suptitle(f'Maze Environment ({self.maze_size}x{self.maze_size})', fontsize=14)
        
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.maze_size - 0.5)
        self.ax.set_ylim(-0.5, self.maze_size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.set_title(f'Step: {self.steps}, Difficulty: {self.difficulty}')
        
        # Draw grid background
        for i in range(self.maze_size + 1):
            self.ax.axhline(y=i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            self.ax.axvline(x=i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        
        # Draw walls (black)
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if self.maze[i, j] == 1:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                            linewidth=0, facecolor='black')
                    self.ax.add_patch(rect)
        
        # Draw goal (green)
        gx, gy = self.goal_pos
        rect = patches.Rectangle((gy - 0.5, gx - 0.5), 1, 1,
                                linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.8)
        self.ax.add_patch(rect)
        self.ax.text(gy, gx, 'G', ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Draw agent (red circle)
        ax, ay = self.agent_pos
        circle = patches.Circle((ay, ax), 0.3, color='red', zorder=10)
        self.ax.add_patch(circle)
        
        self.ax.set_xticks(range(0, self.maze_size))
        self.ax.set_yticks(range(0, self.maze_size))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        plt.tight_layout()
        plt.pause(self.render_delay)
    
    def close(self):
        """Clean up environment."""
        if self.fig is not None:
            plt.close(self.fig)


def create_maze_env(maze_size: int = 8, difficulty: str = 'medium') -> MazeEnv:
    """Factory function to create maze environment."""
    return MazeEnv(maze_size, difficulty)
