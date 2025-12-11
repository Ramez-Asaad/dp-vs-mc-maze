"""
Custom Maze Environment for Dynamic Programming task.
A simple 2D grid-based maze where agent must navigate from start to goal.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Dict, List
import time


class MazeEnv:
    """
    Simple grid-based maze environment.
    Agent navigates from start to goal position, avoiding walls.
    
    Discrete state space: position (x, y) → unique state index
    Discrete actions: 0=up, 1=down, 2=left, 3=right
    """
    
    def __init__(self, maze_size: int = 25, difficulty: str = 'medium'):
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
        self.max_steps = maze_size * maze_size * 5  # Increased from 2x to 5x to allow goal-reaching
        
        # Visualization
        self.fig = None
        self.ax = None
        self.render_delay = 0.1  # seconds per frame
        
    def _generate_maze(self, difficulty: str) -> np.ndarray:
        """
        Generate a RANDOM maze with guaranteed start->goal connectivity.
        Each call generates a different maze layout.
        
        The maze structure varies but always has:
        - A clear path from start (0,0) to goal (7,7)
        - Multiple possible routes to create interesting learning scenarios
        
        Args:
            difficulty: Controls wall density
            
        Returns:
            maze: 2D array where 0=free, 1=wall
        """
        if difficulty == 'easy':
            wall_density = 0.15
        elif difficulty == 'medium':
            wall_density = 0.25
        else:  # hard
            wall_density = 0.40
        
        # Generate completely random maze with specified wall density
        # Use np.random.random() which generates different values each time
        maze = (np.random.random((self.maze_size, self.maze_size)) < wall_density).astype(int)
        
        # Guarantee reachability: create two mandatory clear paths
        # PATH 1: Horizontal corridor (flexible path)
        maze[0, :] = 0  # Top row always clear
        
        # PATH 2: Vertical corridor (alternative path)
        maze[:, 0] = 0  # Left column always clear
        maze[:, self.maze_size - 1] = 0  # Right column always clear
        maze[self.maze_size - 1, :] = 0  # Bottom row always clear
        
        # Ensure start and goal are clear
        maze[0, 0] = 0
        maze[self.maze_size - 1, self.maze_size - 1] = 0
        
        # Ensure neighbors of start and goal are accessible
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = 0 + dx, 0 + dy
                if 0 <= nx < self.maze_size and 0 <= ny < self.maze_size:
                    maze[nx, ny] = 0
                
                nx, ny = (self.maze_size - 1) + dx, (self.maze_size - 1) + dy
                if 0 <= nx < self.maze_size and 0 <= ny < self.maze_size:
                    maze[nx, ny] = 0
        
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
            reward: -1 for step, +10 for reaching goal, -5 for hitting wall
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
        
        # Default: movement cost
        reward = -1.0
        
        # Check bounds
        if new_x < 0 or new_x >= self.maze_size or new_y < 0 or new_y >= self.maze_size:
            # Hit boundary, stay in place
            reward = -5
            new_x, new_y = x, y
        # Check wall
        elif self.maze[new_x, new_y] == 1:
            # Hit wall, stay in place and don't get any bonuses
            reward = -1
            new_x, new_y = x, y
        else:
            # Valid move to free space
            reward = -1.0  # Base step cost
            
            # PATH-SPECIFIC REWARDS: Create competing incentives for different discount factors
            # These bonuses are subtle enough that reaching the goal (+10) is still better
            # But they make different gammas favor different paths
            
            if new_x == 0 or new_y == self.maze_size - 1:
                # Border cells: +0.5 per step (encourages border path for low gamma)
                reward += 0.5
            elif new_x == new_y:
                # Diagonal cells: +0.8 per step (encourages diagonal for high gamma)
                reward += 0.8
        
        self.agent_pos = (new_x, new_y)
        
        # Check goal
        terminated = (self.agent_pos == self.goal_pos)
        if terminated:
            reward = +10  # Goal bonus (overrides path rewards)
        
        # Check max steps
        truncated = self.steps >= self.max_steps
        
        return self._pos_to_state(self.agent_pos), reward, terminated, truncated, {'steps': self.steps}
    
    def get_valid_actions(self, state: int) -> List[int]:
        """Get list of valid actions from a state (for DP planning)."""
        x, y = self._state_to_pos(state)
        valid_actions = []
        
        # Check each direction
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for action, (dx, dy) in enumerate(moves):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.maze_size and 0 <= new_y < self.maze_size:
                if self.maze[new_x, new_y] == 0:
                    valid_actions.append(action)
        
        return valid_actions if valid_actions else [0]  # Return at least one action
    
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
