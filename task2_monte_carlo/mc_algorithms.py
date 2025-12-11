"""
Monte Carlo methods for reinforcement learning on Maze environments.
Implements: First-visit Monte Carlo Policy Evaluation with epsilon-greedy exploration.
"""

import numpy as np
from typing import Tuple, Dict, List


class MonteCarloAgent:
    """
    Monte Carlo agent for policy evaluation and improvement on Maze.
    Uses first-visit Monte Carlo to estimate value function.
    """
    
    def __init__(self, env, gamma: float = 0.99, epsilon: float = 0.1):
        """
        Initialize Monte Carlo agent.
        
        Args:
            env: Maze environment
            gamma: Discount factor
            epsilon: Exploration rate for epsilon-greedy
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Environment properties
        self.num_actions = env.num_actions
        self.num_states = env.num_states
        
        # Initialize value functions
        self.V = {}  # State value function (state -> scalar)
        self.Q = {}  # Action value function (state, action) -> scalar
        self.returns = {}  # Track returns for (state, action) pairs
        self.policy = {}   # Greedy policy (state -> best action)
        
        # For tracking analysis
        self.episode_rewards = []
        self.visit_counts = {}
        self.path_lengths = []
        
    def policy_action(self, state: int) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
        """
        # Epsilon-greedy: random with probability epsilon, greedy otherwise
        if np.random.random() < self.epsilon:
            # Exploration: choose random valid action
            valid_actions = self.env.get_valid_actions(state)
            return np.random.choice(valid_actions)
        else:
            # Exploitation: use greedy policy if available
            if state in self.policy:
                return self.policy[state]
            else:
                # If state not yet visited, choose random valid action
                valid_actions = self.env.get_valid_actions(state)
                return np.random.choice(valid_actions)
    
    def generate_episode(self, max_steps: int = 500) -> Tuple[List[Tuple], float]:
        """
        Generate one episode following epsilon-greedy policy.
        
        Args:
            max_steps: Maximum steps in episode
            
        Returns:
            episode: List of (state, action, reward) tuples
            episode_reward: Total reward in episode
        """
        episode = []
        state = self.env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = self.policy_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode_reward += reward
            
            # Store transition
            episode.append((state, action, reward))
            
            if terminated or truncated:
                break
            
            state = next_state
        
        self.path_lengths.append(len(episode))
        return episode, episode_reward
    
    def policy_evaluation(self, num_episodes: int = 5000, max_steps: int = 100) -> Dict:
        """
        Monte Carlo Policy Evaluation using first-visit method.
        Estimates Q(s,a) and improves policy greedily.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            
        Returns:
            stats: Dictionary with evaluation statistics
        """
        print(f"\n{'='*60}")
        print(f"MONTE CARLO POLICY EVALUATION")
        print(f"{'='*60}")
        print(f"Number of episodes: {num_episodes}")
        print(f"Gamma: {self.gamma}, Epsilon: {self.epsilon}")
        
        self.V = {}
        self.Q = {}
        self.returns = {}
        self.episode_rewards = []
        self.visit_counts = {}
        
        for episode_num in range(num_episodes):
            # Generate episode
            episode, episode_reward = self.generate_episode(max_steps)
            self.episode_rewards.append(episode_reward)
            
            # First-visit Monte Carlo: count each (state, action) pair only once per episode
            visited_pairs = set()
            G = 0  # Return (cumulative discounted reward)
            
            # Process episode in reverse (from end to start)
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                # First-visit: only update if (state, action) pair not visited earlier in episode
                pair = (state, action)
                if pair not in visited_pairs:
                    visited_pairs.add(pair)
                    
                    # Update Q-value (action-value function)
                    if pair not in self.returns:
                        self.returns[pair] = []
                        self.Q[pair] = 0.0
                    
                    self.returns[pair].append(G)
                    self.Q[pair] = np.mean(self.returns[pair])
                    
                    # Update state value as max over actions
                    if state not in self.V:
                        self.V[state] = -np.inf
                        self.visit_counts[state] = 0
                    
                    # V(s) = max_a Q(s,a)
                    valid_actions = self.env.get_valid_actions(state)
                    q_values = [self.Q.get((state, a), 0) for a in valid_actions]
                    self.V[state] = max(q_values) if q_values else 0
                    self.visit_counts[state] += 1
            
            # Improve policy greedily based on Q-values
            for state in self.visit_counts.keys():
                valid_actions = self.env.get_valid_actions(state)
                best_action = valid_actions[0]
                best_q = self.Q.get((state, best_action), 0)
                
                for action in valid_actions:
                    q = self.Q.get((state, action), 0)
                    if q > best_q:
                        best_q = q
                        best_action = action
                
                self.policy[state] = best_action
            
            # Print progress
            if (episode_num + 1) % (num_episodes // 10) == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode_num + 1}/{num_episodes} - "
                      f"Avg Reward (last 100): {avg_reward:.3f} - "
                      f"Unique states: {len(self.V)}")
        
        # Compute statistics
        stats = {
            'num_episodes': num_episodes,
            'num_states_visited': len(self.V),
            'avg_final_reward': np.mean(self.episode_rewards[-100:]),
            'std_final_reward': np.std(self.episode_rewards[-100:]),
            'value_function': self.V,
            'visit_counts': self.visit_counts
        }
        
        print(f"\nEvaluation complete!")
        print(f"States visited: {stats['num_states_visited']}")
        print(f"Final avg reward: {stats['avg_final_reward']:.3f}")
        
        return stats
    
    def policy_evaluation_with_exploration(self, num_episodes: int = 5000) -> Dict:
        """
        Enhanced policy evaluation tracking exploration impact.
        
        Args:
            num_episodes: Number of episodes
            
        Returns:
            stats: Dictionary with analysis statistics
        """
        print(f"\n{'='*60}")
        print(f"MONTE CARLO - EXPLORATION IMPACT ANALYSIS")
        print(f"{'='*60}")
        
        exploration_rates = []
        rewards_per_epsilon = {}
        
        for episode_num in range(num_episodes):
            episode, episode_reward = self.generate_episode()
            self.episode_rewards.append(episode_reward)
            
            # Count exploratory actions (random actions)
            exploration_count = sum(1 for _, action, _ in episode 
                                  if np.random.random() < self.epsilon)
            exploration_rates.append(exploration_count / len(episode) if episode else 0)
            
            # First-visit Monte Carlo update
            visited_states = set()
            G = 0
            
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = reward + self.gamma * G
                
                if state not in visited_states:
                    visited_states.add(state)
                    if state not in self.returns:
                        self.returns[state] = []
                        self.V[state] = 0.0
                    
                    self.returns[state].append(G)
                    self.V[state] = np.mean(self.returns[state])
        
        stats = {
            'episode_rewards': self.episode_rewards,
            'exploration_rates': exploration_rates,
            'num_states': len(self.V),
            'final_value_stats': {
                'mean': np.mean(list(self.V.values())),
                'std': np.std(list(self.V.values())),
                'min': np.min(list(self.V.values())),
                'max': np.max(list(self.V.values()))
            }
        }
        
        return stats
    
    def get_value_function(self) -> Dict:
        """Return estimated value function."""
        return self.V
    
    def get_policy(self) -> Dict:
        """Return current policy."""
        return self.policy
    
    def evaluate_policy(self, num_eval_episodes: int = 100) -> float:
        """
        Evaluate learned policy by running episodes without exploration.
        
        Args:
            num_eval_episodes: Number of evaluation episodes
            
        Returns:
            avg_reward: Average reward in evaluation
        """
        total_reward = 0
        old_epsilon = self.epsilon
        self.epsilon = 0  # No exploration in evaluation
        
        for _ in range(num_eval_episodes):
            _, episode_reward = self.generate_episode()
            total_reward += episode_reward
        
        self.epsilon = old_epsilon
        avg_reward = total_reward / num_eval_episodes
        
        return avg_reward
