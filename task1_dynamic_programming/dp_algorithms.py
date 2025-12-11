"""
Dynamic Programming algorithms for solving MDPs.
Implements: Policy Evaluation, Value Iteration, and Policy Iteration.
"""

import numpy as np
from typing import Tuple, Dict, List
try:
    from .environment_setup import MazeEnv
except ImportError:
    from environment_setup import MazeEnv


class DynamicProgramming:
    """
    Dynamic Programming solver for finite MDPs.
    Implements policy evaluation, value iteration, and policy iteration algorithms.
    """
    
    def __init__(self, env: MazeEnv, gamma: float = 0.99):
        """
        Initialize the DP solver.
        
        Args:
            env: Discretized environment
            gamma: Discount factor (default: 0.99)
        """
        self.env = env
        self.gamma = gamma
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        
        # Initialize value function and policy
        self.V = np.zeros(self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)  # Initialize with action 0
        
        # For tracking convergence
        self.convergence_history = []
        
    def build_transition_model(self, num_episodes: int = 500) -> Dict:
        """
        Build empirical transition model P(s'|s,a) and expected rewards from environment exploration.
        
        Args:
            num_episodes: Number of random episodes to collect statistics
            
        Returns:
            Dictionary with transition probabilities and expected rewards
        """
        print(f"Building transition model with {num_episodes} random episodes...")
        
        # Count transitions: [state, action, next_state]
        transition_counts = np.zeros((self.num_states, self.num_actions, self.num_states))
        # Sum rewards for each transition
        reward_sum = np.zeros((self.num_states, self.num_actions, self.num_states))
        # Count occurrences of (s, a)
        state_action_counts = np.zeros((self.num_states, self.num_actions))
        
        visited_states = set()
        total_transitions = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                visited_states.add(state)
                # Only try valid actions for maze
                valid_actions = self.env.get_valid_actions(state)
                action = np.random.choice(valid_actions)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update statistics - use the actual reward from step()
                transition_counts[state, action, next_state] += 1
                reward_sum[state, action, next_state] += reward  # This now includes goal reward!
                state_action_counts[state, action] += 1
                total_transitions += 1
                
                state = next_state
        
        # Compute transition probabilities and expected rewards
        P = np.zeros((self.num_states, self.num_actions, self.num_states))
        R = np.zeros((self.num_states, self.num_actions))
        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if state_action_counts[s, a] > 0:
                    # Transition probabilities
                    P[s, a, :] = transition_counts[s, a, :] / state_action_counts[s, a]
                    # Expected reward: sum of rewards for each (s,a,s') weighted by transition probability
                    total_reward = 0
                    for s_next in range(self.num_states):
                        if transition_counts[s, a, s_next] > 0:
                            # Average reward for transitions to s_next
                            avg_reward = reward_sum[s, a, s_next] / transition_counts[s, a, s_next]
                            total_reward += P[s, a, s_next] * avg_reward
                    R[s, a] = total_reward
        
        print(f"Transition model built successfully!")
        print(f"  Visited {len(visited_states)}/{self.num_states} states ({100*len(visited_states)/self.num_states:.1f}%)")
        print(f"  Total transitions: {total_transitions}")
        return {'P': P, 'R': R}
    
    def policy_evaluation(self, model: Dict, theta: float = 1e-6, max_iterations: int = 1000) -> float:
        """
        Evaluate the current policy using iterative method.
        
        Args:
            model: Transition model with 'P' (probabilities) and 'R' (rewards)
            theta: Convergence threshold
            max_iterations: Maximum number of iterations
            
        Returns:
            delta: Final change in value function
        """
        P, R = model['P'], model['R']
        delta_history = []
        
        for iteration in range(max_iterations):
            delta = 0
            V_new = np.zeros(self.num_states)
            
            for s in range(self.num_states):
                action = self.policy[s]
                # Bellman expectation equation
                value = R[s, action] + self.gamma * np.dot(P[s, action, :], self.V)
                V_new[s] = value
                delta = max(delta, abs(value - self.V[s]))
            
            self.V = V_new
            delta_history.append(delta)
            
            if delta < theta:
                print(f"Policy Evaluation converged in {iteration + 1} iterations")
                return delta
        
        print(f"Policy Evaluation max iterations ({max_iterations}) reached")
        return delta
    
    def policy_improvement(self, model: Dict) -> bool:
        """
        Improve policy based on current value function.
        
        Args:
            model: Transition model with 'P' (probabilities) and 'R' (rewards)
            
        Returns:
            policy_stable: Whether policy has converged
        """
        P, R = model['P'], model['R']
        policy_stable = True
        
        for s in range(self.num_states):
            old_action = self.policy[s]
            valid_actions = self.env.get_valid_actions(s)
            
            # Compute value of each action
            action_values = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                action_values[a] = R[s, a] + self.gamma * np.dot(P[s, a, :], self.V)
            
            # Select best action among ONLY valid actions
            action_values[np.setdiff1d(np.arange(self.num_actions), valid_actions)] = -np.inf
            new_action = np.argmax(action_values)
            self.policy[s] = new_action
            
            if old_action != new_action:
                policy_stable = False
        
        return policy_stable
    
    def policy_iteration(self, model: Dict, max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Policy Iteration: Alternate between evaluation and improvement until convergence.
        
        Args:
            model: Transition model
            max_iterations: Maximum iterations of policy improvement
            
        Returns:
            policy: Optimal policy
            V: Value function
        """
        print("\n" + "="*60)
        print("POLICY ITERATION")
        print("="*60)
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}")
            
            # Policy Evaluation
            self.policy_evaluation(model, theta=1e-6, max_iterations=100)
            
            # Policy Improvement
            policy_stable = self.policy_improvement(model)
            
            if policy_stable:
                print(f"\nPolicy Iteration converged in {iteration + 1} iterations!")
                break
        
        self.convergence_history.append(('Policy Iteration', self.V.copy()))
        return self.policy, self.V
    
    def value_iteration(self, model: Dict, theta: float = 1e-6, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Value Iteration: Combine evaluation and improvement in single step.
        
        Args:
            model: Transition model
            theta: Convergence threshold
            max_iterations: Maximum iterations
            
        Returns:
            policy: Optimal policy
            V: Value function
        """
        print("\n" + "="*60)
        print("VALUE ITERATION")
        print("="*60)
        
        P, R = model['P'], model['R']
        delta_history = []
        
        for iteration in range(max_iterations):
            delta = 0
            V_new = np.zeros(self.num_states)
            
            for s in range(self.num_states):
                # Only consider valid actions for this state
                valid_actions = self.env.get_valid_actions(s)
                
                # Compute value of each valid action
                action_values = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    action_values[a] = R[s, a] + self.gamma * np.dot(P[s, a, :], self.V)
                
                # Take max over ONLY valid actions
                action_values[np.setdiff1d(np.arange(self.num_actions), valid_actions)] = -np.inf
                V_new[s] = np.max(action_values)
                delta = max(delta, abs(V_new[s] - self.V[s]))
            
            self.V = V_new
            delta_history.append(delta)
            
            if (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}, Delta: {delta:.2e}")
            
            if delta < theta:
                print(f"\nValue Iteration converged in {iteration + 1} iterations!")
                break
        
        # Extract policy from value function
        for s in range(self.num_states):
            valid_actions = self.env.get_valid_actions(s)
            action_values = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                action_values[a] = R[s, a] + self.gamma * np.dot(P[s, a, :], self.V)
            # Only consider valid actions
            action_values[np.setdiff1d(np.arange(self.num_actions), valid_actions)] = -np.inf
            self.policy[s] = np.argmax(action_values)
        
        self.convergence_history.append(('Value Iteration', self.V.copy(), delta_history))
        return self.policy, self.V
    
    def get_convergence_history(self):
        """Return convergence history for analysis."""
        return self.convergence_history
