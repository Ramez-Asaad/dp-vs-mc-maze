"""
Task 1: Dynamic Programming
Main script to implement and analyze DP algorithms on Custom Maze environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from environment_setup import create_maze_env
from dp_algorithms import DynamicProgramming

# Global visualization window that stays open
viz_env = None
viz_fig = None
persistent_window = True


def visualize_policy_execution(policy, title="Policy Execution", render_delay=0.05):
    """
    Visualize the agent executing a learned policy in the persistent window.
    
    Args:
        policy: Policy array mapping states to actions
        title: Title for the visualization
        render_delay: Delay between steps in seconds
    """
    global viz_env
    
    # Use the same environment for consistent maze
    if viz_env is None:
        viz_env = create_maze_env(maze_size=16, difficulty='medium')
    
    state = viz_env.reset()
    total_reward = 0
    steps_taken = 0
    
    # Set custom render delay
    original_delay = viz_env.render_delay
    viz_env.render_delay = render_delay
    
    # Update window title
    if viz_env.ax is not None:
        viz_env.fig.suptitle(title, fontsize=14)
    
    while True:
        # Render current state
        viz_env.render()
        
        # Execute policy action
        action = policy[state]
        state, reward, terminated, truncated, info = viz_env.step(action)
        total_reward += reward
        steps_taken += 1
        
        if terminated:
            viz_env.render()
            plt.pause(0.5)
            return total_reward, steps_taken
        
        if truncated:
            viz_env.render()
            plt.pause(0.5)
            return total_reward, steps_taken
    
    # Restore original delay
    viz_env.render_delay = original_delay


def visualize_maze_exploration():
    """
    Visualize the maze environment with the agent navigating using a random policy.
    Shows the maze in a window with the agent moving in real-time.
    """
    global viz_env
    
    print("\n" + "="*70)
    print("MAZE ENVIRONMENT VISUALIZATION")
    print("="*70)
    print("\nShowing maze with agent navigating randomly...")
    print("(Window will stay open for all experiments)\n")
    
    # Create persistent visualization environment
    if viz_env is None:
        viz_env = create_maze_env(maze_size=16, difficulty='medium')
    
    print("Maze Statistics:")
    print(f"  Grid size: {viz_env.maze_size}x{viz_env.maze_size}")
    print(f"  Total states: {viz_env.num_states}")
    print(f"  Total actions: {viz_env.num_actions} (up, down, left, right)")
    
    wall_count = np.sum(viz_env.maze)
    free_count = viz_env.num_states - wall_count
    print(f"  Free spaces: {free_count}")
    print(f"  Walls: {wall_count}")
    print(f"  Wall density: {wall_count/viz_env.num_states*100:.1f}%")
    print(f"  Start position: {viz_env.start_pos}")
    print(f"  Goal position: {viz_env.goal_pos}")
    
    # Run episode with visualization
    print("\nRunning random exploration episode...\n")
    state = viz_env.reset()
    total_reward = 0
    
    while True:
        # Render current state
        viz_env.render()
        
        # Choose random action
        valid_actions = viz_env.get_valid_actions(state)
        action = np.random.choice(valid_actions)
        
        # Take action
        state, reward, terminated, truncated, info = viz_env.step(action)
        total_reward += reward
        
        if terminated:
            viz_env.render()
            print(f"\n[OK] Goal reached! Total reward: {total_reward:.0f}")
            plt.pause(0.5)
            break
        
        if truncated:
            viz_env.render()
            print(f"\n[FAIL] Max steps exceeded. Total reward: {total_reward:.0f}")
            plt.pause(0.5)
            break
    
    print("\nWindow will remain open for all experiments...")
    time.sleep(0.5)


def evaluate_policy(env, policy, num_episodes: int = 10000, max_steps: int = 1500) -> float:
    """
    Evaluate a policy by running episodes and returning average reward.
    
    Args:
        env: Environment
        policy: Policy to evaluate (array mapping state to action)
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        average_reward: Average reward across episodes
    """
    total_reward = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for _ in range(max_steps):
            action = policy[state]
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
            
            state = next_state
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


def experiment_discount_factors():
    """
    Experiment 1: Analyze impact of discount factors on convergence.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Impact of Discount Factors on Convergence (Maze)")
    print("="*70)
    
    discount_factors = [0.5, 0.7, 0.9, 0.95, 0.99]
    results = {}
    
    # Use the same environment for all gamma experiments
    # This ensures all algorithms learn from identical maze structure
    # allowing fair comparison of how discount factor affects learning
    global viz_env
    if viz_env is None:
        viz_env = create_maze_env(maze_size=16, difficulty='medium')
    env = viz_env
    
    for gamma in discount_factors:
        print(f"\n--- Testing gamma = {gamma} ---")
        
        # Use SAME environment for all gammas
        dp = DynamicProgramming(env, gamma=gamma)
        
        # Build transition model with more episodes for better statistics
        print(f"  Building transition model (500 episodes)...")
        model = dp.build_transition_model(num_episodes=500)
        
        # Run value iteration
        print(f"  Running value iteration (1000 iterations)...")
        policy, V = dp.value_iteration(model, theta=1e-6, max_iterations=1000)
        
        # Debug: show policy statistics
        unique_actions = len(np.unique(policy))
        print(f"  Policy has {unique_actions} unique actions (should be > 1 for good policy)")
        
        # Visualize the learned policy using the SAME environment
        print(f"  Visualizing learned policy for gamma={gamma}...")
        state = env.reset()
        total_reward = 0
        steps_taken = 0
        
        while True:
            env.render()
            action = policy[state]
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps_taken += 1
            
            if terminated or truncated:
                env.render()
                plt.pause(0.3)
                break
        
        print(f"  [OK] Policy executed: Reward={total_reward:.0f}, Steps={steps_taken}")
        
        # Evaluate policy statistically
        print(f"  Evaluating policy (100 episodes)...")
        avg_reward = evaluate_policy(env, policy, num_episodes=100)
        
        results[gamma] = {
            'policy': policy,
            'value_function': V,
            'avg_reward': avg_reward
        }
        
        print(f"  [OK] Average reward with gamma={gamma}: {avg_reward:.2f}")
        time.sleep(0.5)  # Brief pause for readability
    
    # Plot convergence for different discount factors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Average reward vs discount factor
    gammas = list(results.keys())
    rewards = [results[g]['avg_reward'] for g in gammas]
    
    axes[0].plot(gammas, rewards, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Discount Factor (γ)', fontsize=12)
    axes[0].set_ylabel('Average Reward', fontsize=12)
    axes[0].set_title('Impact of Discount Factor on Policy Performance', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Value function distribution for different gammas
    for gamma in discount_factors:
        V_flat = np.sort(results[gamma]['value_function'])
        axes[1].hist(V_flat, bins=30, alpha=0.5, label=f'γ={gamma}')
    
    axes[1].set_xlabel('Value Function', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Value Functions', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task1_discount_factor_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task1_discount_factor_analysis.png'")
    plt.close()
    
    return results


def compare_algorithms():
    """
    Experiment 2: Compare Policy Iteration vs Value Iteration.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Comparing Policy Iteration vs Value Iteration (Maze)")
    print("="*70)
    
    # Use persistent visualization environment
    env = viz_env if viz_env is not None else create_maze_env(maze_size=16, difficulty='medium')
    
    dp_pi = DynamicProgramming(env, gamma=0.99)
    dp_vi = DynamicProgramming(env, gamma=0.99)
    
    # Build transition model with more episodes
    print("\nBuilding transition model (500 episodes)...")
    model = dp_pi.build_transition_model(num_episodes=500)
    
    # Run Policy Iteration
    print("Running Policy Iteration (50 iterations)...")
    policy_pi, V_pi = dp_pi.policy_iteration(model, max_iterations=50)
    print(f"  Visualizing Policy Iteration result...")
    reward_pi, steps_pi = visualize_policy_execution(policy_pi, 
                                                      title=f"Policy Iteration (Learning...)",
                                                      render_delay=0.05)
    print(f"  [OK] Policy Iteration executed: Reward={reward_pi:.0f}, Steps={steps_pi}")
    avg_reward_pi = evaluate_policy(env, policy_pi, num_episodes=100)
    print(f"  [OK] Policy Iteration - Average reward: {avg_reward_pi:.2f}")
    time.sleep(0.5)
    
    # Run Value Iteration
    print("Running Value Iteration (500 iterations)...")
    policy_vi, V_vi = dp_vi.value_iteration(model, theta=1e-6, max_iterations=500)
    print(f"  Visualizing Value Iteration result...")
    reward_vi, steps_vi = visualize_policy_execution(policy_vi, 
                                                      title=f"Value Iteration (Learning...)",
                                                      render_delay=0.05)
    print(f"  [OK] Value Iteration executed: Reward={reward_vi:.0f}, Steps={steps_vi}")
    avg_reward_vi = evaluate_policy(env, policy_vi, num_episodes=100)
    print(f"  [OK] Value Iteration - Average reward: {avg_reward_vi:.2f}")
    time.sleep(0.5)
    
    print(f"\nComparison Summary:")
    print(f"  Policy Iteration - Average Reward: {avg_reward_pi:.2f}")
    print(f"  Value Iteration - Average Reward: {avg_reward_vi:.2f}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Value function comparison
    states = np.arange(len(V_pi))
    axes[0, 0].scatter(states, V_pi, alpha=0.5, s=10, label='Policy Iteration', color='blue')
    axes[0, 0].scatter(states, V_vi, alpha=0.5, s=10, label='Value Iteration', color='red')
    axes[0, 0].set_xlabel('State', fontsize=11)
    axes[0, 0].set_ylabel('Value', fontsize=11)
    axes[0, 0].set_title('Value Function Comparison', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of value differences
    value_diff = np.abs(V_pi - V_vi)
    axes[0, 1].hist(value_diff, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Absolute Difference in Values', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title(f'Value Function Differences (Mean: {np.mean(value_diff):.4f})', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Policy comparison
    policy_agreement = (policy_pi == policy_vi).astype(int)
    policy_match_rate = np.mean(policy_agreement) * 100
    
    axes[1, 0].bar(['Agree', 'Disagree'], 
                   [policy_match_rate, 100 - policy_match_rate],
                   color=['green', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel('Percentage (%)', fontsize=11)
    axes[1, 0].set_title(f'Policy Agreement: {policy_match_rate:.1f}%', fontsize=12)
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Algorithm performance summary
    algorithms = ['Policy\nIteration', 'Value\nIteration']
    rewards = [avg_reward_pi, avg_reward_vi]
    colors = ['blue', 'red']
    
    axes[1, 1].bar(algorithms, rewards, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Average Reward', fontsize=11)
    axes[1, 1].set_title('Algorithm Performance Comparison', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (alg, reward) in enumerate(zip(algorithms, rewards)):
        axes[1, 1].text(i, reward + 5, f'{reward:.1f}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('task1_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task1_algorithm_comparison.png'")
    plt.close()
    
    # Don't close the persistent visualization environment
    
    return {'PI': avg_reward_pi, 'VI': avg_reward_vi}


def convergence_analysis():
    """
    Experiment 3: Detailed convergence analysis.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Convergence Analysis (Maze)")
    print("="*70)
    
    env = viz_env if viz_env is not None else create_maze_env(maze_size=16, difficulty='medium')
    dp = DynamicProgramming(env, gamma=0.99)
    
    print("\nBuilding transition model (500 episodes)...")
    model = dp.build_transition_model(num_episodes=500)
    
    print("Running convergence analysis (500 iterations)...")
    time.sleep(0.5)
    
    # Value iteration with tracking
    P, R = model['P'], model['R']
    max_value_diff = []
    
    for iteration in range(500):
        V_old = dp.V.copy()
        
        # Value iteration step
        V_new = np.zeros(dp.num_states)
        for s in range(dp.num_states):
            valid_actions = env.get_valid_actions(s)
            action_values = np.zeros(dp.num_actions)
            for a in range(dp.num_actions):
                action_values[a] = R[s, a] + dp.gamma * np.dot(P[s, a, :], dp.V)
            # Only consider valid actions
            invalid_actions = np.setdiff1d(np.arange(dp.num_actions), valid_actions)
            action_values[invalid_actions] = -np.inf
            V_new[s] = np.max(action_values)
        
        dp.V = V_new
        max_diff = np.max(np.abs(dp.V - V_old))
        max_value_diff.append(max_diff)
        
        if (iteration + 1) % 50 == 0:
            print(f"  Iteration {iteration+1}: max diff = {max_diff:.2e}")
    
    print("[OK] Convergence analysis complete")
    
    # Extract final policy and visualize
    print("Visualizing final converged policy...")
    # Compute Q-values for all state-action pairs, respecting valid actions
    Q = np.zeros((dp.num_states, dp.num_actions))
    for s in range(dp.num_states):
        valid_actions = env.get_valid_actions(s)
        for a in range(dp.num_actions):
            Q[s, a] = R[s, a] + dp.gamma * np.dot(model['P'][s, a, :], dp.V)
        # Only consider valid actions
        invalid_actions = np.setdiff1d(np.arange(dp.num_actions), valid_actions)
        Q[s, invalid_actions] = -np.inf
    # Extract policy as argmax of Q-values (only from valid actions)
    final_policy = np.argmax(Q, axis=1)
    reward, steps = visualize_policy_execution(final_policy, 
                                                title=f"Converged Policy (Final Result)",
                                                render_delay=0.05)
    print(f"  [OK] Final policy executed: Reward={reward:.0f}, Steps={steps}")
    time.sleep(0.5)
    
    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].plot(max_value_diff, 'b-', linewidth=2)
    axes[0].axhline(y=1e-6, color='r', linestyle='--', label='Convergence threshold (1e-6)')
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Max Value Difference', fontsize=12)
    axes[0].set_title('Value Iteration Convergence (Linear Scale)', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Log scale
    axes[1].semilogy(max_value_diff, 'b-', linewidth=2)
    axes[1].axhline(y=1e-6, color='r', linestyle='--', label='Convergence threshold (1e-6)')
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Max Value Difference (log scale)', fontsize=12)
    axes[1].set_title('Value Iteration Convergence (Log Scale)', fontsize=13)
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('task1_convergence_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task1_convergence_analysis.png'")
    plt.close()


def main():
    """Main execution function."""
    global viz_env
    
    print("\n" + "="*70)
    print("TASK 1: DYNAMIC PROGRAMMING")
    print("="*70)
    print("This task implements and analyzes Dynamic Programming algorithms")
    print("(Policy Iteration and Value Iteration) on a Custom Maze environment")
    print("\nNote: A persistent visualization window will show the agent's behavior")
    print("as the algorithms improve throughout the experiments.\n")
    
    # First: Visualize the maze environment
    visualize_maze_exploration()
    
    # Run experiments
    print("\nStarting experiments...\n")
    experiment_discount_factors()
    compare_algorithms()
    convergence_analysis()
    
    print("\n" + "="*70)
    print("TASK 1 COMPLETE")
    print("="*70)
    print("Generated plots:")
    print("  - task1_discount_factor_analysis.png")
    print("  - task1_algorithm_comparison.png")
    print("  - task1_convergence_analysis.png")
    
    # Clean up visualization
    if viz_env is not None:
        viz_env.close()
    plt.close('all')
    
    print("\nVisualization window closed.")


if __name__ == "__main__":
    main()
