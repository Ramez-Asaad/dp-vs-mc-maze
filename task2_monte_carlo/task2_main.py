"""
Task 2: Monte Carlo Methods
Main script implementing Monte Carlo Policy Evaluation on Maze environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from environment_setup import create_maze_env
from mc_algorithms import MonteCarloAgent

# Global visualization window that stays open
viz_env = None
viz_fig = None
persistent_window = True


def visualize_policy_execution(policy, title="Policy Execution", render_delay=0.05):
    """
    Visualize the agent executing a learned policy in the persistent window.
    
    Args:
        policy: Dictionary mapping states to actions
        title: Title for the visualization
        render_delay: Delay between steps in seconds
    """
    global viz_env
    
    # Use the same environment for consistent maze
    if viz_env is None:
        viz_env = create_maze_env(maze_size=8, difficulty='medium')
    
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
        
        # Execute policy action (policy is a dict for MC)
        action = policy.get(state, 0)  # Default to action 0 if state not in policy
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
        viz_env = create_maze_env(maze_size=8, difficulty='medium')
    
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


def experiment_epsilon_exploration():
    """
    Experiment 1: Analyze impact of epsilon (exploration rate) on learning.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Impact of Exploration Rate (Epsilon) on Maze")
    print("="*70)
    
    global viz_env
    
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.3]
    results = {}
    
    # Use persistent visualization environment
    if viz_env is None:
        viz_env = create_maze_env(maze_size=8, difficulty='medium')
    
    for epsilon in epsilons:
        print(f"\n--- Testing epsilon = {epsilon} ---")
        
        # Use the shared environment for all epsilon experiments
        agent = MonteCarloAgent(viz_env, gamma=0.99, epsilon=epsilon)
        
        # Run policy evaluation
        stats = agent.policy_evaluation(num_episodes=1000)
        
        # Visualize the learned policy using the persistent environment
        print(f"  Visualizing learned policy for epsilon={epsilon}...")
        reward, steps = visualize_policy_execution(agent.get_policy(), 
                                                    title=f"MC Policy (ε={epsilon})",
                                                    render_delay=0.05)
        
        # Evaluate learned policy
        eval_reward = agent.evaluate_policy(num_eval_episodes=50)
        
        results[epsilon] = {
            'stats': stats,
            'eval_reward': eval_reward,
            'episode_rewards': agent.episode_rewards
        }
        
        print(f"[OK] Policy executed: Reward={reward:.0f}, Steps={steps}")
        print(f"     Average evaluation reward: {eval_reward:.3f}")
        time.sleep(0.5)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Learning curves for different epsilons
    for epsilon in epsilons:
        rewards = results[epsilon]['episode_rewards']
        # Smooth with moving average
        smoothed = np.convolve(rewards, np.ones(100)/100, mode='valid')
        axes[0, 0].plot(smoothed, label=f'ε={epsilon}', alpha=0.8)
    
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Reward (Moving Avg)', fontsize=12)
    axes[0, 0].set_title('Learning Curves for Different Exploration Rates', fontsize=13)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Final evaluation rewards
    epsilon_list = list(results.keys())
    eval_rewards = [results[e]['eval_reward'] for e in epsilon_list]
    
    axes[0, 1].plot(epsilon_list, eval_rewards, 'bo-', linewidth=2, markersize=10)
    axes[0, 1].set_xlabel('Exploration Rate (ε)', fontsize=12)
    axes[0, 1].set_ylabel('Evaluation Reward', fontsize=12)
    axes[0, 1].set_title('Impact of Exploration on Final Performance', fontsize=13)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Number of unique states visited
    num_states = [results[e]['stats']['num_states_visited'] for e in epsilon_list]
    
    axes[1, 0].bar(range(len(epsilon_list)), num_states, color='green', alpha=0.7)
    axes[1, 0].set_xticks(range(len(epsilon_list)))
    axes[1, 0].set_xticklabels([f'ε={e}' for e in epsilon_list])
    axes[1, 0].set_ylabel('Unique States Visited', fontsize=12)
    axes[1, 0].set_title('State Space Exploration Coverage', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Convergence speed (episodes to reach 50% of final performance)
    for epsilon in epsilons:
        rewards = results[epsilon]['episode_rewards']
        smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
        target = np.max(smoothed) * 0.5
        
        # Find episode where we reach target
        convergence_ep = np.where(smoothed >= target)[0]
        if len(convergence_ep) > 0:
            conv_episode = convergence_ep[0]
        else:
            conv_episode = len(smoothed)
        
        axes[1, 1].scatter(epsilon, conv_episode, s=100, alpha=0.7)
    
    axes[1, 1].set_xlabel('Exploration Rate (ε)', fontsize=12)
    axes[1, 1].set_ylabel('Episodes to 50% Convergence', fontsize=12)
    axes[1, 1].set_title('Convergence Speed vs Exploration', fontsize=13)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task2_epsilon_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task2_epsilon_analysis.png'")
    plt.close()
    
    return results


def experiment_convergence():
    """
    Experiment 2: Analyze Monte Carlo convergence properties.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Convergence Analysis (Maze)")
    print("="*70)
    
    global viz_env
    
    # Use persistent visualization environment
    env = viz_env if viz_env is not None else create_maze_env(maze_size=8, difficulty='medium')
    agent = MonteCarloAgent(env, gamma=0.99, epsilon=0.1)
    
    # Track convergence over episodes
    value_means = []
    value_stds = []
    num_states_visited = []
    
    stats = agent.policy_evaluation(num_episodes=1000)
    
    # Visualize the learned policy
    print("  Visualizing learned policy...")
    reward, steps = visualize_policy_execution(agent.get_policy(),
                                                title="MC Convergence Analysis",
                                                render_delay=0.05)
    print(f"  [OK] Policy executed: Reward={reward:.0f}, Steps={steps}")
    time.sleep(0.5)
    
    # Manually track convergence by re-running with tracking
    env.reset()
    for episode_num in range(1000):
        episode, _ = agent.generate_episode()
        visited_states = set()
        G = 0
        
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + agent.gamma * G
            
            if state not in visited_states:
                visited_states.add(state)
                if state not in agent.returns:
                    agent.returns[state] = []
                    agent.V[state] = 0.0
                
                agent.returns[state].append(G)
                agent.V[state] = np.mean(agent.returns[state])
        
        # Track statistics every 100 episodes
        if (episode_num + 1) % 100 == 0:
            values = list(agent.V.values())
            value_means.append(np.mean(values))
            value_stds.append(np.std(values))
            num_states_visited.append(len(agent.V))
    
    # Plot convergence
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    episodes = np.arange(1, len(value_means) + 1) * 100
    
    # Plot 1: Mean value function
    axes[0, 0].plot(episodes, value_means, 'b-', linewidth=2)
    axes[0, 0].fill_between(episodes, 
                            np.array(value_means) - np.array(value_stds),
                            np.array(value_means) + np.array(value_stds),
                            alpha=0.3)
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Mean Value', fontsize=12)
    axes[0, 0].set_title('Mean Value Function Over Time (±1 STD)', fontsize=13)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Value STD
    axes[0, 1].plot(episodes, value_stds, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Value Std Dev', fontsize=12)
    axes[0, 1].set_title('Value Function Variance Over Time', fontsize=13)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Number of states visited
    axes[1, 0].plot(episodes, num_states_visited, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Episode', fontsize=12)
    axes[1, 0].set_ylabel('Unique States', fontsize=12)
    axes[1, 0].set_title('State Space Coverage Over Time', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Episode rewards
    rewards = agent.episode_rewards
    smoothed = np.convolve(rewards, np.ones(100)/100, mode='valid')
    axes[1, 1].plot(smoothed, 'purple', linewidth=2, label='Moving Average (100 episodes)')
    axes[1, 1].scatter(range(len(rewards)), rewards, alpha=0.1, s=5, color='gray', label='Raw rewards')
    axes[1, 1].set_xlabel('Episode', fontsize=12)
    axes[1, 1].set_ylabel('Reward', fontsize=12)
    axes[1, 1].set_title('Episode Rewards During Learning', fontsize=13)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task2_convergence_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task2_convergence_analysis.png'")
    plt.close()


def experiment_value_distribution():
    """
    Experiment 3: Analyze learned value function distribution.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Value Function Distribution Analysis (Maze)")
    print("="*70)
    
    global viz_env
    
    # Train with different epsilon values
    epsilons = [0.0, 0.1, 0.3, 0.5]
    value_distributions = {}
    
    # Use persistent visualization environment
    if viz_env is None:
        viz_env = create_maze_env(maze_size=8, difficulty='medium')
    
    for epsilon in epsilons:
        print(f"\nTraining with epsilon={epsilon}...")
        
        agent = MonteCarloAgent(viz_env, gamma=0.99, epsilon=epsilon)
        agent.policy_evaluation(num_episodes=1000)
        
        # Visualize the learned policy
        print(f"  Visualizing learned policy for epsilon={epsilon}...")
        reward, steps = visualize_policy_execution(agent.get_policy(),
                                                    title=f"MC Value Dist (ε={epsilon})",
                                                    render_delay=0.05)
        print(f"  [OK] Policy executed: Reward={reward:.0f}, Steps={steps}")
        
        values = list(agent.V.values())
        value_distributions[epsilon] = values
        
        print(f"  Mean value: {np.mean(values):.3f}")
        print(f"  Std value: {np.std(values):.3f}")
        print(f"  Unique states visited: {len(agent.V)}")
        
        time.sleep(0.5)
    
    # Plot distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (epsilon, values) in enumerate(value_distributions.items()):
        ax = axes[idx // 2, idx % 2]
        
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.3f}')
        ax.set_xlabel('Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Value Distribution (ε={epsilon})', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('task2_value_distribution.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task2_value_distribution.png'")
    plt.close()


def main():
    """Main execution function."""
    global viz_env
    
    print("\n" + "="*70)
    print("TASK 2: MONTE CARLO METHODS")
    print("="*70)
    print("This task implements Monte Carlo Policy Evaluation on the Maze environment")
    print("and analyzes the impact of exploration-exploitation strategies.\n")
    print("Note: A persistent visualization window will show the agent's behavior")
    print("as the algorithms improve throughout the experiments.\n")
    
    # First: Visualize the maze environment
    visualize_maze_exploration()
    
    # Run experiments
    print("\nStarting experiments...\n")
    experiment_epsilon_exploration()
    experiment_convergence()
    experiment_value_distribution()
    
    print("\n" + "="*70)
    print("TASK 2 COMPLETE")
    print("="*70)
    print("Generated plots:")
    print("  - task2_epsilon_analysis.png")
    print("  - task2_convergence_analysis.png")
    print("  - task2_value_distribution.png")
    
    # Clean up visualization
    if viz_env is not None:
        viz_env.close()
    plt.close('all')
    
    print("\nVisualization window closed.")


if __name__ == "__main__":
    main()
