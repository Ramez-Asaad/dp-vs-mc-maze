"""
Task 3: Comparative Analysis
Comparing Dynamic Programming and Monte Carlo methods across different scenarios.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import from task1 and task2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from task1_dynamic_programming.environment_setup import create_maze_env
from task1_dynamic_programming.dp_algorithms import DynamicProgramming
from task2_monte_carlo.mc_algorithms import MonteCarloAgent


def comparison_performance_efficiency():
    """
    Compare DP and MC on computational efficiency and convergence.
    """
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS 1: Performance vs Efficiency")
    print("="*70)
    
    # DP on Maze
    print("\n--- Dynamic Programming on Maze ---")
    env_dp = create_maze_env(maze_size=8, difficulty='medium')
    dp = DynamicProgramming(env_dp, gamma=0.99)
    
    # Build model
    model = dp.build_transition_model(num_episodes=200)
    
    # Value iteration with timing
    import time
    start_time = time.time()
    policy_dp, V_dp = dp.value_iteration(model, theta=1e-6, max_iterations=200)
    dp_time = time.time() - start_time
    
    # MC on Maze
    print("\n--- Monte Carlo on Maze ---")
    env_mc = create_maze_env(maze_size=8, difficulty='medium')
    mc = MonteCarloAgent(env_mc, gamma=0.99, epsilon=0.1)
    
    start_time = time.time()
    mc_stats = mc.policy_evaluation(num_episodes=5000)
    mc_time = time.time() - start_time
    
    # Evaluate both
    env_eval_dp = create_maze_env(maze_size=8, difficulty='medium')
    def evaluate_dp_policy(env, policy, n_episodes=50):
        total_reward = 0
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = policy[state]
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                state = next_state
            total_reward += episode_reward
        return total_reward / n_episodes
    
    dp_eval_reward = evaluate_dp_policy(env_eval_dp, policy_dp)
    mc_eval_reward = mc.evaluate_policy(num_eval_episodes=50)
    
    print(f"\nDP - Time: {dp_time:.2f}s, Reward: {dp_eval_reward:.2f}")
    print(f"MC - Time: {mc_time:.2f}s, Reward: {mc_eval_reward:.2f}")
    
    env_dp.close()
    env_eval_dp.close()
    env_mc.close()
    
    return {
        'dp': {'time': dp_time, 'reward': dp_eval_reward, 'method': 'DP (Maze)'},
        'mc': {'time': mc_time, 'reward': mc_eval_reward, 'method': 'MC (Maze)'}
    }


def comparison_characteristics_analysis():
    """
    Detailed analysis of DP vs MC characteristics.
    """
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS 2: Method Characteristics")
    print("="*70)
    
    # Create comparison table
    comparison_data = {
        'Characteristic': [
            'Model Requirement',
            'State Space',
            'Convergence Type',
            'Computational Cost',
            'Memory Usage',
            'Exploration',
            'Application Type',
            'Implementation',
            'Convergence Rate',
            'Scalability'
        ],
        'Dynamic Programming': [
            'Requires full model',
            'Discrete/Small',
            'Guaranteed (finite states)',
            'High (planning phase)',
            'High (full transition model)',
            'Not needed (planning)',
            'Episodic & Continuing',
            'Complex (matrix operations)',
            'Fast (geometric convergence)',
            'Poor (exponential in state size)'
        ],
        'Monte Carlo': [
            'Model-free',
            'Discrete/Continuous',
            'Probabilistic (with exploration)',
            'Low (only simulation)',
            'Low (stores episodes)',
            'Required (exploration)',
            'Primarily Episodic',
            'Simple (averaging returns)',
            'Slow (O(1/sqrt(n)))',
            'Better (sample complexity)'
        ]
    }
    
    print("\n" + "-"*80)
    print(f"{'Characteristic':<25} {'Dynamic Programming':<25} {'Monte Carlo':<25}")
    print("-"*80)
    
    for i in range(len(comparison_data['Characteristic'])):
        char = comparison_data['Characteristic'][i]
        dp_char = comparison_data['Dynamic Programming'][i]
        mc_char = comparison_data['Monte Carlo'][i]
        print(f"{char:<25} {dp_char:<25} {mc_char:<25}")
    
    print("-"*80)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Computational Cost vs Problem Size
    problem_sizes = np.array([10, 50, 100, 500, 1000])
    dp_cost = problem_sizes ** 2  # Quadratic for DP
    mc_cost = problem_sizes * 10  # Linear for MC
    
    axes[0, 0].loglog(problem_sizes, dp_cost, 'b-o', label='DP', linewidth=2, markersize=8)
    axes[0, 0].loglog(problem_sizes, mc_cost, 'r-s', label='MC', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Problem Size (State Space)', fontsize=12)
    axes[0, 0].set_ylabel('Computational Cost', fontsize=12)
    axes[0, 0].set_title('Scalability: Computational Cost', fontsize=13)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, which='both')
    
    # Plot 2: Convergence Rate Comparison
    iterations = np.arange(0, 100)
    dp_convergence = 0.95 ** iterations  # Geometric convergence
    mc_convergence = 1 / (np.sqrt(iterations + 1))  # Polynomial convergence
    
    axes[0, 1].semilogy(iterations, dp_convergence, 'b-', label='DP (Geometric)', linewidth=2)
    axes[0, 1].semilogy(iterations, mc_convergence, 'r-', label='MC (Polynomial)', linewidth=2)
    axes[0, 1].set_xlabel('Iterations/Episodes', fontsize=12)
    axes[0, 1].set_ylabel('Error (log scale)', fontsize=12)
    axes[0, 1].set_title('Convergence Rate Comparison', fontsize=13)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, which='both')
    
    # Plot 3: Suitability for Problem Types
    problem_types = ['Discrete\nSmall', 'Discrete\nLarge', 'Continuous', 'Model\nUnavailable', 'Episodic\nTasks']
    dp_scores = [10, 3, 2, 1, 8]
    mc_scores = [8, 8, 9, 9, 9]
    
    x = np.arange(len(problem_types))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, dp_scores, width, label='DP', alpha=0.8, color='blue')
    axes[1, 0].bar(x + width/2, mc_scores, width, label='MC', alpha=0.8, color='red')
    axes[1, 0].set_ylabel('Suitability Score', fontsize=12)
    axes[1, 0].set_title('Problem Type Suitability', fontsize=13)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(problem_types, fontsize=10)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].set_ylim([0, 11])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Implementation Complexity
    categories = ['Learning\nCurve', 'Algorithm\nComplexity', 'Debugging\nDifficulty', 'Efficiency\nTuning']
    dp_complexity = [7, 8, 6, 7]
    mc_complexity = [4, 3, 4, 5]
    
    x2 = np.arange(len(categories))
    axes[1, 1].bar(x2 - width/2, dp_complexity, width, label='DP', alpha=0.8, color='blue')
    axes[1, 1].bar(x2 + width/2, mc_complexity, width, label='MC', alpha=0.8, color='red')
    axes[1, 1].set_ylabel('Complexity (1-10)', fontsize=12)
    axes[1, 1].set_title('Implementation Complexity', fontsize=13)
    axes[1, 1].set_xticks(x2)
    axes[1, 1].set_xticklabels(categories, fontsize=10)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].set_ylim([0, 10])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('task3_method_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task3_method_comparison.png'")
    plt.close()


def decision_tree_analysis():
    """
    Create decision framework for choosing between DP and MC.
    """
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS 3: Method Selection Framework")
    print("="*70)
    
    decision_framework = """
    
    DECISION FRAMEWORK: Choosing Between DP and MC
    ====================================================================
    
    1. PROBLEM MODEL AVAILABILITY
       * Model is KNOWN and SMALL (< 1000 states)
         --> USE DYNAMIC PROGRAMMING
             - Can leverage full environment knowledge
             - Faster convergence (geometric)
             - Can use policy/value iteration
       
       * Model is UNKNOWN or LARGE (> 10000 states)
         --> USE MONTE CARLO
             - No model needed
             - Handles continuous spaces (with discretization)
             - Better for high-dimensional problems
    
    2. TASK TYPE
       * EPISODIC (tasks have clear ending)
         --> MONTE CARLO is EXCELLENT
             - Natural for one-shot tasks (blackjack, games)
             - Easy to compute returns
             - Handles variable episode lengths
       
       * CONTINUING (infinite horizons)
         --> DYNAMIC PROGRAMMING is BETTER
             - Uses discounted rewards efficiently
             - Steady-state convergence
             - Better for infinite-horizon problems
    
    3. COMPUTATIONAL CONSTRAINTS
       * LOW latency required
         --> USE MONTE CARLO
             - No planning phase needed
             - Anytime algorithm (usable at any time)
             - Incrementally improves with more episodes
       
       * Can afford planning phase
         --> USE DYNAMIC PROGRAMMING
             - Plan once, act many times
             - Optimal after convergence
             - Matrix operations (parallelizable)
    
    SUMMARY TABLE:
    ====================================================================
    Factor              | Dynamic Programming | Monte Carlo
    ====================================================================
    Model Required      | YES                | NO
    Max States          | <10K               | >10K
    Convergence Speed   | Fast (geometric)   | Slow (O(1/sqrt n))
    Memory              | High (full model)  | Low (episodes)
    Exploration         | Not needed         | Essential
    Best For            | Episodic + known   | Model-free + large
    Implementation      | Complex            | Simple
    ====================================================================
    
    PRACTICAL EXAMPLES:
    
    Use DP: Robot navigation in known map, Game playing with rules
    Use MC: Blackjack (model unknown), Poker (high variance),
            Stock trading (no model), Robotics (real world)
    
    """
    
    print(decision_framework)
    
    # Save to file
    with open('task3_decision_framework.txt', 'w') as f:
        f.write(decision_framework)
    
    print("Decision framework saved to 'task3_decision_framework.txt'")


def scenario_analysis():
    """
    Analyze specific scenarios where one method outperforms the other.
    """
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS 4: Scenario-Based Analysis")
    print("="*70)
    
    scenarios = {
        'Small Discrete World': {
            'DP': 'Excellent - Fast, accurate, can solve exactly',
            'MC': 'Good - Works but slower',
            'Winner': 'DP'
        },
        'Large Discrete World': {
            'DP': 'Poor - Computation/memory explosion',
            'MC': 'Good - Scales reasonably',
            'Winner': 'MC'
        },
        'Unknown Environment': {
            'DP': 'Impossible - Requires model',
            'MC': 'Excellent - Model-free learning',
            'Winner': 'MC'
        },
        'Stochastic Environment': {
            'DP': 'Good - Can handle with full model',
            'MC': 'Excellent - Naturally handles variance',
            'Winner': 'MC'
        },
        'Limited Samples': {
            'DP': 'Good - Uses model efficiently',
            'MC': 'Poor - Needs many episodes',
            'Winner': 'DP'
        },
        'Real-time Decision': {
            'DP': 'Poor - Need planning phase first',
            'MC': 'Good - Can act anytime',
            'Winner': 'MC'
        },
        'Episodic Task': {
            'DP': 'Good - Works with discount factor',
            'MC': 'Excellent - Natural fit',
            'Winner': 'MC'
        },
        'Financial Domain': {
            'DP': 'Poor - Exponential state space',
            'MC': 'Good - Can handle complexity',
            'Winner': 'MC'
        }
    }
    
    print("\n" + "-"*80)
    print(f"{'Scenario':<25} {'Dynamic Programming':<25} {'Monte Carlo':<25}")
    print("-"*80)
    
    for scenario, analysis in scenarios.items():
        print(f"\n{scenario}:")
        print(f"  DP:  {analysis['DP']}")
        print(f"  MC:  {analysis['MC']}")
        print(f"  Winner: {analysis['Winner']}")
    
    print("\n" + "-"*80)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scenario_names = list(scenarios.keys())
    dp_suitability = []
    mc_suitability = []
    
    suitability_map = {
        'Excellent': 10,
        'Good': 7,
        'Poor': 2,
        'Impossible': 0
    }
    
    for scenario in scenario_names:
        dp_text = scenarios[scenario]['DP'].split(' - ')[0]
        mc_text = scenarios[scenario]['MC'].split(' - ')[0]
        
        dp_suitability.append(suitability_map.get(dp_text, 5))
        mc_suitability.append(suitability_map.get(mc_text, 5))
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, dp_suitability, width, label='DP', alpha=0.8, color='blue')
    bars2 = ax.barh(x + width/2, mc_suitability, width, label='MC', alpha=0.8, color='red')
    
    ax.set_xlabel('Suitability Score (0-10)', fontsize=12)
    ax.set_title('Method Suitability Across Different Scenarios', fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(scenario_names, fontsize=11)
    ax.legend(fontsize=12)
    ax.set_xlim([0, 11])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('task3_scenario_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'task3_scenario_analysis.png'")
    plt.close()


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("TASK 3: COMPARATIVE ANALYSIS")
    print("="*70)
    print("Comparing Dynamic Programming and Monte Carlo methods")
    print("across different dimensions and scenarios")
    print("Using Custom Maze environment (8x8 grid with configurable difficulty)")
    
    # Run analyses
    comparison_performance_efficiency()
    comparison_characteristics_analysis()
    decision_tree_analysis()
    scenario_analysis()
    
    print("\n" + "="*70)
    print("TASK 3 COMPLETE")
    print("="*70)
    print("Generated outputs:")
    print("  - task3_method_comparison.png")
    print("  - task3_scenario_analysis.png")
    print("  - task3_decision_framework.txt")


if __name__ == "__main__":
    main()
