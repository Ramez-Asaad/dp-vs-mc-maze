# Dynamic Programming vs Monte Carlo: Reinforcement Learning on Maze Navigation

A comprehensive implementation and comparative analysis of Dynamic Programming and Monte Carlo reinforcement learning algorithms applied to maze navigation.

## 📋 Overview

This project implements two foundational RL algorithms from scratch and compares their performance:

- **Dynamic Programming (Value Iteration)**: Model-based planning requiring full environment knowledge
- **Monte Carlo (First-Visit)**: Model-free learning from experience

Both algorithms are tested on a custom 8×8 grid-world maze environment with obstacles and rewards.

**Key Finding**: DP converges **72× faster** (0.08s) but requires a complete model, while MC learns without a model at the cost of slower convergence (5.78s). Both achieve equivalent final performance.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Ramez-Asaad/dp-vs-mc-maze
cd dp-vs-mc-maze

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Examples

```bash
# Task 1: Dynamic Programming
python task1_dynamic_programming/task1_main.py

# Task 2: Monte Carlo
python task2_monte_carlo/task2_main.py

# Task 3: Comparative Analysis
python task3_analysis/task3_main.py
```

## 📁 Project Structure

```
.
├── README.md                    # This file
├── REPORT.pdf                   # Full academic report
├── requirements.txt             # Python dependencies
│
├── task1_dynamic_programming/   # Value Iteration & Policy Iteration
│   ├── task1_main.py           # 3 experiments with visualizations
│   ├── dp_algorithms.py        # Core DP implementation
│   └── environment_setup.py    # Maze environment
│
├── task2_monte_carlo/          # First-Visit Monte Carlo
│   ├── task2_main.py           # Epsilon sensitivity analysis
│   ├── mc_algorithms.py        # MC algorithm with Q-learning
│   └── environment_setup.py    # Maze environment
│
├── task3_analysis/             # Comparative framework
│   └── task3_main.py           # Decision guide & metrics
│
└── results/                    # Generated visualizations
    ├── task1_dp/              # 10 DP analysis plots
    ├── task2_mc/              # 3 MC convergence plots
    └── task3_comparison/      # 2 comparative plots
```

## 🔬 What's Implemented

### Task 1: Dynamic Programming

Implements Value Iteration and Policy Iteration with three experiments:

1. **Discount Factor Sensitivity** ($\gamma \in \{0.50, 0.70, 0.90, 0.99\}$)
   - Tests how long-term planning affects convergence
   - Result: $\gamma = 0.99$ optimal

2. **Algorithm Comparison** (PI vs VI)
   - Both converge to near-identical policies
   - VI: 16 iterations, PI: 12 iterations

3. **Convergence Analysis**
   - Plots value function convergence
   - Geometric convergence rate $O(\gamma^k)$

### Task 2: Monte Carlo

Implements First-Visit MC with ε-greedy exploration, three experiments:

1. **Exploration Rate Sensitivity** ($\epsilon \in \{0.01, 0.05, 0.10, 0.30\}$)
   - Tests exploration vs exploitation trade-off
   - Result: $\epsilon = 0.05$ optimal

2. **Learning Curves**
   - Episode-by-episode reward tracking
   - Convergence at ~500 episodes

3. **Value Distribution Analysis**
   - Heatmaps of Q-value distributions
   - Policy quality assessment

### Task 3: Comparative Analysis

Develops practical decision framework:

- **Performance Metrics**: Iterations, time, final reward, state coverage
- **Algorithm Selection Guide**: 8 scenarios with recommendations
- **Decision Tree**: When to use DP vs MC

## 📊 Key Results

### Performance Comparison

| Metric | DP (VI) | MC |
|--------|---------|-----|
| **Convergence** | 16 iterations | 5000 episodes |
| **Execution Time** | 0.08 seconds | 5.78 seconds |
| **Speedup** | **72× faster** | — |
| **Final Reward** | 3.50 | 3.50 |
| **Model Required** | Yes | No |

### Algorithm Selection

| Scenario | Recommendation | Reason |
|----------|---|---------|
| Model available | **DP** | Can solve exactly, fast |
| Unknown environment | **MC** | Learns from experience |
| Small state space | **DP** | Efficient |
| Large state space | **MC** | Scales better |
| Limited samples | **DP** | Uses model efficiently |
| Abundant experience | **MC** | Takes advantage of data |

## 🎯 Core Algorithms

### Value Iteration
```python
V(s) ← max_a [ R(s,a) + γ ∑_s' P(s'|s,a) V(s') ]
```

### First-Visit Monte Carlo
```python
Q(s,a) = (1/N(s,a)) ∑ G_t   # Average returns for (state,action)
π(a|s) = ε-greedy policy     # Exploration with ε-greedy
```

## 🔧 Technologies

- **Python 3.11+**
- **NumPy**: Numerical computations
- **Matplotlib**: Visualizations
- **SciPy**: Scientific computing

## 📈 Visualizations

15 high-resolution PNG visualizations generated:

- **Task 1** (10): Discount factor analysis, algorithm comparison, convergence curves, value heatmaps
- **Task 2** (3): Epsilon sensitivity, learning convergence, value distributions
- **Task 3** (2): Method comparison, scenario analysis

All stored in `results/` organized by task.

## 💡 Key Insights

1. **Speed vs Knowledge Trade-off**: DP is 72× faster but needs the model. MC is slower but model-free.

2. **Equivalent Quality**: Both achieve identical final performance on this maze (reward 3.50).

3. **Hyperparameter Sensitivity**: Performance is very sensitive to $\gamma$ (DP) and $\epsilon$ (MC).

4. **Practical Implications**:
   - Use DP for simulated environments (games, physics simulators)
   - Use MC for real-world learning (robotics, autonomous systems)
   - Modern approach: Temporal Difference (Q-learning) combines benefits of both

## 📝 Algorithm Details

### Dynamic Programming

- **Type**: Model-based, offline planning
- **Convergence**: Guaranteed, geometric rate $O(\gamma^k)$
- **Optimality**: Guaranteed
- **Model**: Requires complete transition model
- **Scalability**: Limited to small-medium state spaces

### Monte Carlo

- **Type**: Model-free, online learning
- **Convergence**: Probabilistic, rate $O(1/N)$
- **Optimality**: Greedy approximation
- **Model**: No model required
- **Scalability**: Better for large/continuous spaces

## 🎓 Academic Context

- **Course**: AIE322 - Advanced Machine Learning
- **Semester**: Fall 2025-26
- **Institution**: Alamein International University

## 📖 References

1. Bellman, R. (1957). Dynamic Programming. Princeton University Press.
2. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
3. Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-Critic Algorithms. SIAM Journal on Control and Optimization.

## 📄 Full Report

For comprehensive methodology, detailed results, and analysis, see `REPORT.pdf`.

## 🤝 Contributing

This is an academic assignment project. However, feel free to:
- Report issues
- Suggest improvements
- Extend the implementation (TD methods, continuous control, etc.)

## 📜 License

Academic use only. See LICENSE file for details.

## 👤 Author

Student Report - Advanced Machine Learning Assignment 1

---

**Questions?** Check the full `REPORT.pdf` or examine the code in each task folder.
