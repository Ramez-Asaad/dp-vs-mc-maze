# Contributing Guide

Thank you for your interest in this project! Here's how to contribute.

## Code Structure

The project is organized into three main task folders:

- `task1_dynamic_programming/`: Value Iteration and Policy Iteration implementations
- `task2_monte_carlo/`: First-Visit Monte Carlo with ε-greedy exploration
- `task3_analysis/`: Comparative analysis and decision framework

## Running Tests

Each task has a main script:

```bash
python task1_dynamic_programming/task1_main.py
python task2_monte_carlo/task2_main.py
python task3_analysis/task3_main.py
```

Verify that:
1. No errors occur during execution
2. Visualizations are generated in `results/` folder
3. Console output shows convergence metrics

## Making Changes

### To modify an algorithm:

1. Edit the relevant `*_algorithms.py` file
2. Update the corresponding `task*_main.py` to test changes
3. Regenerate visualizations
4. Update results in documentation if metrics change

### To extend the project:

Consider these enhancements:
- Implement Temporal Difference (TD) methods (Q-learning, SARSA)
- Add support for continuous state spaces
- Implement function approximation (neural networks)
- Add more complex environments
- Optimize for larger state spaces

### Code Style

Follow PEP 8:
- 4-space indentation
- Meaningful variable names
- Comments for complex logic
- Docstrings for functions

Example:
```python
def value_iteration(env, gamma=0.99, theta=1e-6, max_iterations=1000):
    """
    Compute optimal value function using Value Iteration.
    
    Args:
        env: Environment with P (transition probs) and R (rewards)
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum iterations
        
    Returns:
        V: Optimal value function
    """
    ...
```

## Adding Features

### New Environment

Create `task*/new_environment.py`:
```python
class CustomEnvironment:
    def __init__(self):
        self.num_states = ...
        self.num_actions = ...
        self.P = {}  # transition probabilities
        self.R = {}  # rewards
        
    def reset(self):
        return initial_state
        
    def step(self, action):
        return next_state, reward
```

### New Algorithm

Create appropriate file and follow module structure:
```python
class NewAlgorithm:
    def __init__(self, env, **params):
        ...
    
    def train(self):
        ...
    
    def get_policy(self):
        ...
```

## Testing

Before submitting changes:

```bash
# Run all tasks
python task1_dynamic_programming/task1_main.py
python task2_monte_carlo/task2_main.py
python task3_analysis/task3_main.py

# Verify visualizations
ls results/task*/*.png
```

## Documentation

Update documentation when:
- Adding new features
- Changing algorithm parameters
- Modifying environment structure
- Improving methodology

Keep these files in sync:
- `README.md` - Quick start and overview
- `REPORT.pdf` - Detailed academic report
- Code comments - Implementation details

## Reporting Issues

Provide:
1. Steps to reproduce
2. Expected behavior
3. Actual behavior
4. Python version and dependency versions
5. Error message/traceback

## Commit Messages

Use clear, descriptive messages:
```
✓ Good:
  "Implement Q-learning algorithm with epsilon-greedy exploration"
  "Fix convergence threshold check in Value Iteration"
  "Add continuous state space support"

✗ Avoid:
  "fix bug"
  "update"
  "changes"
```

## Questions?

See the full `REPORT.pdf` for methodology and theory, or examine the code comments in each task folder.

Happy contributing!
