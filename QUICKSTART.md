# Quick Start Guide

Get this project running in 2 minutes.

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/reinforcement-learning-maze.git
cd reinforcement-learning-maze
```

### 2. Create Virtual Environment

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Examples

### Task 1: Dynamic Programming

```bash
python task1_dynamic_programming/task1_main.py
```

**Output:**
- Console: Convergence metrics for different discount factors
- Plots: `results/task1_dp/` (10 visualizations)

**What it does:**
1. Tests Value Iteration with different $\gamma$ values
2. Compares Policy Iteration vs Value Iteration
3. Analyzes convergence behavior

### Task 2: Monte Carlo Learning

```bash
python task2_monte_carlo/task2_main.py
```

**Output:**
- Console: Epsilon sensitivity analysis
- Plots: `results/task2_mc/` (3 visualizations)

**What it does:**
1. Tests First-Visit MC with different exploration rates
2. Plots learning curves
3. Shows value distribution heatmaps

### Task 3: Comparative Analysis

```bash
python task3_analysis/task3_main.py
```

**Output:**
- Console: Decision framework and recommendations
- Plots: `results/task3_comparison/` (2 visualizations)

**What it does:**
1. Compares all metrics (speed, quality, requirements)
2. Recommends algorithms for different scenarios
3. Generates decision guide

## Check Results

```bash
# View all generated plots
ls results/
ls results/task1_dp/
ls results/task2_mc/
ls results/task3_comparison/
```

## Read the Report

For detailed methodology, theory, and analysis:

```bash
# View the PDF report
open REPORT.pdf          # Mac
xdg-open REPORT.pdf      # Linux
start REPORT.pdf         # Windows
```

## Project Structure

```
.
├── README_GITHUB.md              # Full documentation
├── REPORT.pdf                    # Academic report
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
├── .gitignore                    # Git ignore rules
│
├── task1_dynamic_programming/    # Value Iteration & Policy Iteration
│   ├── task1_main.py
│   ├── dp_algorithms.py
│   └── environment_setup.py
│
├── task2_monte_carlo/            # First-Visit Monte Carlo
│   ├── task2_main.py
│   ├── mc_algorithms.py
│   └── environment_setup.py
│
├── task3_analysis/               # Comparative analysis
│   └── task3_main.py
│
└── results/                      # Generated visualizations
    ├── task1_dp/                 # 10 plots
    ├── task2_mc/                 # 3 plots
    └── task3_comparison/         # 2 plots
```

## Key Results

| Metric | DP | MC |
|--------|----|----|
| Convergence | 16 iterations | 5000 episodes |
| Time | 0.08s | 5.78s |
| Speedup | **72× faster** | — |
| Final Reward | 3.50 | 3.50 |
| Model Required | Yes | No |

**Recommendation:**
- Use **DP** for simulated environments with known models
- Use **MC** for real-world scenarios with unknown dynamics

## Troubleshooting

### Python version error
```bash
python --version  # Check version (needs 3.8+)
```

### Missing dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### ImportError in tasks
```bash
# Ensure you're in the project root directory
pwd  # or cd on Windows

# Run from correct location
python task1_dynamic_programming/task1_main.py
```

### Plots not showing
Matplotlib displays are generated as PNG files in `results/`. Check there if plots don't appear in a window.

## Next Steps

1. **Understand the theory**: Read REPORT.pdf Introduction and Background sections
2. **Explore the code**: Look at `dp_algorithms.py` and `mc_algorithms.py`
3. **Modify hyperparameters**: Try different $\gamma$ and $\epsilon$ values
4. **Extend functionality**: Implement new environments or algorithms

## Questions?

- See `CONTRIBUTING.md` for contribution guidelines
- Check `REPORT.pdf` for detailed methodology
- Examine code comments in task folders

## License

MIT License - see LICENSE file for details

---

**Enjoy learning RL algorithms!** 🚀
