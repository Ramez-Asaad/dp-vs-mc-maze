# RL Maze Navigation: DP vs Monte Carlo

A clean, well-documented implementation comparing **Dynamic Programming** and **Monte Carlo** reinforcement learning algorithms on maze navigation.

**Status**: Complete ✅ | **Python**: 3.8+ | **License**: MIT

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone <repo-url>
cd reinforcement-learning-maze
python -m venv venv && source venv/bin/activate

# 2. Install & run
pip install -r requirements.txt
python task1_dynamic_programming/task1_main.py
python task2_monte_carlo/task2_main.py
python task3_analysis/task3_main.py
```

See **[QUICKSTART.md](QUICKSTART.md)** for detailed setup guide.

## 📊 Key Results

| Metric | DP | MC | Winner |
|--------|----|----|--------|
| **Speed** | 0.08s | 5.78s | DP (72× faster) |
| **Final Reward** | 3.50 | 3.50 | Tie |
| **Model Required** | ✓ Yes | ✗ No | MC (model-free) |
| **Best For** | Known environments | Unknown environments | Context-dependent |

## 📁 What's Here

```
.
├── QUICKSTART.md                     # 👈 Start here (2 min setup)
├── README_GITHUB.md                  # Full documentation
├── REPORT.pdf                        # Academic report (detailed)
├── requirements.txt                  # Dependencies
├── setup.py                          # Package installer
├── .gitignore                        # Git configuration
├── LICENSE                           # MIT License
├── CONTRIBUTING.md                   # Contribution guide
│
├── task1_dynamic_programming/        # ✓ Value Iteration
│   ├── task1_main.py                # 3 experiments
│   ├── dp_algorithms.py             # Core algorithm
│   └── environment_setup.py         # Maze environment
│
├── task2_monte_carlo/               # ✓ First-Visit MC
│   ├── task2_main.py                # 3 experiments
│   ├── mc_algorithms.py             # Core algorithm
│   └── environment_setup.py         # Maze with viz
│
├── task3_analysis/                  # ✓ Comparison
│   └── task3_main.py                # Decision framework
│
└── results/                         # Generated visualizations
    ├── task1_dp/      (10 plots)
    ├── task2_mc/      (3 plots)
    └── task3_comparison/ (2 plots)
```

## 🎯 What Each Task Does

### Task 1: Dynamic Programming
Tests **Value Iteration** with different hyperparameters:
- Tests γ ∈ {0.50, 0.70, 0.90, 0.99}
- Compares Policy Iteration vs Value Iteration
- Analyzes convergence curves

**Results**: ✅ 16 iterations, 0.08s, optimal γ = 0.99

### Task 2: Monte Carlo
Tests **First-Visit MC** with ε-greedy exploration:
- Tests ε ∈ {0.01, 0.05, 0.10, 0.30}
- Plots learning convergence over 5000 episodes
- Shows value distribution heatmaps

**Results**: ✅ 5000 episodes, 5.78s, optimal ε = 0.05

### Task 3: Comparative Analysis
Develops **decision framework** for algorithm selection:
- Compares 10 dimensions (speed, model req, scalability, etc.)
- Recommends algorithms for 8 scenarios
- Explains when to use each method

**Results**: ✅ Clear guidelines on DP vs MC trade-offs

## 🔧 Installation

```bash
# Clone
git clone <repo>
cd reinforcement-learning-maze

# Setup
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)

# Install
pip install -r requirements.txt
```

## ▶️ Run Examples

```bash
# All tasks (recommended)
python task1_dynamic_programming/task1_main.py
python task2_monte_carlo/task2_main.py
python task3_analysis/task3_main.py

# Or individual tasks
cd task1_dynamic_programming && python task1_main.py
```

Output: Console metrics + PNG visualizations in `results/`

## 💡 Key Insight

| | **DP** | **MC** |
|---|--------|--------|
| **Speed** | 0.08s | 5.78s |
| **Speedup** | **72× faster** | baseline |
| **Model** | Needed | Not needed |
| **Best For** | Simulations | Real world |

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 2 min setup guide
- **[README_GITHUB.md](README_GITHUB.md)** - Full docs
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to extend
- **[REPORT.pdf](REPORT.pdf)** - Detailed methodology & theory

## 🚀 Next Steps

1. Run the examples above
2. Check `results/` for visualizations
3. Read `REPORT.pdf` for theory
4. Explore code in `task*/`
5. Try modifying hyperparameters

## 📋 Requirements

- Python 3.8+
- NumPy, Matplotlib, SciPy (see requirements.txt)

## 📄 License

MIT - See [LICENSE](LICENSE)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on extending this project.

---

**Ready to dive in?** → [Start with QUICKSTART.md](QUICKSTART.md) ⚡

