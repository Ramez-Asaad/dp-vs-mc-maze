# 📚 Documentation Index

Quick links to navigate the repository:

## 🚀 Getting Started (Pick one)
- **[QUICKSTART.md](QUICKSTART.md)** - Fast 2-minute setup (Recommended!)
- **[README.md](README.md)** - Main repository overview

## 📖 Full Documentation
- **[README_GITHUB.md](README_GITHUB.md)** - Complete guide with all details
- **[REPORT.pdf](REPORT.pdf)** - Academic report with full methodology & theory
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to extend the project

## 📁 Project Code

### Task 1: Dynamic Programming
```
task1_dynamic_programming/
├── task1_main.py          # Run experiments: python task1_main.py
├── dp_algorithms.py       # Value Iteration implementation
└── environment_setup.py   # Maze environment
```

### Task 2: Monte Carlo
```
task2_monte_carlo/
├── task2_main.py          # Run experiments: python task2_main.py
├── mc_algorithms.py       # First-Visit MC implementation
└── environment_setup.py   # Maze environment
```

### Task 3: Comparative Analysis
```
task3_analysis/
└── task3_main.py          # Run comparison: python task3_main.py
```

## 📊 Results

All generated visualizations (15 PNG files) are in:
```
results/
├── task1_dp/              # 10 DP analysis plots
├── task2_mc/              # 3 MC convergence plots
└── task3_comparison/      # 2 comparative plots
```

## 🔍 Key Files Summary

| File | Purpose |
|------|---------|
| `README.md` | Start here - main overview |
| `QUICKSTART.md` | 2-min setup guide |
| `README_GITHUB.md` | Full documentation |
| `REPORT.pdf` | Academic report (detailed) |
| `CONTRIBUTING.md` | Extension guidelines |
| `LICENSE` | MIT License |
| `setup.py` | Package installer |
| `requirements.txt` | Dependencies |
| `.gitignore` | Git configuration |

## 📊 Quick Comparison

| | DP | MC |
|---|----|----|
| Speed | 0.08s | 5.78s |
| Model Required | Yes | No |
| Best For | Simulations | Real-world |
| Convergence | 16 iterations | 5000 episodes |

## 🎯 Quick Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run all tasks
python task1_dynamic_programming/task1_main.py
python task2_monte_carlo/task2_main.py
python task3_analysis/task3_main.py

# View results
ls results/task*/*.png
```

## 📚 Reading Guide

**For Quick Understanding**:
1. Read [README.md](README.md) (5 min)
2. Run examples (5 min)
3. Check visualizations in `results/` (5 min)

**For Complete Understanding**:
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Read [REPORT.pdf](REPORT.pdf) Introduction
3. Run all tasks
4. Explore code in `task*/`
5. Read [README_GITHUB.md](README_GITHUB.md) for details

**For Development**:
1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Examine `task1_main.py` and `dp_algorithms.py`
3. Modify and test your changes
4. Generate new visualizations

## 🤔 Common Questions

**Q: Where do I start?**
A: Open [QUICKSTART.md](QUICKSTART.md)

**Q: How do I run the code?**
A: Follow [QUICKSTART.md](QUICKSTART.md) or [README_GITHUB.md](README_GITHUB.md)

**Q: What are the key results?**
A: See the "Key Results" table in [README.md](README.md)

**Q: When to use DP vs MC?**
A: See "When to Use Each Method" in [README_GITHUB.md](README_GITHUB.md)

**Q: Can I modify the code?**
A: Yes! See [CONTRIBUTING.md](CONTRIBUTING.md)

**Q: What's the math behind it?**
A: Check [REPORT.pdf](REPORT.pdf)

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

**Happy learning!** 🚀
