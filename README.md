# Monte Carlo Simulation & Statistical Modeling

![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

A vectorized Monte Carlo simulation engine for analyzing complex probability distributions, validating statistical theory, and pricing financial derivatives — built from scratch with Python and NumPy.

---

## Overview

This project implements a full Monte Carlo pipeline across three modules:

| Module | Description |
|---|---|
| `monte_carlo.py` | Vectorized simulation engine — 4 distributions + Black-Scholes option pricer |
| `analysis.py` | Statistical validation against theoretical moments + convergence analysis |
| `plots.py` | Interactive Plotly dashboard — histograms, convergence plots, option pricing |

---

## Results

### Distribution Validation — 1,000,000 paths

All four distributions pass goodness-of-fit tests (KS / χ²) with p-values well above 0.05. Empirical moments match theoretical values to 4+ significant figures.

| Distribution | Empirical Mean | Theoretical Mean | Empirical Std | Theoretical Std | Test | p-value |
|---|---|---|---|---|---|---|
| Normal (μ=0, σ=1) | 0.000098 | 0.000000 | 1.000483 | 1.000000 | KS | 0.1871 ✓ |
| Log-Normal (μ=0, σ=0.5) | 1.133568 | 1.133148 | 0.603999 | 0.603901 | KS | 0.6946 ✓ |
| Exponential (λ=2) | 0.499858 | 0.500000 | 0.498916 | 0.500000 | KS | 0.3999 ✓ |
| Poisson (λ=5) | 5.000647 | 5.000000 | 2.235591 | 2.236068 | χ² | 0.2271 ✓ |

### Convergence Rates (CLT predicts −0.50)

The sample mean converges to the true mean at exactly the rate predicted by the Central Limit Theorem across all distributions.

| Distribution | True Mean | Empirical Slope |
|---|---|---|
| Normal | 0.0000 | −0.5091 |
| Log-Normal | 1.1331 | −0.5031 |
| Exponential | 0.5000 | −0.4867 |
| Poisson | 5.0000 | −0.5066 |

### European Call Option Pricing

Black-Scholes Monte Carlo vs analytical solution (S₀=100, K=100, T=1yr, r=5%, σ=20%):

| Method | Price | Error |
|---|---|---|
| Black-Scholes Analytical | 10.4506 | — |
| Monte Carlo (1M paths) | 10.4284 | 0.0222 (<0.3%) |

---

## Project Structure

```
monte-carlo-statistical-modeling/
├── monte_carlo.py      # Simulation engine
├── analysis.py         # Statistical validation & convergence
├── plots.py            # Plotly visualizations & dashboard
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/AbdellahKah/monte-carlo-statistical-modeling
cd monte-carlo-statistical-modeling
pip install -r requirements.txt
```

---

## Usage

```python
from monte_carlo import MonteCarloEngine
from analysis import MCAnalyser

# Run simulations
engine = MonteCarloEngine(n_paths=1_000_000, seed=42)
result = engine.simulate_normal(mu=0, sigma=1)
print(f"Mean: {result.mean:.6f} | Std: {result.std:.6f}")

# Validate against theory
analyser = MCAnalyser(engine)
analyser.full_report()

# Price a European call
call = engine.price_european_call(S0=100, K=100, T=1, r=0.05, sigma=0.2)
print(f"MC Price: {call['price']:.4f} | 95% CI: {call['ci_95']}")
```

```bash
# Generate full interactive dashboard
python plots.py
# Opens dashboard.html in your browser
```

---

## Key Concepts

**Monte Carlo Method** — estimates quantities by simulating a large number of random samples. Accuracy scales as 1/√n by the Central Limit Theorem — confirmed empirically here with slopes ≈ −0.50.

**Goodness-of-fit testing** — KS test for continuous distributions (Normal, Log-Normal, Exponential), chi-squared test for discrete (Poisson). All pass at the 5% significance level.

**Black-Scholes Monte Carlo** — simulates geometric Brownian motion paths under the risk-neutral measure, discounts expected payoffs. Converges to the closed-form BS price within 0.3% at 1M paths.

---

## Stack

`Python 3.10+` · `NumPy` · `SciPy` · `Plotly` · `LaTeX` (report)

---

## Author

**Abdellah Kahlaoui** — Master of Applied Mathematics, FST Settat

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/kahabdu1808)
[![GitHub](https://img.shields.io/badge/GitHub-AbdellahKah-181717?style=flat-square&logo=github)](https://github.com/AbdellahKah)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
