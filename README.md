# Age of Information in Emergency Crowd-Finding: A Game-Theoretic Analysis

> **Course Project** — Game Theory, University of Padova  
> **Author**: Antonio Tangaro
> **Supervisor**: Prof. Leonardo Badia

## Overview

This project analyzes **strategic volunteer participation** in emergency crowd-finding networks (e.g., Apple Find My) using **Age of Information (AoI)** as the performance metric. We model the system as a public goods game where volunteers balance personal costs against shared benefits, derive closed-form equilibrium solutions, and design platform incentives to achieve socially optimal outcomes.

### Key Question

> In a crowd-sourced emergency search, how many volunteers will participate voluntarily, and how does this compare to the social optimum?

### Main Results

| Finding | Formula |
|---------|---------|
| Nash equilibrium | $k^* = \lfloor 1 + \ln(c/B\rho) / \ln(1-\rho) \rfloor$ |
| Social optimum | $k^{\text{opt}} = \lfloor 1 + \ln(c/NB\rho) / \ln(1-\rho) \rfloor$ |
| Under-participation gap | $\Delta k \approx \ln(N) / \rho$ |
| Optimal incentive | $p^* = c - B\rho(1-\rho)^{k^{\text{opt}}-1}$ |

**Bottom line**: Selfish behavior leads to systematic under-participation, causing up to 70% efficiency loss in rescue success rate. Platform incentives can fully recover the social optimum.

---

## Project Structure

```
aoi_crowdfinding/
│
├── paper/                              # LaTeX paper (IEEE format)
│   ├── main.tex                        # Main document
│   ├── references.bib                  # Bibliography
│   └── figures/                        # Paper figures
│
├── src/                                # Core theory implementation
│   ├── __init__.py
│   ├── config.py                       # Parameters, thresholds
│   ├── spatial.py                      # 2D geometry, coverage
│   ├── aoi.py                          # AoI computations
│   ├── game.py                         # Nash, social optimum, PoA
│   ├── stackelberg.py                  # Incentive design
│   ├── simulation.py                   # Monte Carlo (static)
│   └── visualization.py                # IEEE-style plots
│
├── simulations/
│   ├── emergency_simulation_v3.py      # Final corrected simulation
│   └── emergency_visualization.jsx     # React interactive demo
│
├── results/
│   ├── data/                           # CSV outputs
│   └── figures/                        # Generated plots
│
├── docs/
│   ├── formal_model.md                 # Complete mathematical model
│   ├── simulation_pipeline.md          # Simulation architecture
│   └── VALIDATION_VS_DEMONSTRATION.md  # Conceptual separation
│
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

---

## Installation

### Requirements

- Python 3.9+
- LaTeX distribution (for paper compilation)

### Setup

```bash
# Clone or extract the project
cd aoi_crowdfinding

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21
matplotlib>=3.5
pandas>=1.3
tqdm>=4.62
```

---

## Usage

### 1. Run Validation + Cost Sweep

```bash
python simulations/emergency_simulation.py
```

**Output**:
- `results/figures/complete_analysis.png` — 6-panel analysis figure
- Console output with validation results and cost sweep summary

**Expected output**:
```
VALIDATION: P_det with STATIC simulation (i.i.d. positions)
k=  5: theory=0.0553, sim=0.0555±0.0006, error=0.4% ✓
k= 50: theory=0.4337, sim=0.4328±0.0011, error=0.2% ✓
k=100: theory=0.6794, sim=0.6775±0.0012, error=0.3% ✓

Passed: 11/11 (<10% error)
```

### 2. Compile the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Output**: `main.pdf` (5-7 pages IEEE format)

### 3. Interactive Visualization (React)

The file `simulations/emergency_visualization.jsx` provides an interactive demo. To use:
1. Copy into a React project, or
2. Use an online React playground (CodeSandbox, StackBlitz)

---

## Theoretical Model

### System

- **Area**: $[0, L]^2$ square region
- **Volunteers**: $N$ agents with detection radius $R$
- **Coverage ratio**: $\rho = \pi R^2 / L^2$
- **Target**: Missing person at unknown location

### AoI Dynamics

Detection probability with $k$ active volunteers:
$$P_{\text{det}}(k) = 1 - (1-\rho)^k$$

Expected Age of Information:
$$\bar{\Delta}(k) = \frac{(1-\rho)^k}{1 - (1-\rho)^k}$$

### Game Model

**Utility function**:
$$u_i(a_i, k_{-i}) = a_i \cdot \left[ B \rho (1-\rho)^{k_{-i}} - c \right]$$

- $B$ = benefit from fresh information
- $c$ = participation cost
- $a_i \in \{0, 1\}$ = volunteer $i$'s action

**Nash Equilibrium**: Largest $k$ such that marginal benefit ≥ cost:
$$k^* = \left\lfloor 1 + \frac{\ln(c / B\rho)}{\ln(1-\rho)} \right\rfloor$$

**Social Optimum**: Maximizes total welfare $W(k) = NB[1 - (1-\rho)^k] - ck$

---

## Simulation Framework

### Two Distinct Modes

| Mode | Purpose | Validates Theory? |
|------|---------|-------------------|
| **Static** | Validate $P_{\text{det}}$ formula | ✅ Yes (i.i.d. positions) |
| **Dynamic** | Demonstrate qualitative effect | ❌ No (correlated movement) |

### Why Two Modes?

The analytical formula assumes **i.i.d. volunteer positions** at each time step. The dynamic agent-based simulation introduces:
- Temporal correlation (agents move toward waypoints)
- Exploration effects (agents cover more area over time)

These effects make dynamic $P_{\text{det}}$ **higher** than theory (agents explore), but the **qualitative under-participation phenomenon persists**.

See `docs/VALIDATION_VS_DEMONSTRATION.md` for detailed explanation.

---

## Key Results

### Validation (Static Simulation)

All 11 test cases pass with <1% relative error, confirming:
$$P_{\text{det}}^{\text{sim}} = P_{\text{det}}^{\text{theory}} \pm \text{Monte Carlo error}$$

### Cost Sweep (Dynamic Simulation)

| $c/(B\rho)$ | $k^*$ | $k^{\text{opt}}$ | Gap | Nash Success | Optimal Success |
|-------------|-------|------------------|-----|--------------|-----------------|
| 0.10 | 100 | 100 | 0 | 100% | 100% |
| 0.50 | 61 | 100 | 39 | 100% | 100% |
| 0.72 | 30 | 100 | 70 | 87% | 100% |
| 0.95 | 5 | 100 | 95 | 30% | 100% |

**Insight**: As cost approaches $B\rho$, Nash participation collapses, causing up to 70% efficiency loss.

---

## Paper Structure

| Section | Content | Pages |
|---------|---------|-------|
| I. Introduction | Motivation, contributions | ~0.5 |
| II. Related Work | AoI, game theory, MCS | ~0.5 |
| III. System Model | Area, detection, AoI, utility | ~1 |
| IV. Game Analysis | Nash, optimum, PoA, Stackelberg | ~1.5 |
| V. Numerical Results | Validation, cost sweep, dynamic | ~1.5 |
| VI. Conclusions | Summary, limitations, future work | ~0.5 |

**Total**: ~5-6 pages (IEEE double-column)

---

## References

Key papers this work builds upon:

1. **Badia (2021)** — AoI from strategic sources
2. **Badia et al. (2023)** — Federated data ecosystems
3. **Yates et al. (2021)** — AoI survey
4. **Dasari et al. (2020)** — Game theory in mobile crowdsensing
5. **Heinrich et al. (2021)** — Find My network analysis

Full bibliography in `paper/references.bib`.

---

## Limitations

- **Homogeneous volunteers**: All have same cost $c$
- **Complete information**: Parameters known to all
- **Single-shot game**: No repeated interactions
- **Static coverage model**: Theory assumes i.i.d. positions

### Future Work

- Heterogeneous costs (Bayesian games)
- Repeated interactions (dynamic incentives)
- Multiple targets (competing searches)
- Spatial correlations (clustered volunteers)

---

## License

Academic project — University of Padova, 2025.

---

## Acknowledgments

- Prof. Leonardo Badia for the course and methodological guidance
- Badia's papers on AoI and game theory as the primary methodological template
