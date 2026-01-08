# AoI-Aware Emergency Crowd-Finding: A Game-Theoretic Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LaTeX](https://img.shields.io/badge/LaTeX-IEEE%20Format-green.svg)](https://www.ieee.org/)

> **Game Theory Course Project** — University of Padova, 2024/25  
> **Author**: Antonio Tangaro  

---

## Abstract

This project analyzes strategic volunteer participation in emergency crowd-finding networks using **Age of Information (AoI)** as the core performance metric. We model the system as a public goods game where rational volunteers balance personal participation costs against shared detection benefits.

**Key contributions:**

- Closed-form expressions for Nash equilibrium and social optimum participation levels
- Characterization of the under-participation gap and Price of Anarchy
- Stackelberg incentive mechanism design to achieve socially optimal outcomes
- Validated simulation framework (static + dynamic) with interactive visualization

**Main finding:** Selfish behavior leads to systematic under-participation, causing up to **70% efficiency loss** in rescue success rate. Platform incentives can fully recover the social optimum.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Background](#background)
- [Theoretical Model](#theoretical-model)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Interactive Demo](#interactive-demo)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/antoniooodev/aoi_crowdfinding.git
cd aoi_crowdfinding
pip install -r requirements.txt

# Run validation and generate figures
python simulations/emergency_simulation.py

# Compile paper
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Background

### Problem Setting

Emergency crowd-finding networks (e.g., Apple Find My, Samsung SmartThings) leverage volunteers' devices to locate missing persons or objects. Each volunteer incurs a cost (battery, bandwidth, privacy) to participate, while detection benefits are shared by all.

<p align="center">
  <img src="https://i.ibb.co/gLvRqg6r/spatial-snapshot.png" alt="System Diagram" width="700"/>
</p>
<p align="center"><em>Figure 1: Spatial configuration showing N=100 volunteers (22 active, 78 inactive) searching for a target in area L². The detection zone (red circle) indicates the radius R within which volunteers can detect the target.</em></p>

### Research Questions

1. **Equilibrium**: How many volunteers participate under selfish behavior?
2. **Efficiency**: How does Nash equilibrium compare to the social optimum?
3. **Mechanism Design**: What incentives achieve optimal participation?

---

## Theoretical Model

### System Parameters

| Symbol | Description                    | Typical Value |
| ------ | ------------------------------ | ------------- |
| $L$    | Search area side length        | 500 m         |
| $N$    | Total volunteers               | 100           |
| $R$    | Detection radius               | 30 m          |
| $\rho$ | Coverage ratio $\pi R^2 / L^2$ | 0.011         |
| $B$    | Benefit from detection         | 10            |
| $c$    | Participation cost             | variable      |

### Key Formulas

**Detection probability** with $k$ active volunteers:

$$P_{\text{det}}(k) = 1 - (1-\rho)^k$$

**Nash equilibrium** participation:

$$k^* = \left\lfloor 1 + \frac{\ln(c / B\rho)}{\ln(1-\rho)} \right\rfloor$$

**Social optimum** participation:

$$k^{\text{opt}} = \left\lfloor 1 + \frac{\ln(c / NB\rho)}{\ln(1-\rho)} \right\rfloor$$

**Under-participation gap**:

$$\Delta k = k^{\text{opt}} - k^* \approx \frac{\ln(N)}{\rho}$$

**Optimal platform incentive** (Stackelberg):

$$p^* = c - B\rho(1-\rho)^{k^{\text{opt}}-1}$$

<p align="center">
  <img src="https://i.ibb.co/kgr0gwFM/complete-analysis.png" alt="Detection Probability Curve" width="600"/>
</p>
<p align="center"><em>Figure 2: Complete analysis showing (top row) P_det validation, relative error, and participation gap; (bottom row) theoretical AoI comparison, rescue success rate from dynamic simulation, and summary statistics. The shaded regions highlight efficiency loss from under-participation.</em></p>

---

## Project Structure

```
aoi_crowdfinding/
├── paper/                           # LaTeX paper (IEEE format)
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
├── src/                             # Core Python modules
│   ├── config.py                    # Parameters and configuration
│   ├── spatial.py                   # 2D geometry, coverage computation
│   ├── aoi.py                       # Age of Information formulas
│   ├── game.py                      # Nash equilibrium, social optimum, PoA
│   ├── stackelberg.py               # Incentive mechanism design
│   ├── simulation.py                # Monte Carlo simulation engine
│   └── visualization.py             # IEEE-style plotting utilities
│
├── simulations/
│   ├── emergency_simulation.py      # Main simulation script
│   └── emergency_visualization.jsx  # React interactive demo
│
├── docs/
│   ├── formal_model.md              # Complete mathematical derivations
│   ├── simulation_pipeline.md       # Simulation architecture details
│   ├── VALIDATION_VS_DEMONSTRATION.md
│   └── figures/                     # Documentation figures
│
├── results/
│   ├── data/                        # Raw CSV outputs
│   └── figures/                     # Generated plots
│
├── tests/                           # Unit tests
│   ├── test_game.py
│   ├── test_simulation.py
│   └── test_integration.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

### Requirements

- Python 3.9 or higher
- LaTeX distribution (TeX Live, MiKTeX) for paper compilation
- Node.js 18+ (optional, for interactive demo)

### Python Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21
matplotlib>=3.5
pandas>=1.3
scipy>=1.7
tqdm>=4.62
```

### Verify Installation

```bash
python -c "from src.game import compute_nash_k; print('OK')"
```

---

## Usage

### 1. Run Full Analysis

```bash
python simulations/emergency_simulation.py
```

This executes:

1. **Static validation**: Verifies P_det formula against Monte Carlo (i.i.d. positions)
2. **Cost sweep**: Analyzes Nash vs Optimal across cost ratios c/Bρ ∈ [0.1, 0.95]
3. **Figure generation**: Produces publication-ready plots

**Expected output:**

```
================================================================================
VALIDATION: P_det with STATIC simulation (i.i.d. positions)
================================================================================

k=  5: theory=0.0553, sim=0.0555±0.0006, error=0.4% ✓
k= 10: theory=0.1071, sim=0.1065±0.0009, error=0.6% ✓
k= 25: theory=0.2453, sim=0.2448±0.0011, error=0.2% ✓
...
Validation: 11/11 passed (<1% error)

================================================================================
COST SWEEP: Nash vs Social Optimum
================================================================================

c/Bρ=0.50: k*=61, k_opt=100, gap=39, Nash=100.0%, Opt=100.0%
c/Bρ=0.72: k*=30, k_opt=100, gap=70, Nash=87.0%, Opt=100.0%
c/Bρ=0.88: k*=12, k_opt=100, gap=88, Nash=42.0%, Opt=100.0%
...

Figures saved to results/figures/
```

### 2. Run Specific Experiments

```bash
# Validation only
python -c "from simulations.emergency_simulation import validate_P_det_static; validate_P_det_static()"

# Custom parameter sweep
python simulations/emergency_simulation.py --cost-min 0.3 --cost-max 0.9 --runs 100
```

### 3. Compile Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `paper/main.pdf`

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## Results

### Validation Results

The static simulation (i.i.d. volunteer positions) validates the analytical P_det formula with <1% relative error across all test cases.

| k   | P_det (Theory) | P_det (Simulation) | Error |
| --- | -------------- | ------------------ | ----- |
| 5   | 0.0553         | 0.0555 ± 0.0006    | 0.4%  |
| 25  | 0.2453         | 0.2448 ± 0.0011    | 0.2%  |
| 50  | 0.4337         | 0.4328 ± 0.0011    | 0.2%  |
| 100 | 0.6794         | 0.6775 ± 0.0012    | 0.3%  |

### Cost Sweep Results

<p align="center">
  <img src="https://i.ibb.co/KjRFXYWM/welfare-comparison.png" alt="Cost Sweep Results" width="700"/>
</p>
<p align="center"><em>Figure 3: Social welfare comparison. Nash equilibrium welfare (blue) collapses as cost increases, while the Stackelberg mechanism (red) successfully recovers the social optimum (green) through platform incentives.</em></p>

| Cost Ratio | k\* (Nash) | k_opt | Gap | Nash Success | Optimal Success | Efficiency Loss |
| ---------- | ---------- | ----- | --- | ------------ | --------------- | --------------- |
| 0.10       | 100        | 100   | 0   | 100%         | 100%            | 0%              |
| 0.50       | 61         | 100   | 39  | 100%         | 100%            | 0%              |
| 0.72       | 30         | 100   | 70  | 87%          | 100%            | 13%             |
| 0.88       | 12         | 100   | 88  | 42%          | 100%            | 58%             |
| 0.95       | 5          | 100   | 95  | 18%          | 100%            | 82%             |

**Key insight:** At high cost ratios (c/Bρ > 0.8), Nash equilibrium participation collapses, causing dramatic efficiency losses that platform incentives can fully recover.

---

## Interactive Demo

The React-based visualization (`simulations/emergency_visualization.jsx`) provides an interactive exploration of the under-participation phenomenon.

<p align="center">
  <img src="https://i.ibb.co/HTf0n6Tz/simulation-validation.png"
   alt="Interactive Demo Screenshot" width="800"/>
</p>
<p align="center"><em>Figure 4: Interactive simulation comparing Nash equilibrium (left) vs social optimum (right) under identical initial conditions. The batch run feature provides statistical validation.</em></p>

### Features

| Feature                     | Description                                |
| --------------------------- | ------------------------------------------ |
| **Side-by-side comparison** | Nash vs Optimal with shared random seed    |
| **Parameter controls**      | Cost ratio, area size, volunteer count     |
| **Batch runs**              | 10-200 runs with aggregated statistics     |
| **Real-time metrics**       | AoI, detections, coverage, rescue progress |
| **P_det visualization**     | Theoretical curve with equilibrium markers |

### Setup

```bash
cd demos/emergency/ui
npm install
npm install recharts
npm run dev
```

Open `http://localhost:5173` in browser.

### Alternative: Online Playground

Copy `emergency_visualization.jsx` to [CodeSandbox](https://codesandbox.io) or [StackBlitz](https://stackblitz.com) with a React template.

---

## Documentation

| Document                                                                     | Description                              |
| ---------------------------------------------------------------------------- | ---------------------------------------- |
| [`docs/formal_model.md`](docs/formal_model.md)                               | Complete mathematical derivations        |
| [`docs/simulation_pipeline.md`](docs/simulation_pipeline.md)                 | Simulation architecture and module specs |
| [`docs/VALIDATION_VS_DEMONSTRATION.md`](docs/VALIDATION_VS_DEMONSTRATION.md) | Static vs dynamic simulation methodology |

### Simulation Methodology

The framework distinguishes between **validation** (static) and **demonstration** (dynamic) simulations:

| Mode    | Positions             | Target | Purpose                 | Validates Theory? |
| ------- | --------------------- | ------ | ----------------------- | ----------------- |
| Static  | i.i.d. each step      | Fixed  | Validate P_det formula  | ✅ Yes            |
| Dynamic | Correlated (movement) | Mobile | Show qualitative effect | ❌ No             |

This separation is critical: the analytical model assumes independent positions, which the dynamic simulation intentionally violates to demonstrate real-world relevance.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{tangaro2025aoi,
  author       = {Tangaro, Antonio},
  title        = {{AoI}-Aware Emergency Crowd-Finding: A Game-Theoretic Analysis},
  year         = {2025},
  institution  = {University of Padova},
  note         = {Game Theory Course Project, supervised by Prof. Leonardo Badia}
}
```

### Key References

1. L. Badia, "Age of Information from Multiple Strategic Sources," _IEEE WiOpt_, 2021.
2. L. Badia et al., "Freshness and Forgetting in Federated Data Ecosystems," _IEEE Trans. Netw. Sci. Eng._, 2023.
3. R. D. Yates et al., "Age of Information: An Introduction and Survey," _IEEE JSAC_, 2021.
4. S. Heinrich et al., "Who Can Find My Devices?," _PETS_, 2021.

Full bibliography available in [`paper/references.bib`](paper/references.bib).

---

## Limitations and Future Work

### Current Limitations

- **Homogeneous agents**: All volunteers have identical cost c
- **Complete information**: Game parameters known to all players
- **Single-shot game**: No repeated interactions or learning
- **Uniform spatial model**: Theory assumes uniform volunteer distribution

### Future Extensions

- [ ] Heterogeneous costs (Bayesian mechanism design)
- [ ] Repeated games with reputation systems
- [ ] Multiple concurrent targets (competing searches)
- [ ] Spatial correlations and clustering effects
- [ ] Privacy-aware utility functions

---

## Troubleshooting

### Common Issues

**Import errors:**

```bash
# Ensure you're in the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**LaTeX compilation fails:**

```bash
# Install missing packages
tlmgr install ieeetran cite algorithmic
```

**React demo not loading:**

```bash
# Verify recharts is installed
npm list recharts || npm install recharts
```

### Getting Help

Open an issue on GitHub or contact the author.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Prof. Leonardo Badia** for course instruction and methodological guidance
- Badia's research on AoI and game theory as the primary theoretical foundation
- The open-source community for NumPy, Matplotlib, React, and Recharts

---

<p align="center">
  <strong>University of Padova — Department of Information Engineering</strong><br>
  Game Theory Course, A.Y. 2024/25
</p>
