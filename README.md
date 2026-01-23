# AoI-Aware Emergency Crowd-Finding: A Game-Theoretic Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LaTeX](https://img.shields.io/badge/LaTeX-IEEE%20Format-green.svg)](https://www.ieee.org/)

> **Game Theory Course Project** — University of Padova, 2025/26  
> **Author**: Antonio Tangaro

---

## Abstract

This project analyzes strategic volunteer participation in emergency crowd-finding networks, modeling detection as a public good and timeliness via **Age of Information (AoI)**. We model the system as a public goods game where rational volunteers balance personal participation costs against shared detection benefits.

The main contributions include closed-form expressions for Nash equilibrium and social optimum participation levels, a characterization of the under-participation gap and Price of Anarchy, a Stackelberg incentive mechanism to achieve socially optimal outcomes, and a validated simulation framework with interactive visualization. We also extend the analysis to **heterogeneous costs**, where each volunteer has a private participation cost drawn from a distribution.

**Main finding:** Selfish behavior leads to systematic under-participation, causing up to **53% efficiency loss** in rescue success rate. Platform incentives can fully recover the social optimum.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Background](#background)
3. [Theoretical Model](#theoretical-model)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Interactive Demo](#interactive-demo)
9. [Documentation](#documentation)
10. [Citation](#citation)
11. [License](#license)

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/antoniooodev/aoi_crowdfinding.git
cd aoi_crowdfinding
pip install -r requirements.txt

# Run validation and generate figures
python demos/emergency/emergency_simulation.py

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

The project addresses three fundamental questions. First, the **equilibrium problem**: how many volunteers participate under selfish behavior? Second, the **efficiency problem**: how does Nash equilibrium compare to the social optimum? Third, the **mechanism design problem**: what incentives achieve optimal participation?

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

### Key Formulas (Homogeneous Model)

**Detection probability** with $k$ active volunteers:

$$P_{\text{det}}(k) = 1 - (1-\rho)^k$$

**Nash equilibrium** participation:

$$k^* = \left\lfloor 1 + \frac{\ln(c / B\rho)}{\ln(1-\rho)} \right\rfloor$$

**Social optimum** participation:

$$k^{\text{opt}} = \left\lfloor 1 + \frac{\ln(c / NB\rho)}{\ln(1-\rho)} \right\rfloor$$

**Under-participation gap**:

$$\Delta k = k^{\text{opt}} - k^* \approx \frac{\ln(N)}{-\ln(1-\rho)} \approx \frac{\ln(N)}{\rho}\quad(\rho\ll 1),\;\; \Delta k \le N$$

**Minimum implementing subsidy** (Stackelberg):

$$p^* = \max\{0,\;c - B\rho(1-\rho)^{k^{\text{opt}}-1}\}$$

### Heterogeneous Cost Extension

In the extended model, each volunteer $i$ has a private cost $c_i$ drawn i.i.d. from a distribution $F$ on $[c_{\min}, c_{\max}]$. The Nash equilibrium becomes a threshold equilibrium: volunteer $i$ participates if and only if $c_i \leq \bar{c}(k^*)$, where the threshold $\bar{c}(k) = B\rho(1-\rho)^k$ and $k^* = N \cdot F(\bar{c}(k^*))$. This extension captures realistic scenarios where volunteers have heterogeneous opportunity costs. See `docs/formal_model.md` for complete derivations.

<p align="center">
  <img src="https://i.ibb.co/x82yHgjx/complete-analysis.png" alt="Detection Probability Curve" width="600"/>
</p>
<p align="center"><em>Figure 2: Complete analysis showing P_det validation, participation gap, and efficiency loss for both homogeneous and heterogeneous cost models.</em></p>

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
│   ├── config.py                    # Parameters, HeterogeneousCostParams
│   ├── spatial.py                   # 2D geometry, coverage computation
│   ├── aoi.py                       # Age of Information formulas
│   ├── game.py                      # Nash, social optimum, PoA (both models)
│   ├── stackelberg.py               # Incentive mechanism design
│   ├── simulation.py                # Monte Carlo validation engine
│   └── visualization.py             # IEEE-style plotting utilities
│
├── experiments/                     # Reproducible experiments
│   ├── exp01_equilibrium_analysis.py
│   ├── exp02_social_optimum.py
│   ├── exp03_price_of_anarchy.py
│   ├── exp04_stackelberg.py
│   ├── exp05_sensitivity.py
│   ├── exp06_heterogeneous_analysis.py
│   └── run_all.py
│
├── demos/emergency/                 # Simulation and visualization
│   ├── emergency_simulation.py      # Main simulation (hom + het)
│   └── ui/                          # React interactive demo
│
├── docs/
│   ├── formal_model.md              # Homogeneous model derivations
│   ├── formal_model_v2.md           # Heterogeneous cost extension
│   └── simulation_pipeline.md       # Simulation architecture
│
├── results/
│   ├── data/                        # Raw CSV outputs
│   └── figures/                     # Generated plots
│
├── tests/                           # Unit tests
│   ├── test_game.py
│   ├── test_simulation.py
│   ├── test_heterogeneous.py
│   ├── test_config.py
│   ├── test_stackelberg.py
│   ├── test_aoi.py
│   └── test_spatial.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

### Requirements

The project requires Python 3.9 or higher. For paper compilation, a LaTeX distribution (TeX Live or MiKTeX) is needed. The interactive demo optionally requires Node.js 18+.

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

The core dependencies are numpy (≥1.21), matplotlib (≥3.5), pandas (≥1.3), scipy (≥1.7), and tqdm (≥4.62).

### Verify Installation

```bash
python -c "from src.game import find_nash_equilibrium; print('OK')"
```

---

## Usage

### Run Full Analysis

```bash
python demos/emergency/emergency_simulation.py
```

This executes static validation of the P_det formula against Monte Carlo simulation, a cost sweep analyzing Nash vs Optimal across cost ratios, comparison between homogeneous and heterogeneous models, and generates publication-ready figures.

**Expected output:**

```
============================================================
EMERGENCY CROWD-FINDING SIMULATION
============================================================

Configuration:
  L = 500m, N = 100, R = 30m
  ρ = 0.011310
  Use heterogeneous model: True

STEP 1: Validate P_det (static i.i.d. simulation)
k=  5: theory=0.0553, sim=0.0555±0.0006, error=0.4% ✓
k= 50: theory=0.4337, sim=0.4328±0.0011, error=0.2% ✓
k=100: theory=0.6794, sim=0.6775±0.0012, error=0.3% ✓
Passed: 11/11 (<10% error)

STEP 2a: Cost sweep - Homogeneous model
STEP 2b: Cost sweep - Heterogeneous model

Figures saved to results/figures/
```

### Run All Experiments

```bash
python -m experiments.run_all
```

This runs experiments 01–06 and generates all figures for the paper.

### Compile Paper

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Results

### Validation Results

The static simulation (i.i.d. volunteer positions) validates the $P_{\text{det}}(k)$ formula with <1% relative error in the reported cases; the console “Passed” line uses a 10% acceptance threshold.

| k   | P_det (Theory) | P_det (Simulation) | Error |
| --- | -------------- | ------------------ | ----- |
| 5   | 0.0553         | 0.0555 ± 0.0006    | 0.4%  |
| 25  | 0.2453         | 0.2448 ± 0.0011    | 0.2%  |
| 50  | 0.4337         | 0.4328 ± 0.0011    | 0.2%  |
| 100 | 0.6794         | 0.6775 ± 0.0012    | 0.3%  |

### Homogeneous vs Heterogeneous Comparison

With heterogeneous costs (spread ratio $c_{\max}/c_{\min} = 2$), the participation gap is slightly smaller because low-cost volunteers consistently participate even when the mean cost is high. This results in more stable Nash equilibrium participation and lower efficiency loss compared to the homogeneous model.

| Model         | Avg Gap | Avg Efficiency Loss | Max Efficiency Loss |
| ------------- | ------- | ------------------- | ------------------- |
| Homogeneous   | 41.8    | 9.2%                | 53.3%               |
| Heterogeneous | 40.2    | 3.9%                | 10.0%               |

<p align="center">
  <img src="https://i.ibb.co/KjRFXYWM/welfare-comparison.png" alt="Cost Sweep Results" width="700"/>
</p>
<p align="center"><em>Figure 3: Social welfare comparison. Nash vs. social optimum; platform incentives recover the social optimum in the homogeneous model and recover the socially optimal participation level in expectation under heterogeneous private costs.</em></p>

---

## Interactive Demo

The React-based visualization provides an interactive exploration of the under-participation phenomenon with side-by-side comparison of Nash equilibrium vs social optimum under identical initial conditions.

<p align="center">
  <img src="https://i.ibb.co/HTf0n6Tz/simulation-validation.png" alt="Interactive Demo Screenshot" width="800"/>
</p>
<p align="center"><em>Figure 4: Interactive simulation comparing Nash equilibrium (left) vs social optimum (right). The batch run feature provides statistical validation.</em></p>

### Setup

```bash
cd demos/emergency/ui
npm install
npm install recharts
npm run dev
```

Open `http://localhost:5173` in browser.

### Alternative

Copy the visualization component to [CodeSandbox](https://codesandbox.io) or [StackBlitz](https://stackblitz.com) with a React template.

---

## Documentation

| Document                      | Description                                     |
| ----------------------------- | ----------------------------------------------- |
| `docs/formal_model.md`        | Homogeneous and Heterogeneous model derivations |
| `docs/simulation_pipeline.md` | Simulation architecture and module specs        |

### Simulation Methodology

The framework distinguishes between validation and demonstration modes. **Static simulation** uses i.i.d. positions each step with a fixed target to validate the P_det formula exactly. **Dynamic simulation** uses correlated positions (agent movement) with a mobile target to show qualitative effects in realistic scenarios. This separation is critical because the analytical model assumes independent positions, which the dynamic simulation intentionally violates to demonstrate real-world relevance.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{tangaro2025aoi,
  author       = {Tangaro, Antonio},
  title        = {{AoI}-Aware Emergency Crowd-Finding: A Game-Theoretic Analysis},
  year         = {2026},
  institution  = {University of Padova},
  note         = {Game Theory Course Project, supervised by Prof. Leonardo Badia}
}
```

### Key References

The theoretical foundation builds primarily on: (i) game-theoretic AoI with strategic sources, (ii) efficiency loss / Price of Anarchy under selfish participation, (iii) Stackelberg incentives for AoI-aware crowdsensing, and (iv) real-world crowd-finding and privacy-preserving object-finding systems. Full bibliography is available in `paper/references.bib`.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Thanks to **Prof. Leonardo Badia** for course instruction and methodological guidance, whose research on AoI and game theory provides the primary theoretical foundation for this work.

---

<p align="center">
  <strong>University of Padova — Department of Information Engineering</strong><br>
  Game Theory Course, A.Y. 2025/26
</p>
