# Simulation Framework: Validation vs Demonstration

## Overview

This simulation framework serves **two distinct purposes** that must not be conflated:

| Component                       | Purpose                              | Validates Theory?              |
| ------------------------------- | ------------------------------------ | ------------------------------ |
| **Static Simulation (Python)**  | Validate analytical formulas         | ✅ Yes, exactly                |
| **Dynamic Simulation (Python)** | Demonstrate qualitative phenomenon   | ❌ No, intentionally different |
| **Interactive UI (React)**      | Visualize and explore the phenomenon | ❌ Demonstration only          |

---

## Part 1: Static Simulation (Validation)

### Purpose

Verify that the analytical formula $P_{\text{det}}(k) = 1 - (1-\rho)^k$ matches Monte Carlo simulation.

### Assumptions (match theory exactly)

- Volunteer positions: **i.i.d. uniform** in $[0,L]^2$ at each time step
- Target position: **fixed** in interior $[R, L-R]^2$
- No temporal correlation between steps
- Detection is instantaneous

### Implementation

```python
def simulate_static(config, k, n_steps, seed):
    """Each step: fresh i.i.d. positions (matches theory)."""
    for step in range(n_steps):
        # Fresh positions every step - NO movement, NO correlation
        vol_x = rng.uniform(0, L, size=k)
        vol_y = rng.uniform(0, L, size=k)
        distances = sqrt((vol_x - target_x)² + (vol_y - target_y)²)
        detected = any(distances <= R)
```

### Expected Result

$$P_{\text{det}}^{\text{empirical}} = P_{\text{det}}^{\text{analytical}} \pm \text{Monte Carlo error}$$

### Actual Result

| k   | Theory | Simulation    | Error  |
| --- | ------ | ------------- | ------ |
| 5   | 0.0553 | 0.0555±0.0006 | 0.4% ✓ |
| 50  | 0.4337 | 0.4328±0.0011 | 0.2% ✓ |
| 100 | 0.6794 | 0.6775±0.0012 | 0.3% ✓ |

**Conclusion**: Theory validated (11/11 tests passed with <1% error).

---

## Part 2: Dynamic Simulation (Qualitative Demonstration)

### Purpose

Show that the **qualitative phenomenon** of under-participation persists in a more realistic setting with:

- Moving agents
- Mobile target
- Time-dependent rescue logic

### Assumptions (intentionally different from theory)

- Volunteers: **move toward waypoints** (correlated positions across time)
- Target: **random walk** within interior
- Rescue: requires **sustained low AoI** during response period

### Why This Doesn't Validate the Formula

The analytical model assumes:
$$P_{\text{det}}(k) = 1 - (1-\rho)^k \quad \text{with } \rho = \frac{\pi R^2}{L^2}$$

This requires **independent** positions at each step. In the dynamic simulation:

1. **Temporal correlation**: A volunteer near the target at $t$ is likely still near at $t+1$
2. **Exploration effect**: Volunteers "explore" the area over time
3. **Variable effective ρ**: Depends on volunteer distribution, not uniform

### What It Does Show

The dynamic simulation demonstrates that:

1. **Under-participation gap persists**: $k^* < k^{\text{opt}}$ leads to worse outcomes
2. **Rescue success degrades with cost**: As $c \to B\rho$, Nash success rate drops
3. **AoI matters causally**: Rescue requires sustained detection, not just one-time

---

## Part 3: Interactive UI (React)

### Purpose

Provide an interactive visualization for exploring the under-participation phenomenon. This is for **educational and demonstration purposes**, not validation.

### Key Features

| Feature                  | Description                                     | Purpose                                |
| ------------------------ | ----------------------------------------------- | -------------------------------------- |
| **Shared Seed**          | Nash and Optimal start with identical positions | Fair comparison (isolates effect of k) |
| **Batch Runs**           | 10-200 runs with aggregated statistics          | Statistical robustness                 |
| **Monte Carlo Coverage** | Real-time coverage estimation (800 samples)     | Visualize spatial coverage             |
| **P_det Curve**          | Recharts visualization of theoretical curve     | Connect simulation to theory           |
| **AoI History**          | Comparative time series of AoI                  | Show detection frequency difference    |
| **View Modes**           | Nash / Comparison / Optimal                     | Focus on specific scenarios            |

### Shared Seed Implementation

Critical for fair comparison: both simulations start from **identical initial conditions**.

```javascript
const createSimulationPair = (N, L, kNash, kOpt, seed) => {
  const rng = createRNG(seed);

  // Same positions for all N volunteers
  const baseVolunteers = [];
  for (let i = 0; i < N; i++) {
    baseVolunteers.push({
      x: rng() * L,
      y: rng() * L,
      waypointX: rng() * L,
      waypointY: rng() * L,
    });
  }

  // Same shuffle order
  const indices = shuffleDeterministic(N, rng);

  // Nash: activate first k* volunteers
  // Optimal: activate first k_opt volunteers (includes all k* from Nash)
  // ...
};
```

**Key insight**: The only difference between Nash and Optimal is **which volunteers are active**. Optimal activates all volunteers that Nash activates, plus additional ones. This ensures any performance difference is due to the participation gap, not random initial conditions.

### Rescue Logic

The UI implements the same rescue logic as the Python dynamic simulation:

```javascript
// Rescue requires sustained detection
if (detected) {
  this.aoi = 0;
  if (!this.rescueStarted) {
    this.rescueStarted = true;
    this.rescueStartTime = this.time;
  }
} else {
  this.aoi++;
}

// Check rescue progress
if (this.rescueStarted) {
  if (this.aoi > aoiThreshold) {
    // Lost contact - rescue aborted
    this.rescueStarted = false;
    this.rescueProgress = 0;
  } else if (this.time - this.rescueStartTime >= responseTime) {
    // Sustained contact - rescue successful
    this.outcome = "rescued";
  }
}
```

### Parameters

| Parameter            | Value     | Notes                          |
| -------------------- | --------- | ------------------------------ |
| R (detection radius) | 30        | Fixed                          |
| B (benefit)          | 10        | Fixed                          |
| responseTime         | 50 steps  | Time needed to complete rescue |
| aoiThreshold         | 25 steps  | Max AoI during rescue          |
| maxTime              | 400 steps | Simulation timeout             |

User-configurable:

- **Cost ratio** c/Bρ: 0.1 to 0.95
- **Area size** L: 300 to 800
- **Volunteers** N: 50, 100, 150, 200

### Presets

| Preset   | c/Bρ | N   | L   | Expected Behavior                |
| -------- | ---- | --- | --- | -------------------------------- |
| Easy     | 0.25 | 100 | 400 | Small gap, both succeed          |
| Medium   | 0.50 | 100 | 500 | Moderate gap                     |
| Hard     | 0.70 | 100 | 600 | Large gap, Nash struggles        |
| Critical | 0.88 | 100 | 700 | Near-threshold, Nash often fails |

---

## Validation Checklist

### Formula Validation (manual)

For any parameter set, verify:

```
ρ = π × R² / L²

k* = floor(1 + ln(c / Bρ) / ln(1 - ρ))
k_opt = floor(1 + ln(c / NBρ) / ln(1 - ρ))

P_det(k) = 1 - (1 - ρ)^k
```

### Sanity Checks

| Check                             | Expected      | If Fails             |
| --------------------------------- | ------------- | -------------------- |
| k\* ≤ k_opt                       | Always        | Formula bug          |
| P_det(0) = 0                      | Always        | Formula bug          |
| P_det monotonic in k              | Always        | Formula bug          |
| c → 0 ⟹ k\* → N                   | Always        | Formula bug          |
| c → Bρ ⟹ k\* → 0                  | Always        | Formula bug          |
| Batch: success_opt ≥ success_nash | Statistically | Bug or high variance |

### Cross-validation Python ↔ JavaScript

Both implementations must produce identical results for the same inputs:

```python
# Python
config = Config(L=500, N=100, R=30, B=10)
c = 0.5 * config.B_rho
print(f"k* = {compute_k_star(100, config.rho, 10, c)}")
print(f"k_opt = {compute_k_opt(100, config.rho, 10, c)}")
```

```javascript
// JavaScript
const rho = computeRho(30, 500);
const c = 0.5 * 10 * rho;
console.log("k* =", computeNashK(100, rho, 10, c));
console.log("k_opt =", computeOptimalK(100, rho, 10, c));
```

---

## Summary Table

| Aspect            | Static (Python)      | Dynamic (Python)         | Interactive (React)    |
| ----------------- | -------------------- | ------------------------ | ---------------------- |
| **Positions**     | i.i.d. each step     | Correlated (movement)    | Correlated (movement)  |
| **Target**        | Fixed                | Mobile                   | Mobile                 |
| **P_det formula** | Exactly validated    | Not applicable           | Not applicable         |
| **Purpose**       | Prove theory correct | Show phenomenon persists | Visualize & explore    |
| **Seed control**  | Per-run seed         | Per-run seed             | Shared seed (Nash=Opt) |
| **In paper**      | Section V.A          | Section V.B              | Demo / Appendix        |

---

## Code Organization

```
project/
├── src/                              # Python modules
│   ├── game.py                       # Analytical formulas (k*, k_opt, PoA)
│   ├── aoi.py                        # AoI computations
│   └── ...
│
├── emergency_simulation.py           # Python simulations
│   ├── simulate_static()             # VALIDATION: i.i.d. positions
│   ├── DynamicSimulation             # DEMONSTRATION: agent movement
│   ├── validate_P_det_static()       # Run validation suite
│   └── run_informative_sweep()       # Run qualitative study
│
├── demos/emergency/ui/               # React interactive demo
│   └── src/
│       ├── emergency_visualization.jsx   # Main component
│       │   ├── SimulationEngine          # Dynamic simulation logic
│       │   ├── createSimulationPair()    # Shared seed initialization
│       │   ├── estimateCoverage()        # Monte Carlo coverage
│       │   └── UI components             # Visualization
│       └── App.jsx
│
└── docs/
    ├── formal_model.md               # Theoretical model (Phase 1-2)
    ├── simulation_pipeline.md        # Development pipeline
    └── VALIDATION_VS_DEMONSTRATION.md # This file
```

---

## Conclusion

The simulation framework has three complementary components:

1. **Static (Python)**: Proves the theory is mathematically correct
2. **Dynamic (Python)**: Shows the theory's predictions have real-world relevance
3. **Interactive (React)**: Enables exploration and builds intuition

Neither invalidates the other. Together, they provide **rigor** (validation), **relevance** (demonstration), and **accessibility** (visualization).
