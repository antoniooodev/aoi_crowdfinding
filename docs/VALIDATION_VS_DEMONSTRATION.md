# Simulation Framework: Validation vs Demonstration

## Overview

This simulation framework serves **two distinct purposes** that must not be conflated:

| Component | Purpose | Validates Theory? |
|-----------|---------|-------------------|
| **Static Simulation** | Validate analytical formulas | ✅ Yes, exactly |
| **Dynamic Simulation** | Demonstrate qualitative phenomenon | ❌ No, intentionally different |

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
| k | Theory | Simulation | Error |
|---|--------|------------|-------|
| 5 | 0.0553 | 0.0555±0.0006 | 0.4% ✓ |
| 50 | 0.4337 | 0.4328±0.0011 | 0.2% ✓ |
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
3. **Variable effective $\rho$**: Depends on volunteer distribution, not uniform

### What It Does Show

The dynamic simulation demonstrates that:

1. **Under-participation gap persists**: $k^* < k^{\text{opt}}$ leads to worse outcomes
2. **Rescue success degrades with cost**: As $c \to B\rho$, Nash success rate drops
3. **AoI matters causally**: Rescue requires sustained detection, not just one-time

### Comparison of P_det

| k | Theory (i.i.d.) | Dynamic Sim | Difference |
|---|-----------------|-------------|------------|
| 50 | 0.434 | ~0.53 | +22% (exploration helps) |

The dynamic simulation gives **higher** P_det because movement helps coverage. This is expected and does not invalidate the theory - it shows the theory is **conservative**.

---

## Summary Table

| Aspect | Static (Validation) | Dynamic (Demonstration) |
|--------|---------------------|-------------------------|
| **Positions** | i.i.d. each step | Correlated (movement) |
| **Target** | Fixed | Mobile |
| **P_det formula** | Exactly validated | Not applicable |
| **Purpose** | Prove theory correct | Show phenomenon persists |
| **In paper** | Section V.A | Section V.B (qualitative) |

---

## Recommended Paper Structure

### Section V: Numerical Results

#### V.A Validation of Analytical Model
> We validate the detection probability formula using Monte Carlo simulation with **i.i.d. volunteer positions** at each time step, matching the theoretical assumptions. Results confirm $P_{\text{det}}(k) = 1-(1-\rho)^k$ within 1% relative error for all tested values of $k$.

#### V.B Dynamic Simulation Study
> To demonstrate the practical impact of under-participation, we extend the model to an **agent-based simulation** with mobile volunteers and target. While this setting does not satisfy the i.i.d. assumption (and thus does not validate the exact formula), it shows that the **qualitative phenomenon persists**: Nash equilibrium participation leads to significantly lower rescue success rates compared to the social optimum, with efficiency losses up to 70% as cost approaches the critical threshold $B\rho$.

#### V.C Key Findings
> 1. The analytical model is validated in the static setting
> 2. Under-participation causes measurable efficiency loss in realistic scenarios
> 3. Stackelberg incentives can recover optimal performance

---

## Code Organization

```
aoi_crowdfinding/
├── src/
│   ├── game.py              # Analytical formulas (k*, k_opt, PoA)
│   ├── aoi.py               # AoI computations
│   └── ...
├── emergency_simulation_v3.py
│   ├── simulate_static()    # VALIDATION: i.i.d. positions
│   ├── DynamicSimulation    # DEMONSTRATION: agent movement
│   ├── validate_P_det_static()  # Run validation suite
│   └── run_informative_sweep()  # Run qualitative study
└── docs/
    └── VALIDATION_VS_DEMONSTRATION.md  # This file
```

---

## Conclusion

The two simulation modes serve complementary purposes:

1. **Static**: Proves the theory is mathematically correct
2. **Dynamic**: Shows the theory's predictions have real-world relevance

Neither invalidates the other. Together, they provide both **rigor** (validation) and **intuition** (demonstration).
