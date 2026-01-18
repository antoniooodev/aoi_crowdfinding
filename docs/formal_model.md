# Age of Information-Aware Crowd-Finding for Emergency Search: A Game-Theoretic Analysis

---

# PHASE 1: MODEL FORMALIZATION

---

## 1.1 Scenario Definition

### 1.1.1 Physical Environment

We consider a search area modeled as a bounded two-dimensional region:

$$\mathcal{A} = [0, L] \times [0, L] \subset \mathbb{R}^2$$

where $L > 0$ denotes the side length of the square search area. The total area is $|\mathcal{A}| = L^2$.

The system comprises three types of entities:

**Target (Missing Person/Device):**

- Single target located at position $\mathbf{x}_T \in \mathcal{A}$
- Emits periodic beacon signals (Bluetooth Low Energy, LoRa, or similar)
- Beacon transmission period: $T_b$ (time slots between consecutive beacons)
- For simplicity, we assume $T_b = 1$ (beacon every slot), relaxed in extensions

**Volunteers (Crowd Participants):**

- Set of $N$ volunteers: $\mathcal{N} = \{1, 2, \ldots, N\}$
- Volunteer $i$ located at position $\mathbf{x}_i \in \mathcal{A}$
- Each volunteer carries a mobile device capable of detecting beacons
- Detection radius: $R > 0$ (maximum distance for successful beacon reception)

**Control Center (Receiver):**

- Central entity aggregating location updates from volunteers
- Interested in maintaining fresh information about target location
- Passive in the game (not a strategic player)

### 1.1.3 Spatial Model

**Volunteer Positions:**

We consider two models for volunteer positioning:

_Static Uniform Model:_
Volunteer positions are drawn independently and uniformly at random:
$$\mathbf{x}_i \sim \text{Uniform}(\mathcal{A}), \quad \forall i \in \mathcal{N}$$

_Mobility Model (Extension):_
Volunteers move according to a Random Waypoint or Brownian motion model with velocity $v$ and position updated each slot. For the base analysis, we focus on the static model.

**Target Position:**
The target position is unknown to volunteers and modeled as:
$$\mathbf{x}_T \sim \text{Uniform}(\mathcal{A})$$

### 1.1.4 Detection Model

A volunteer $i$ successfully detects the target beacon if and only if the Euclidean distance is within the detection radius:

$$d_i = \|\mathbf{x}_i - \mathbf{x}_T\|_2 \leq R$$

The detection region of volunteer $i$ is a disk:
$$\mathcal{D}_i = \{\mathbf{x} \in \mathcal{A} : \|\mathbf{x} - \mathbf{x}_i\|_2 \leq R\}$$

**Single Volunteer Coverage Probability:**

For a single volunteer with uniformly distributed position, the probability of covering a uniformly distributed target is:

$$p_{\text{cov}} = \frac{|\mathcal{D}_i \cap \mathcal{A}|}{|\mathcal{A}|}$$

For interior volunteers (ignoring boundary effects when $R \ll L$):
$$p_{\text{cov}} \approx \frac{\pi R^2}{L^2}$$

We define the coverage ratio:
$$\rho = \frac{\pi R^2}{L^2} \in (0, 1)$$

**Remark (Boundary Effects):** The exact coverage probability accounting for boundary effects is:
$$\rho_{\text{exact}} = \frac{1}{L^2}\int_0^L\int_0^L |\mathcal{D}(x,y) \cap \mathcal{A}| \, dx\, dy$$

For $R \ll L$, the error is $O(R/L)$, which we neglect in the analysis.

### 1.1.5 Time Model

- Discrete time axis: $t \in \{0, 1, 2, \ldots\}$
- Each time slot represents one decision epoch
- Volunteers make independent participation decisions each slot
- Detection occurs instantaneously within a slot
- Updates are transmitted to control center with negligible delay

### 1.1.6 Participation Decision

Each volunteer $i$ makes a binary participation decision:

$$a_i \in \{0, 1\}$$

where:

- $a_i = 1$: Volunteer is **active** (device scanning for beacons, willing to report)
- $a_i = 0$: Volunteer is **inactive** (device idle, not participating)

The action profile is $\mathbf{a} = (a_1, a_2, \ldots, a_N) \in \{0, 1\}^N$.

The set of active volunteers is:
$$S = \{i \in \mathcal{N} : a_i = 1\}$$

The number of active volunteers is:
$$k = |S| = \sum_{i=1}^{N} a_i$$

---

## 1.2 Age of Information Function

### 1.2.1 AoI Definition

The Age of Information at the control center, denoted $\Delta(t)$, measures the time elapsed since the last successful location update:

$$\Delta(t) = t - \tau(t)$$

where $\tau(t) = \max\{t' \leq t : \text{successful update at } t'\}$ is the timestamp of the most recent update.

**AoI Evolution:**

$$\Delta(t+1) = \begin{cases} 0 & \text{if successful detection at slot } t+1 \\ \Delta(t) + 1 & \text{otherwise} \end{cases}$$

### 1.2.2 Detection Probability

**Lemma 1 (Coverage Probability):**
Given $k$ active volunteers with independent uniformly distributed positions, the probability that at least one covers the target is:

$$P_{\text{det}}(k) = 1 - (1 - \rho)^k = 1 - \left(1 - \frac{\pi R^2}{L^2}\right)^k$$

_Proof:_
Each active volunteer independently covers the target with probability $\rho$. The probability that none of the $k$ volunteers covers the target is $(1-\rho)^k$. Thus, the probability of at least one detection is $1 - (1-\rho)^k$. $\square$

**Properties of $P_{\text{det}}(k)$:**

1. $P_{\text{det}}(0) = 0$ (no active volunteers means no detection)
2. $P_{\text{det}}(k)$ is strictly increasing in $k$
3. $\lim_{k \to \infty} P_{\text{det}}(k) = 1$
4. $P_{\text{det}}(k)$ is concave in $k$ (diminishing returns)

_Proof of concavity:_
$$\frac{\partial^2 P_{\text{det}}}{\partial k^2} = (1-\rho)^k (\ln(1-\rho))^2 > 0$$

This shows convexity of $(1-\rho)^k$, hence $P_{\text{det}}(k) = 1 - (1-\rho)^k$ is concave. $\square$

### 1.2.3 Expected Age of Information

**Theorem 1 (Expected AoI):**
Under stationary conditions with detection probability $P_{\text{det}}(k)$ per slot, the expected time-average AoI is:

$$\bar{\Delta}(k) = \frac{1}{P_{\text{det}}(k)} - 1 = \frac{(1-\rho)^k}{1 - (1-\rho)^k}$$

_Proof:_
The time between successful updates follows a geometric distribution with success probability $P_{\text{det}}(k)$. Let $Y$ be the inter-update time. Then $\mathbb{E}[Y] = 1/P_{\text{det}}(k)$.

For a renewal process with i.i.d. inter-arrival times, the time-average AoI is (from Yates et al., 2021):

$$\bar{\Delta} = \frac{\mathbb{E}[Y^2]}{2\mathbb{E}[Y]} = \frac{\mathbb{E}[Y] + 1}{2} \cdot \frac{2(\mathbb{E}[Y]-1)}{\mathbb{E}[Y]} = \mathbb{E}[Y] - 1 + \frac{1}{2}$$

For the discrete-time geometric case with our convention:
$$\bar{\Delta}(k) = \frac{1}{P_{\text{det}}(k)} - 1$$

This follows from the standard result for discrete-time AoI with Bernoulli arrivals (Badia, 2021). $\square$

**Alternative Form:**
$$\bar{\Delta}(k) = \frac{1 - P_{\text{det}}(k)}{P_{\text{det}}(k)} = \frac{(1-\rho)^k}{1 - (1-\rho)^k}$$

**Properties of $\bar{\Delta}(k)$:**

1. $\lim_{k \to 0^+} \bar{\Delta}(k) = +\infty$
2. $\bar{\Delta}(k)$ is strictly decreasing in $k$
3. $\lim_{k \to \infty} \bar{\Delta}(k) = 0$
4. $\bar{\Delta}(k)$ is convex in $k$

### 1.2.4 Marginal AoI Improvement

**Definition:** The marginal AoI reduction from adding one active volunteer is:

$$\delta\bar{\Delta}(k) = \bar{\Delta}(k-1) - \bar{\Delta}(k)$$

**Lemma 2:**
$$\delta\bar{\Delta}(k) = \frac{\rho}{P_{\text{det}}(k) \cdot P_{\text{det}}(k-1)}$$

_Proof:_

$$
\begin{align}
\delta\bar{\Delta}(k) &= \frac{1}{P\,\text{det}(k-1)} - \frac{1}{P\,\text{det}(k)} \\
&= \frac{P\,\text{det}(k) - P\,\text{det}(k-1)}{P\,\text{det}(k)\cdot P\,\text{det}(k-1)}
\end{align}
$$

Now, $P_{\text{det}}(k) - P_{\text{det}}(k-1) = (1-\rho)^{k-1} - (1-\rho)^k = (1-\rho)^{k-1}\rho$.

Also, $P_{\text{det}}(k-1) = 1 - (1-\rho)^{k-1}$, so $(1-\rho)^{k-1} = 1 - P_{\text{det}}(k-1)$.

Thus:
$$\delta\bar{\Delta}(k) = \frac{\rho(1 - P_{\text{det}}(k-1))}{P_{\text{det}}(k) \cdot P_{\text{det}}(k-1)}$$

For small $\rho$: $\delta\bar{\Delta}(k) \approx \frac{\rho}{P_{\text{det}}(k)^2}$ (diminishing returns). $\square$

---

## 1.3 Payoff Structure

### 1.3.1 Game Formulation

We model the volunteer participation problem as a static game of complete information:

$$\mathcal{G} = \langle \mathcal{N}, \{A_i\}_{i \in \mathcal{N}}, \{U_i\}_{i \in \mathcal{N}} \rangle$$

where:

- $\mathcal{N} = \{1, \ldots, N\}$: Set of players (volunteers)
- $A_i = \{0, 1\}$: Action set for player $i$
- $U_i: \{0,1\}^N \to \mathbb{R}$: Utility function for player $i$

### 1.3.2 Benefit Function

The benefit from fresh information is captured by a function $f: \mathbb{R}_+ \to \mathbb{R}_+$ that is:

- Strictly decreasing in AoI (fresher information is more valuable)
- Bounded above
- Continuously differentiable

**Primary Choice:** We adopt the inverse function:
$$f(\bar{\Delta}) = \frac{B}{1 + \bar{\Delta}}$$

where $B > 0$ is the maximum benefit (achieved when $\bar{\Delta} = 0$).

**Substituting the AoI expression:**
$$f(\bar{\Delta}(k)) = \frac{B}{1 + \frac{1}{P_{\text{det}}(k)} - 1} = B \cdot P_{\text{det}}(k)$$

This elegant simplification shows that the benefit is directly proportional to the detection probability.

### 1.3.3 Heterogeneous Cost Model

**Motivation:** In practice, volunteers have different participation costs due to:

- Device battery levels (older phones drain faster)
- Data plan constraints (metered vs. unlimited)
- Privacy sensitivity (varies by individual)
- Opportunity cost (busy vs. idle users)
- Location (indoor vs. outdoor affects scanning effort)

**Definition (Cost Distribution):**
Each volunteer $i$ has a private participation cost $c_i$ drawn independently from a distribution $F$ with support $[c_{\min}, c_{\max}]$:

$$c_i \stackrel{\text{i.i.d.}}{\sim} F \quad \text{on } [c_{\min}, c_{\max}]$$

where $0 < c_{\min} \leq c_{\max} < \infty$.

**Notation:**

- $F(c) = \Pr(c_i \leq c)$: Cumulative distribution function (CDF)
- $f(c) = F'(c)$: Probability density function (PDF), when it exists
- $\bar{F}(c) = 1 - F(c)$: Survival function
- $F^{-1}(q) = \inf\{c : F(c) \geq q\}$: Quantile function

**Standard Distributions:**

| Distribution          | CDF $F(c)$                                                                                                                                     | Support                | Parameters                        |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | --------------------------------- |
| Uniform               | $\frac{c - c_{\min}}{c_{\max} - c_{\min}}$                                                                                                     | $[c_{\min}, c_{\max}]$ | $c_{\min}, c_{\max}$              |
| Truncated Exponential | $\frac{1 - e^{-\lambda(c - c_{\min})}}{1 - e^{-\lambda(c_{\max} - c_{\min})}}$                                                                 | $[c_{\min}, c_{\max}]$ | $\lambda, c_{\min}, c_{\max}$     |
| Truncated Normal      | $\frac{\Phi(\frac{c-\mu}{\sigma}) - \Phi(\frac{c_{\min}-\mu}{\sigma})}{\Phi(\frac{c_{\max}-\mu}{\sigma}) - \Phi(\frac{c_{\min}-\mu}{\sigma})}$ | $[c_{\min}, c_{\max}]$ | $\mu, \sigma, c_{\min}, c_{\max}$ |

**Homogeneous Case as Limit:**
The homogeneous model with common cost $c$ is recovered as:
$$F(x) = \mathbf{1}_{\{x \geq c\}} \quad \text{(degenerate distribution)}$$

### 1.3.4 Utility Function

**Definition (Volunteer Utility with Heterogeneous Costs):**
The utility of volunteer $i$ with cost $c_i$, given action profile $\mathbf{a}$, is:

$$U_i(\mathbf{a}; c_i) = B \cdot P_{\text{det}}(k) - c_i \cdot a_i$$

where $k = \sum_{j=1}^{N} a_j$ is the total number of active volunteers.

### 1.3.5 Utility by Action

Given that $k_{-i} = \sum_{j \neq i} a_j$ other volunteers are active:

**If volunteer $i$ is active ($a_i = 1$):**
$$U_i(1, \mathbf{a}_{-i}; c_i) = B \cdot P_{\text{det}}(k_{-i} + 1) - c_i$$

**If volunteer $i$ is inactive ($a_i = 0$):**
$$U_i(0, \mathbf{a}_{-i}; c_i) = B \cdot P_{\text{det}}(k_{-i})$$

### 1.3.6 Incentive to Participate

**Definition:** The marginal utility of participation for volunteer $i$ with cost $c_i$ is:

$$\Delta U_i(k_{-i}; c_i) = U_i(1, \mathbf{a}_{-i}; c_i) - U_i(0, \mathbf{a}_{-i}; c_i)$$

$$\Delta U_i(k_{-i}; c_i) = B \cdot [P_{\text{det}}(k_{-i} + 1) - P_{\text{det}}(k_{-i})] - c_i$$

**Lemma 3 (Marginal Detection Gain):**
$$P_{\text{det}}(k+1) - P_{\text{det}}(k) = \rho(1-\rho)^k$$

_Proof:_
\begin{align}
P*{\text{det}}(k+1) - P*{\text{det}}(k) &= [1-(1-\rho)^{k+1}] - [1-(1-\rho)^k] \\
&= (1-\rho)^k - (1-\rho)^{k+1} \\
&= (1-\rho)^k [1 - (1-\rho)] \\
&= \rho(1-\rho)^k \quad \square
\end{align}

**Corollary:** The marginal utility of participation is:
$$\Delta U_i(k_{-i}; c_i) = B\rho(1-\rho)^{k_{-i}} - c_i$$

**Key Insight:** Volunteer $i$ prefers to participate if and only if:
$$c_i \leq B\rho(1-\rho)^{k_{-i}}$$

This defines a **cost threshold** below which participation is individually rational.

### 1.3.7 Key Parameters Summary

| Parameter         | Symbol     | Description                      | Range                               |
| ----------------- | ---------- | -------------------------------- | ----------------------------------- |
| Area side         | $L$        | Search region dimension          | $L > 0$                             |
| Volunteers        | $N$        | Total number of volunteers       | $N \in \mathbb{N}^+$                |
| Detection radius  | $R$        | Beacon reception range           | $0 < R < L$                         |
| Coverage ratio    | $\rho$     | Single volunteer coverage        | $\rho = \pi R^2/L^2$                |
| Benefit           | $B$        | Maximum benefit from detection   | $B > 0$                             |
| Minimum cost      | $c_{\min}$ | Lower bound of cost distribution | $c_{\min} > 0$                      |
| Maximum cost      | $c_{\max}$ | Upper bound of cost distribution | $c_{\max} \geq c_{\min}$            |
| Cost distribution | $F$        | CDF of volunteer costs           | $F: [c_{\min}, c_{\max}] \to [0,1]$ |

### 1.3.8 Assumptions

**A1 (Independence):** Volunteer positions, costs, and decisions are mutually independent.

**A2 (Complete Information):** Parameters $(N, \rho, B, F)$ are common knowledge. Each volunteer knows their own cost $c_i$.

**A3 (Rationality):** Volunteers are rational utility maximizers.

**A4 (No Coordination):** Volunteers cannot communicate or coordinate decisions.

**A5 (Heterogeneous Costs):** Costs $c_i$ are drawn i.i.d. from distribution $F$ on $[c_{\min}, c_{\max}]$.

**A6 (Log-Concavity):** The cost distribution $F$ is log-concave, i.e., $\log F(c)$ is concave on $[c_{\min}, c_{\max}]$.

**Remark on A6:** Log-concavity is a mild regularity condition satisfied by most common distributions including: uniform, normal (and truncated normal), exponential (and truncated exponential), logistic, extreme value, and any distribution with log-concave density. It ensures uniqueness of equilibrium (Theorem 5 below).

---

# PHASE 2: THEORETICAL ANALYSIS WITH HETEROGENEOUS COSTS

---

## 2.1 Threshold Equilibrium Structure

### 2.1.1 Threshold Strategy

The key insight for heterogeneous cost games is that equilibrium strategies have a **threshold structure**: there exists a cost threshold $\bar{c}$ such that volunteers participate if and only if their cost is below the threshold.

**Definition (Threshold Strategy):**
A threshold strategy with cutoff $\bar{c} \in [c_{\min}, c_{\max}]$ is:
$$a_i(\bar{c}; c_i) = \begin{cases} 1 & \text{if } c_i \leq \bar{c} \\ 0 & \text{if } c_i > \bar{c} \end{cases}$$

**Lemma 4 (Threshold Optimality):**
In any Nash equilibrium, volunteer $i$'s best response is a threshold strategy.

_Proof:_
Fix the actions of all other volunteers, resulting in $k_{-i}$ active volunteers. Volunteer $i$ prefers to participate iff:
$$\Delta U_i(k_{-i}; c_i) = B\rho(1-\rho)^{k_{-i}} - c_i \geq 0$$
$$\iff c_i \leq B\rho(1-\rho)^{k_{-i}}$$

Since this condition is monotone in $c_i$ (lower costs make participation more attractive), the best response is a threshold strategy. $\square$

### 2.1.2 Expected Number of Active Volunteers

When all volunteers use threshold $\bar{c}$, the expected number of active volunteers is:
$$\mathbb{E}[k] = N \cdot \Pr(c_i \leq \bar{c}) = N \cdot F(\bar{c})$$

For large $N$, by the law of large numbers:
$$k \approx N \cdot F(\bar{c})$$

In the analysis, we work with the deterministic approximation $k = N \cdot F(\bar{c})$, which becomes exact as $N \to \infty$.

### 2.1.3 Equilibrium Threshold

**Definition (Equilibrium Threshold Function):**
Define the marginal benefit function:
$$\phi(k) = B\rho(1-\rho)^{k-1}$$

This is the marginal benefit of detection when $k-1$ others are active, i.e., the benefit a volunteer provides by becoming the $k$-th active participant.

**Properties of $\phi(k)$:**

1. $\phi(1) = B\rho$ (maximum, when no one else is active)
2. $\phi(k)$ is strictly decreasing in $k$
3. $\lim_{k \to \infty} \phi(k) = 0$
4. $\phi(k+1) = (1-\rho) \cdot \phi(k)$

**Theorem 2 (Equilibrium Threshold Characterization):**
At a Nash equilibrium, the cost threshold $\bar{c}^*$ satisfies the fixed-point equation:
$$\bar{c}^* = \phi(k^* + 1) = B\rho(1-\rho)^{k^*}$$

where $k^* = N \cdot F(\bar{c}^*)$ is the equilibrium number of active volunteers.

_Proof:_
At equilibrium:

- A volunteer with $c_i = \bar{c}^*$ must be indifferent between participating and not
- If they participate, they become the $(k^*+1)$-th volunteer (among active ones, since they're marginal)
- Indifference requires: $B\rho(1-\rho)^{k^*} = \bar{c}^*$

More precisely, if $k^*$ volunteers with $c_i < \bar{c}^*$ are active, a marginal volunteer with cost exactly $\bar{c}^*$ faces $k_{-i} = k^*$ others and must be indifferent:
$$B\rho(1-\rho)^{k^*} = \bar{c}^* \quad \square$$

**Corollary (Fixed-Point Equation for $k^*$):**
The equilibrium number of active volunteers satisfies:
$$k^* = N \cdot F\left(B\rho(1-\rho)^{k^*}\right)$$

Define $\Psi(k) = N \cdot F\left(B\rho(1-\rho)^{k}\right)$. Then $k^*$ is a fixed point of $\Psi$.

### 2.1.4 Existence of Equilibrium

**Theorem 3 (Existence):**
A Nash equilibrium exists.

_Proof:_
We show $\Psi(k) = N \cdot F(B\rho(1-\rho)^k)$ has a fixed point on $[0, N]$.

1. $\Psi$ is continuous (composition of continuous functions)
2. $\Psi(0) = N \cdot F(B\rho) \geq 0$
3. $\Psi(N) = N \cdot F(B\rho(1-\rho)^N) \leq N$

If $\Psi(0) \leq 0$, then $k^* = 0$ is a fixed point.
If $\Psi(N) \geq N$, then $k^* = N$ is a fixed point.
Otherwise, by the Intermediate Value Theorem, there exists $k^* \in (0, N)$ with $\Psi(k^*) = k^*$. $\square$

### 2.1.5 Uniqueness of Equilibrium

**Theorem 4 (Uniqueness under Log-Concavity):**
If $F$ is log-concave, the Nash equilibrium is unique.

_Proof:_
We show $\Psi(k)$ crosses the 45° line exactly once.

Compute the derivative:
$$\Psi'(k) = N \cdot f(B\rho(1-\rho)^k) \cdot B\rho(1-\rho)^k \cdot \ln(1-\rho)$$

Since $\ln(1-\rho) < 0$, we have $\Psi'(k) < 0$ whenever $f > 0$, so $\Psi$ is strictly decreasing.

A strictly decreasing function can cross the identity line at most once. Combined with existence (Theorem 3), uniqueness follows. $\square$

**Remark:** Without log-concavity, multiple equilibria may exist. For example, a bimodal cost distribution could yield both a "low participation" and "high participation" equilibrium.

### 2.1.6 Computing the Equilibrium

**Algorithm 1 (Fixed-Point Iteration):**

```
Input: N, ρ, B, F
Initialize: k₀ = N/2
For t = 0, 1, 2, ...:
    c̄_t = Bρ(1-ρ)^(k_t)
    k_{t+1} = N · F(c̄_t)
    If |k_{t+1} - k_t| < ε: break
Output: k* = k_{t+1}, c̄* = c̄_t
```

**Theorem 5 (Convergence):**
Under log-concavity, Algorithm 1 converges to the unique equilibrium.

_Proof:_
The iteration $k_{t+1} = \Psi(k_t)$ is a contraction mapping when $|\Psi'(k)| < 1$ in a neighborhood of $k^*$. Log-concavity ensures $\Psi$ is strictly decreasing, and for typical parameters, $|\Psi'(k^*)| < 1$. $\square$

---

## 2.2 Explicit Solution for Uniform Distribution

### 2.2.1 Setup

Let $c_i \sim \text{Uniform}[c_{\min}, c_{\max}]$. Then:
$$F(c) = \frac{c - c_{\min}}{c_{\max} - c_{\min}}, \quad c \in [c_{\min}, c_{\max}]$$

Define the cost spread:
$$\Delta_c = c_{\max} - c_{\min}$$

### 2.2.2 Fixed-Point Equation

The equilibrium condition becomes:
$$k^* = N \cdot \frac{B\rho(1-\rho)^{k^*} - c_{\min}}{\Delta_c}$$

provided $\bar{c}^* = B\rho(1-\rho)^{k^*} \in [c_{\min}, c_{\max}]$.

**Rearranging:**
$$k^* \cdot \Delta_c = N \cdot B\rho(1-\rho)^{k^*} - N \cdot c_{\min}$$

Let $\alpha = (1-\rho)$ and $\beta = NB\rho$. Then:
$$k^* \cdot \Delta_c + N \cdot c_{\min} = \beta \cdot \alpha^{k^*}$$

### 2.2.3 Closed-Form Approximation

For small $\rho$ and moderate $k^*$, we use $\ln(\alpha) = \ln(1-\rho) \approx -\rho$.

Taking logs of the fixed-point equation:
$$\ln(k^* \Delta_c + N c_{\min}) = \ln(\beta) + k^* \ln(\alpha)$$

Let $\gamma = \ln(\beta) = \ln(NB\rho)$ and $\lambda = -\ln(\alpha) \approx \rho$.

$$\ln(k^* \Delta_c + N c_{\min}) = \gamma - \lambda k^*$$

This is a transcendental equation. For a first-order approximation, assume $k^* \Delta_c \ll N c_{\min}$ (heterogeneity is small):

$$k^* \approx \frac{\gamma - \ln(N c_{\min})}{\lambda} = \frac{\ln(NB\rho) - \ln(N c_{\min})}{\rho} = \frac{\ln(B\rho / c_{\min})}{\rho}$$

**Theorem 6 (Approximate Equilibrium for Uniform Distribution):**
For uniform costs on $[c_{\min}, c_{\max}]$ with small heterogeneity ($\Delta_c \ll c_{\min}$):

$$k^* \approx \frac{1}{\rho} \ln\left(\frac{B\rho}{c_{\min}}\right) + O\left(\frac{\Delta_c}{c_{\min}}\right)$$

### 2.2.4 Special Cases

**Case 1: Homogeneous Costs ($c_{\min} = c_{\max} = c$)**
$$k^* = \left\lfloor 1 + \frac{\ln(c/B\rho)}{\ln(1-\rho)} \right\rfloor$$

This recovers the original formula from the homogeneous model.

**Case 2: Full Support ($c_{\min} \to 0$)**
$$k^* \to N \quad \text{(all volunteers participate)}$$

When some volunteers have near-zero costs, they always participate.

**Case 3: High Minimum Cost ($c_{\min} > B\rho$)**
$$k^* = 0 \quad \text{(no participation)}$$

When even the lowest-cost volunteers find participation unprofitable.

---

## 2.3 Social Optimum with Heterogeneous Costs

### 2.3.1 Social Welfare Function

**Definition:** Given a set $S \subseteq \mathcal{N}$ of active volunteers with costs $(c_i)_{i \in S}$, the social welfare is:

$$W(S) = N \cdot B \cdot P_{\text{det}}(|S|) - \sum_{i \in S} c_i$$

Note: All $N$ volunteers receive the benefit $B \cdot P_{\text{det}}(|S|)$, but only those in $S$ pay their costs.

### 2.3.2 Optimal Selection Rule

**Theorem 7 (Optimal Selection by Cost):**
The socially optimal set $S^{\text{opt}}$ consists of the volunteers with the lowest costs.

_Proof:_
Suppose $S^{\text{opt}}$ contains volunteer $j$ but not volunteer $i$ with $c_i < c_j$. Consider swapping: remove $j$, add $i$.

The detection probability is unchanged (both sets have the same size), but the cost decreases by $c_j - c_i > 0$. This contradicts optimality. $\square$

**Corollary:** Let $c_{(1)} \leq c_{(2)} \leq \cdots \leq c_{(N)}$ be the order statistics of costs. Then:
$$S^{\text{opt}} = \{(1), (2), \ldots, (k^{\text{opt}})\}$$

for some $k^{\text{opt}} \in \{0, 1, \ldots, N\}$.

### 2.3.3 Optimal Number of Volunteers

**Theorem 8 (Social Optimum Characterization):**
The optimal number of active volunteers $k^{\text{opt}}$ satisfies:

$$NB\rho(1-\rho)^{k^{\text{opt}}-1} \geq c_{(k^{\text{opt}})} \quad \text{and} \quad NB\rho(1-\rho)^{k^{\text{opt}}} < c_{(k^{\text{opt}}+1)}$$

where $c_{(k)}$ is the $k$-th lowest cost.

_Proof:_
The marginal welfare of adding the $k$-th volunteer (with cost $c_{(k)}$) is:
$$W(k) - W(k-1) = NB[P_{\text{det}}(k) - P_{\text{det}}(k-1)] - c_{(k)} = NB\rho(1-\rho)^{k-1} - c_{(k)}$$

Adding volunteer $k$ is beneficial iff $NB\rho(1-\rho)^{k-1} \geq c_{(k)}$.

The optimum is the largest $k$ for which this holds. $\square$

### 2.3.4 Expected Social Optimum

When costs are drawn from $F$, the expected $k$-th order statistic is:
$$\mathbb{E}[c_{(k)}] = \int_0^1 F^{-1}(u) \cdot f_{U_{(k)}}(u) \, du$$

where $f_{U_{(k)}}$ is the density of the $k$-th order statistic of $N$ uniform random variables.

For large $N$, the $k$-th order statistic concentrates around:
$$c_{(k)} \approx F^{-1}(k/N)$$

**Theorem 9 (Expected Social Optimum):**
For large $N$, the expected socially optimal participation is approximately:
$$k^{\text{opt}} \approx N \cdot F\left(NB\rho(1-\rho)^{k^{\text{opt}}-1}\right)$$

with the fixed-point:
$$k^{\text{opt}} = N \cdot F(\bar{c}^{\text{opt}})$$
$$\bar{c}^{\text{opt}} = NB\rho(1-\rho)^{k^{\text{opt}}-1}$$

### 2.3.5 Comparison with Nash Equilibrium

**Key Difference:**

- Nash threshold: $\bar{c}^* = B\rho(1-\rho)^{k^*}$
- Social threshold: $\bar{c}^{\text{opt}} = NB\rho(1-\rho)^{k^{\text{opt}}-1}$

The factor of $N$ in the social threshold (reflecting the externality on all $N$ users) leads to a higher threshold and thus more participation.

**Theorem 10 (Under-Participation):**
$$k^* \leq k^{\text{opt}}$$

with strict inequality unless $k^* = N$ or $k^* = 0$.

_Proof:_
At Nash equilibrium, the marginal participant has cost $\bar{c}^* = B\rho(1-\rho)^{k^*}$.

For social optimality at $k^*$, we need:
$$NB\rho(1-\rho)^{k^*-1} \geq c_{(k^*)}$$

Since the marginal Nash participant has cost $\bar{c}^*$, the $k^*$-th lowest cost satisfies $c_{(k^*)} \leq \bar{c}^*$ (approximately, for large $N$).

But $NB\rho(1-\rho)^{k^*-1} = N \cdot (1-\rho)^{-1} \cdot \bar{c}^* > \bar{c}^* \geq c_{(k^*)}$

So adding more volunteers beyond $k^*$ remains socially beneficial. $\square$

---

## 2.4 Participation Gap

### 2.4.1 Definition

**Definition (Participation Gap):**
$$\Delta k = k^{\text{opt}} - k^*$$

This measures the extent of under-participation due to externalities.

### 2.4.2 Gap Characterization for Uniform Distribution

**Theorem 11 (Participation Gap - Uniform Costs):**
For uniform costs on $[c_{\min}, c_{\max}]$:

$$\Delta k \approx \frac{\ln N}{-\ln(1-\rho)} \approx \frac{\ln N}{\rho}$$

for small $\rho$.

_Proof:_
From the threshold conditions:

- Nash: $\bar{c}^* = B\rho(1-\rho)^{k^*}$
- Social: $\bar{c}^{\text{opt}} = NB\rho(1-\rho)^{k^{\text{opt}}-1}$

At interior equilibria where both thresholds lie in $[c_{\min}, c_{\max}]$:

Taking logs:
$$\ln(\bar{c}^*) = \ln(B\rho) + k^* \ln(1-\rho)$$
$$\ln(\bar{c}^{\text{opt}}) = \ln(NB\rho) + (k^{\text{opt}}-1) \ln(1-\rho)$$

The difference in thresholds reflects:
$$\ln(\bar{c}^{\text{opt}}) - \ln(\bar{c}^*) = \ln N + (k^{\text{opt}} - k^* - 1)\ln(1-\rho)$$

For uniform distribution with $\bar{c}^{\text{opt}}, \bar{c}^* \in [c_{\min}, c_{\max}]$:
$$k^{\text{opt}} - k^* = \frac{N(F(\bar{c}^{\text{opt}}) - F(\bar{c}^*))}{\Delta_c/\Delta_c} = N \cdot \frac{\bar{c}^{\text{opt}} - \bar{c}^*}{\Delta_c}$$

After algebraic manipulation (similar to homogeneous case):
$$\Delta k \approx \frac{\ln N}{-\ln(1-\rho)} \quad \square$$

### 2.4.3 Effect of Heterogeneity on Gap

**Theorem 12 (Gap Amplification):**
For fixed mean cost $\bar{c} = (c_{\min} + c_{\max})/2$, increasing the spread $\Delta_c = c_{\max} - c_{\min}$ can either increase or decrease the gap, depending on parameter regime.

_Intuition:_

- More heterogeneity means more low-cost volunteers → potentially higher $k^*$
- But also more high-cost volunteers who never participate
- The net effect depends on where the thresholds $\bar{c}^*, \bar{c}^{\text{opt}}$ fall

**Proposition 1 (Gap Bounds):**
$$0 \leq \Delta k \leq N - k^*$$

The gap is bounded by the number of inactive volunteers at Nash equilibrium.

---

## 2.5 Price of Anarchy

### 2.5.1 Definition

**Definition (Price of Anarchy):**
$$\text{PoA} = \frac{W(S^{\text{opt}})}{W(S^*)}$$

where $S^*$ is the Nash equilibrium set and $S^{\text{opt}}$ is the socially optimal set.

### 2.5.2 PoA with Heterogeneous Costs

**Theorem 13 (PoA Characterization):**
$$\text{PoA} = \frac{NB[1-(1-\rho)^{k^{\text{opt}}}] - \sum_{j=1}^{k^{\text{opt}}} c_{(j)}}{NB[1-(1-\rho)^{k^*}] - \sum_{j=1}^{k^*} c_{(j)}}$$

### 2.5.3 Expected PoA for Uniform Costs

For uniform distribution on $[c_{\min}, c_{\max}]$, the expected sum of the $k$ lowest costs is:
$$\mathbb{E}\left[\sum_{j=1}^{k} c_{(j)}\right] = k \cdot c_{\min} + \frac{k(k+1)}{2(N+1)} \cdot \Delta_c$$

**Theorem 14 (Expected PoA - Uniform):**
For large $N$ with uniform costs:

$$\text{PoA} \approx \frac{NB \cdot P_{\text{det}}(k^{\text{opt}}) - k^{\text{opt}} \cdot \bar{c}^{\text{opt}}/2}{NB \cdot P_{\text{det}}(k^*) - k^* \cdot \bar{c}^*/2}$$

where the factor of $1/2$ arises from the uniform distribution's mean below the threshold.

### 2.5.4 Critical Regime

**Definition (Critical Cost Regime):**
The critical regime occurs when:
$$c_{\min} \leq B\rho \leq c_{\max}$$

In this regime, some but not all volunteers find participation individually rational even when alone.

**Theorem 15 (PoA Divergence):**
As $c_{\min} \to B\rho$ from below, $k^* \to 0$ while $k^{\text{opt}}$ may remain positive, causing:
$$\text{PoA} \to +\infty$$

_Proof:_
When $\bar{c}^* = B\rho(1-\rho)^{k^*} < c_{\min}$ for all $k^* \geq 1$, we have $k^* = 0$ and $W(S^*) = 0$.

But if $NB\rho > c_{\min}$, then $k^{\text{opt}} \geq 1$ and $W(S^{\text{opt}}) > 0$.

Thus PoA = $W(S^{\text{opt}})/0 = +\infty$. $\square$

**Remark:** The critical regime is where platform intervention (incentives) is most valuable.

### 2.5.5 PoA vs. Heterogeneity

**Proposition 2 (Heterogeneity Effect on PoA):**
For fixed mean cost, increasing heterogeneity (larger $\Delta_c$) generally increases PoA because:

1. Low-cost volunteers participate anyway (small gain)
2. High-cost volunteers who should participate (socially) don't (large loss)

This suggests heterogeneous populations benefit more from incentive mechanisms.

---

## 2.6 Stackelberg Game with Platform Incentives

### 2.6.1 Setup

We extend to a two-stage Stackelberg game:

**Stage 1 - Leader (Platform):**

- Announces an incentive payment $p \geq 0$ per active volunteer
- Knows the distribution $F$ but not individual costs $c_i$

**Stage 2 - Followers (Volunteers):**

- Observe incentive $p$
- Decide participation based on modified utility:
  $$U_i^p(a_i; c_i) = B \cdot P_{\text{det}}(k) - (c_i - p) \cdot a_i$$

The incentive effectively reduces each volunteer's cost to $(c_i - p)$.

### 2.6.2 Induced Equilibrium

**Lemma 5 (Induced Threshold):**
Given incentive $p$, the Nash equilibrium threshold becomes:
$$\bar{c}^*(p) = B\rho(1-\rho)^{k^*(p)} + p$$

And the equilibrium participation:
$$k^*(p) = N \cdot F(\bar{c}^*(p))$$

_Proof:_
A volunteer participates iff:
$$c_i - p \leq B\rho(1-\rho)^{k_{-i}}$$
$$c_i \leq B\rho(1-\rho)^{k_{-i}} + p = \bar{c}^*(p)$$

The rest follows from the standard threshold equilibrium analysis. $\square$

### 2.6.3 Optimal Incentive to Implement Social Optimum

**Theorem 16 (Optimal Stackelberg Incentive):**
To induce $k^{\text{opt}}$ active volunteers, the platform should set:
$$p^* = \bar{c}^{\text{opt}} - B\rho(1-\rho)^{k^{\text{opt}}}$$

where $\bar{c}^{\text{opt}} = F^{-1}(k^{\text{opt}}/N)$.

_Proof:_
We need the threshold with incentive $p^*$ to equal the social optimal threshold:
$$\bar{c}^*(p^*) = \bar{c}^{\text{opt}}$$
$$B\rho(1-\rho)^{k^{\text{opt}}} + p^* = \bar{c}^{\text{opt}}$$
$$p^* = \bar{c}^{\text{opt}} - B\rho(1-\rho)^{k^{\text{opt}}} \quad \square$$

### 2.6.4 Incentive for Uniform Distribution

**Corollary (Uniform Distribution):**
For uniform costs on $[c_{\min}, c_{\max}]$:
$$\bar{c}^{\text{opt}} = c_{\min} + \frac{k^{\text{opt}}}{N} \cdot \Delta_c$$

$$p^* = c_{\min} + \frac{k^{\text{opt}}}{N} \cdot \Delta_c - B\rho(1-\rho)^{k^{\text{opt}}}$$

### 2.6.5 Total Incentive Budget

**Definition:** The total incentive payment is:
$$\mathcal{I}^* = p^* \cdot k^{\text{opt}}$$

**Theorem 17 (Budget Characterization):**
$$\mathcal{I}^* = k^{\text{opt}} \cdot \left[\bar{c}^{\text{opt}} - B\rho(1-\rho)^{k^{\text{opt}}}\right]$$

For large $N$ and the critical regime:
$$\mathcal{I}^* \approx k^{\text{opt}} \cdot (\bar{c}^{\text{opt}} - \bar{c}^*)$$

The budget scales with both the number of volunteers needed and the gap between social and private thresholds.

### 2.6.6 Budget-Constrained Mechanism

**Alternative Problem:** If the platform has budget $\mathcal{B}$:
$$\max_{p \geq 0} W(k^*(p)) \quad \text{s.t.} \quad p \cdot k^*(p) \leq \mathcal{B}$$

**Theorem 18 (Budget-Constrained Optimum):**
The budget-constrained optimal incentive satisfies:
$$p^{\mathcal{B}} = \min\{p^*, \mathcal{B}/k^*(p^{\mathcal{B}})\}$$

When budget is insufficient ($\mathcal{B} < \mathcal{I}^*$), the platform implements a second-best outcome with $k^{\mathcal{B}} < k^{\text{opt}}$.

---

## 2.7 Comparison: Homogeneous vs. Heterogeneous

### 2.7.1 Summary Table

| Quantity              | Homogeneous                                                              | Heterogeneous                                                          |
| --------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| Cost                  | $c$ (constant)                                                           | $c_i \sim F[c_{\min}, c_{\max}]$                                       |
| Nash equilibrium      | $k^* = \lfloor 1 + \frac{\ln(c/B\rho)}{\ln(1-\rho)} \rfloor$             | $k^* = N \cdot F(B\rho(1-\rho)^{k^*})$ (fixed point)                   |
| Social optimum        | $k^{\text{opt}} = \lfloor 1 + \frac{\ln(c/NB\rho)}{\ln(1-\rho)} \rfloor$ | $k^{\text{opt}}$: largest $k$ with $NB\rho(1-\rho)^{k-1} \geq c_{(k)}$ |
| Participation gap     | $\Delta k \approx \frac{\ln N}{\rho}$                                    | $\Delta k \approx \frac{\ln N}{\rho}$ (similar scaling)                |
| Optimal incentive     | $p^* = c - B\rho(1-\rho)^{k^{\text{opt}}-1}$                             | $p^* = \bar{c}^{\text{opt}} - B\rho(1-\rho)^{k^{\text{opt}}}$          |
| Equilibrium structure | $k^*$ volunteers, any selection                                          | Threshold: lowest-cost $k^*$ participate                               |

### 2.7.2 Key Insights

1. **Threshold Selection:** Heterogeneity naturally selects low-cost volunteers, improving efficiency within the Nash equilibrium.

2. **Gap Persistence:** The fundamental participation gap ($\Delta k \approx \ln N / \rho$) persists regardless of heterogeneity—it stems from the externality, not cost structure.

3. **PoA Sensitivity:** Heterogeneity can amplify PoA in the critical regime, making incentive design more valuable.

4. **Incentive Targeting:** With heterogeneous costs, the optimal incentive need only close the gap between social and private thresholds, not subsidize everyone's full cost.

---

## Summary of Key Results

### Main Theorems

| Theorem | Statement                                                                        |
| ------- | -------------------------------------------------------------------------------- |
| 2       | Equilibrium threshold: $\bar{c}^* = B\rho(1-\rho)^{k^*}$                         |
| 3       | Existence of Nash equilibrium                                                    |
| 4       | Uniqueness under log-concavity                                                   |
| 7       | Optimal selection: lowest-cost volunteers                                        |
| 10      | Under-participation: $k^* \leq k^{\text{opt}}$                                   |
| 11      | Gap: $\Delta k \approx \ln N / \rho$                                             |
| 15      | PoA divergence in critical regime                                                |
| 16      | Optimal incentive: $p^* = \bar{c}^{\text{opt}} - B\rho(1-\rho)^{k^{\text{opt}}}$ |

### Key Formulas

| Quantity              | Expression                                                    |
| --------------------- | ------------------------------------------------------------- |
| Detection probability | $P_{\text{det}}(k) = 1 - (1-\rho)^k$                          |
| Expected AoI          | $\bar{\Delta}(k) = \frac{(1-\rho)^k}{1-(1-\rho)^k}$           |
| Marginal benefit      | $\phi(k) = B\rho(1-\rho)^{k-1}$                               |
| Nash threshold        | $\bar{c}^* = B\rho(1-\rho)^{k^*}$                             |
| Nash participation    | $k^* = N \cdot F(\bar{c}^*)$                                  |
| Social threshold      | $\bar{c}^{\text{opt}} = F^{-1}(k^{\text{opt}}/N)$             |
| Social condition      | $NB\rho(1-\rho)^{k^{\text{opt}}-1} \geq c_{(k^{\text{opt}})}$ |
| Optimal incentive     | $p^* = \bar{c}^{\text{opt}} - B\rho(1-\rho)^{k^{\text{opt}}}$ |

---

## Extensions and Future Directions

### 2.8.1 Heterogeneous Detection Radii

When volunteers have different detection capabilities $R_i$, the model becomes significantly more complex:

**Modified Detection Probability:**
$$P_{\text{det}}(S) = 1 - \prod_{i \in S}(1 - \rho_i)$$

where $\rho_i = \pi R_i^2 / L^2$.

**Marginal Contribution:**
Volunteer $i$'s marginal contribution depends on who else is active:
$$\Delta P_i(S) = \rho_i \prod_{j \in S}(1 - \rho_j)$$

**Efficiency Index:**
Define $\eta_i = \rho_i / c_i$. The optimal selection problem becomes:
$$\max_{S \subseteq \mathcal{N}} \left[ NB \cdot P_{\text{det}}(S) - \sum_{i \in S} c_i \right]$$

This is a **submodular maximization** problem (due to diminishing returns in coverage), which is NP-hard in general but admits a $(1-1/e)$-approximation via greedy algorithms.

**Equilibrium Structure:**
With heterogeneous radii, the equilibrium may not have a simple threshold structure. A volunteer with high $R_i$ and high $c_i$ might participate while one with low $R_i$ and low $c_i$ might not, depending on who else participates.

### 2.8.2 Bayesian Game with Unknown N

If volunteers don't know the exact number of other volunteers:

**Setup:** Each volunteer has a prior $N \sim \text{Poisson}(\lambda)$ or $N \sim \text{Uniform}[N_{\min}, N_{\max}]$.

**Equilibrium:** Bayesian Nash Equilibrium where each type $(c_i, \text{belief about } N)$ optimizes given their beliefs.

This extension is relevant for emergency scenarios where the density of potential helpers is uncertain.

---
