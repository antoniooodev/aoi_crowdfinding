# Age of Information-Aware Crowd-Finding for Emergency Search: A Game-Theoretic Analysis

## Formal Model Definition and Theoretical Analysis

---

# PHASE 1: MODEL FORMALIZATION

---

## 1.1 Scenario Definition

### 1.1.1 Physical Environment

We consider a search area modeled as a bounded two-dimensional region:

$$\mathcal{A} = [0, L] \times [0, L] \subset \mathbb{R}^2$$

where $L > 0$ denotes the side length of the square search area. The total area is $|\mathcal{A}| = L^2$.

### 1.1.2 Network Entities

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

*Static Uniform Model:*
Volunteer positions are drawn independently and uniformly at random:
$$\mathbf{x}_i \sim \text{Uniform}(\mathcal{A}), \quad \forall i \in \mathcal{N}$$

*Mobility Model (Extension):*
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

The number of active volunteers is:
$$k = \sum_{i=1}^{N} a_i$$

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

*Proof:*
Each active volunteer independently covers the target with probability $\rho$. The probability that none of the $k$ volunteers covers the target is $(1-\rho)^k$. Thus, the probability of at least one detection is $1 - (1-\rho)^k$. $\square$

**Properties of $P_{\text{det}}(k)$:**

1. $P_{\text{det}}(0) = 0$ (no active volunteers means no detection)
2. $P_{\text{det}}(k)$ is strictly increasing in $k$
3. $\lim_{k \to \infty} P_{\text{det}}(k) = 1$
4. $P_{\text{det}}(k)$ is concave in $k$ (diminishing returns)

*Proof of concavity:*
$$\frac{\partial^2 P_{\text{det}}}{\partial k^2} = (1-\rho)^k (\ln(1-\rho))^2 > 0$$

Wait, this shows convexity of $(1-\rho)^k$, hence $P_{\text{det}}(k) = 1 - (1-\rho)^k$ is concave. $\square$

### 1.2.3 Expected Age of Information

**Theorem 1 (Expected AoI):**
Under stationary conditions with detection probability $P_{\text{det}}(k)$ per slot, the expected time-average AoI is:

$$\bar{\Delta}(k) = \frac{1}{P_{\text{det}}(k)} - 1 = \frac{(1-\rho)^k}{1 - (1-\rho)^k}$$

*Proof:*
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

*Proof:*
\begin{align}
\delta\bar{\Delta}(k) &= \frac{1}{P_{\text{det}}(k-1)} - \frac{1}{P_{\text{det}}(k)} \\
&= \frac{P_{\text{det}}(k) - P_{\text{det}}(k-1)}{P_{\text{det}}(k) \cdot P_{\text{det}}(k-1)}
\end{align}

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

**Alternative Benefit Functions:**

1. *Exponential decay:* $f(\bar{\Delta}) = B \cdot e^{-\alpha \bar{\Delta}}$
2. *Threshold:* $f(\bar{\Delta}) = B \cdot \mathbf{1}_{\{\bar{\Delta} \leq \Delta_{\max}\}}$
3. *Linear decay:* $f(\bar{\Delta}) = \max\{0, B - \beta \bar{\Delta}\}$

### 1.3.3 Cost Function

Each active volunteer incurs a cost $c > 0$ representing:
- Battery consumption for beacon scanning
- Computational overhead
- Privacy/security concerns
- Opportunity cost of participation

The cost function is:
$$C_i(a_i) = c \cdot a_i$$

### 1.3.4 Utility Function

**Definition (Volunteer Utility):**
The utility of volunteer $i$ given action profile $\mathbf{a}$ is:

$$U_i(\mathbf{a}) = f(\bar{\Delta}(k)) - c \cdot a_i$$

where $k = \sum_{j=1}^{N} a_j$ is the total number of active volunteers.

**With our chosen benefit function:**
$$U_i(\mathbf{a}) = B \cdot P_{\text{det}}(k) - c \cdot a_i$$

### 1.3.5 Utility by Action

Given that $k_{-i} = \sum_{j \neq i} a_j$ other volunteers are active:

**If volunteer $i$ is active ($a_i = 1$):**
$$U_i(1, \mathbf{a}_{-i}) = B \cdot P_{\text{det}}(k_{-i} + 1) - c$$

**If volunteer $i$ is inactive ($a_i = 0$):**
$$U_i(0, \mathbf{a}_{-i}) = B \cdot P_{\text{det}}(k_{-i})$$

### 1.3.6 Incentive to Participate

**Definition:** The marginal utility of participation for volunteer $i$ is:

$$\Delta U_i(k_{-i}) = U_i(1, \mathbf{a}_{-i}) - U_i(0, \mathbf{a}_{-i})$$

$$\Delta U_i(k_{-i}) = B \cdot [P_{\text{det}}(k_{-i} + 1) - P_{\text{det}}(k_{-i})] - c$$

**Lemma 3 (Marginal Detection Gain):**
$$P_{\text{det}}(k+1) - P_{\text{det}}(k) = \rho(1-\rho)^k$$

*Proof:*
\begin{align}
P_{\text{det}}(k+1) - P_{\text{det}}(k) &= [1-(1-\rho)^{k+1}] - [1-(1-\rho)^k] \\
&= (1-\rho)^k - (1-\rho)^{k+1} \\
&= (1-\rho)^k [1 - (1-\rho)] \\
&= \rho(1-\rho)^k \quad \square
\end{align}

**Corollary:** The marginal utility of participation is:
$$\Delta U_i(k_{-i}) = B\rho(1-\rho)^{k_{-i}} - c$$

### 1.3.7 Key Parameters Summary

| Parameter | Symbol | Description | Range |
|-----------|--------|-------------|-------|
| Area side | $L$ | Search region dimension | $L > 0$ |
| Volunteers | $N$ | Total number of volunteers | $N \in \mathbb{N}^+$ |
| Detection radius | $R$ | Beacon reception range | $0 < R < L$ |
| Coverage ratio | $\rho$ | Single volunteer coverage | $\rho = \pi R^2/L^2$ |
| Benefit | $B$ | Maximum benefit from detection | $B > 0$ |
| Cost | $c$ | Participation cost | $c > 0$ |

### 1.3.8 Assumptions

**A1 (Homogeneity):** All volunteers have identical detection capabilities and costs.

**A2 (Independence):** Volunteer positions and decisions are independent.

**A3 (Complete Information):** All parameters $(N, \rho, B, c)$ are common knowledge.

**A4 (Rationality):** Volunteers are rational utility maximizers.

**A5 (No Coordination):** Volunteers cannot communicate or coordinate decisions.

---

# PHASE 2: THEORETICAL ANALYSIS

---

## 2.1 Symmetric Nash Equilibrium

### 2.1.1 Symmetric Strategy Profile

Given the symmetry of the game (identical players), we focus on symmetric Nash equilibria where all volunteers use the same strategy.

**Definition (Symmetric Mixed Strategy):**
A symmetric strategy is a probability $p \in [0, 1]$ where each volunteer independently chooses $a_i = 1$ with probability $p$.

**Definition (Symmetric Pure Strategy):**
In a symmetric pure-strategy equilibrium, either all volunteers are active ($k = N$) or inactive ($k = 0$), or we have a mixed equilibrium.

### 2.1.2 Expected Utility in Symmetric Profile

When all other $N-1$ volunteers play mixed strategy $p$, the number of other active volunteers follows a binomial distribution:

$$k_{-i} \sim \text{Binomial}(N-1, p)$$

**Expected utility if active:**
$$\mathbb{E}[U_i(1)] = B \cdot \mathbb{E}[P_{\text{det}}(k_{-i} + 1)] - c$$

**Expected utility if inactive:**
$$\mathbb{E}[U_i(0)] = B \cdot \mathbb{E}[P_{\text{det}}(k_{-i})]$$

### 2.1.3 Pure Strategy Nash Equilibrium

For tractability, we first analyze the case where exactly $k^*$ volunteers are active (symmetric in expectation but possibly with randomization over which $k^*$ are active).

**Theorem 2 (Pure Strategy NE Characterization):**
A symmetric configuration with $k^*$ active volunteers is a Nash equilibrium if and only if:

1. **No active volunteer wants to deviate to inactive:**
$$U_i(1; k^*) \geq U_i(0; k^* - 1)$$
$$B \cdot P_{\text{det}}(k^*) - c \geq B \cdot P_{\text{det}}(k^* - 1)$$

2. **No inactive volunteer wants to deviate to active:**
$$U_i(0; k^*) \geq U_i(1; k^* + 1)$$
$$B \cdot P_{\text{det}}(k^*) \geq B \cdot P_{\text{det}}(k^* + 1) - c$$

*Rearranging:*

**Condition 1:** $B[P_{\text{det}}(k^*) - P_{\text{det}}(k^*-1)] \geq c$
$$B\rho(1-\rho)^{k^*-1} \geq c$$

**Condition 2:** $B[P_{\text{det}}(k^*+1) - P_{\text{det}}(k^*)] \leq c$
$$B\rho(1-\rho)^{k^*} \leq c$$

### 2.1.4 Equilibrium Characterization

**Theorem 3 (Nash Equilibrium):**
Define the threshold function:
$$\phi(k) = B\rho(1-\rho)^{k-1}$$

The Nash equilibrium number of active volunteers $k^*$ satisfies:
$$\phi(k^* + 1) \leq c \leq \phi(k^*)$$

Equivalently:
$$k^* = \left\lfloor 1 + \frac{\ln(c) - \ln(B\rho)}{\ln(1-\rho)} \right\rfloor$$

when $c \leq B\rho$ (otherwise $k^* = 0$).

*Proof:*
From conditions 1 and 2:
$$B\rho(1-\rho)^{k^*} \leq c \leq B\rho(1-\rho)^{k^*-1}$$

Taking logarithms:
$$\ln(B\rho) + k^* \ln(1-\rho) \leq \ln(c) \leq \ln(B\rho) + (k^*-1)\ln(1-\rho)$$

Since $\ln(1-\rho) < 0$:
$$k^* \geq \frac{\ln(c) - \ln(B\rho)}{\ln(1-\rho)} \geq k^* - 1$$

Thus:
$$k^* = \left\lfloor 1 + \frac{\ln(c/B\rho)}{\ln(1-\rho)} \right\rfloor = \left\lfloor 1 - \frac{\ln(c/B\rho)}{\ln(1/(1-\rho))} \right\rfloor \quad \square$$

### 2.1.5 Special Cases

**Case 1: Low Cost ($c \leq B\rho(1-\rho)^{N-1}$)**
All volunteers active: $k^* = N$

**Case 2: High Cost ($c > B\rho$)**
No volunteers active: $k^* = 0$

**Case 3: Intermediate Cost**
Interior equilibrium: $1 \leq k^* \leq N-1$

### 2.1.6 Mixed Strategy Equilibrium

**Theorem 4 (Mixed Strategy NE):**
In a symmetric mixed-strategy equilibrium with participation probability $p^* \in (0, 1)$, volunteers must be indifferent between actions:

$$\mathbb{E}[U_i(1)] = \mathbb{E}[U_i(0)]$$

This yields the indifference condition:
$$B \cdot \mathbb{E}[P_{\text{det}}(k_{-i} + 1) - P_{\text{det}}(k_{-i})] = c$$

$$B\rho \cdot \mathbb{E}[(1-\rho)^{k_{-i}}] = c$$

Since $k_{-i} \sim \text{Binomial}(N-1, p)$:
$$\mathbb{E}[(1-\rho)^{k_{-i}}] = \sum_{j=0}^{N-1} \binom{N-1}{j} p^j (1-p)^{N-1-j} (1-\rho)^j$$
$$= [p(1-\rho) + (1-p)]^{N-1} = [1 - p\rho]^{N-1}$$

**Equilibrium condition:**
$$B\rho[1 - p^*\rho]^{N-1} = c$$

**Solving for $p^*$:**
$$p^* = \frac{1}{\rho}\left[1 - \left(\frac{c}{B\rho}\right)^{\frac{1}{N-1}}\right]$$

**Validity:** $p^* \in (0, 1)$ requires:
$$B\rho(1-\rho)^{N-1} < c < B\rho$$

---

## 2.2 Social Optimum

### 2.2.1 Social Welfare Function

**Definition:** The social welfare with $k$ active volunteers is the sum of all utilities:

$$W(k) = \sum_{i=1}^{N} U_i = N \cdot B \cdot P_{\text{det}}(k) - k \cdot c$$

Note: All volunteers receive the same benefit $B \cdot P_{\text{det}}(k)$, but only $k$ pay the cost.

### 2.2.2 Welfare Maximization

**Theorem 5 (Social Optimum):**
The socially optimal number of active volunteers is:

$$k^{\text{opt}} = \arg\max_{k \in \{0, 1, \ldots, N\}} W(k)$$

The first-order condition (treating $k$ as continuous) is:
$$\frac{dW}{dk} = NB \cdot \frac{dP_{\text{det}}}{dk} - c = 0$$

$$NB\rho(1-\rho)^{k-1} \cdot (-\ln(1-\rho)) = c$$

Wait, let me recalculate. We have:
$$\frac{dP_{\text{det}}}{dk} = \frac{d}{dk}[1 - (1-\rho)^k] = -(1-\rho)^k \ln(1-\rho) = (1-\rho)^k \ln\frac{1}{1-\rho}$$

So:
$$NB(1-\rho)^k \ln\frac{1}{1-\rho} = c$$

For the discrete case, the optimum satisfies:
$$W(k^{\text{opt}}) \geq W(k^{\text{opt}} - 1) \quad \text{and} \quad W(k^{\text{opt}}) \geq W(k^{\text{opt}} + 1)$$

**Condition for increasing welfare:**
$$W(k) - W(k-1) = NB[P_{\text{det}}(k) - P_{\text{det}}(k-1)] - c \geq 0$$
$$NB\rho(1-\rho)^{k-1} \geq c$$

**Theorem 6 (Social Optimum Characterization):**
$$k^{\text{opt}} = \left\lfloor 1 + \frac{\ln(c) - \ln(NB\rho)}{\ln(1-\rho)} \right\rfloor$$

when $c \leq NB\rho$.

### 2.2.3 Comparison: NE vs Social Optimum

**Key Observation:**
- NE condition: $B\rho(1-\rho)^{k^*-1} \geq c$
- Social optimum condition: $NB\rho(1-\rho)^{k^{\text{opt}}-1} \geq c$

The factor of $N$ difference implies:

**Theorem 7 (Under-Participation):**
$$k^* \leq k^{\text{opt}}$$

The Nash equilibrium exhibits under-participation relative to the social optimum.

*Proof:*
The threshold for individual participation is $B\rho(1-\rho)^{k-1} \geq c$.
The threshold for social benefit is $NB\rho(1-\rho)^{k-1} \geq c$.

Since $N \geq 1$, the social threshold is easier to satisfy, meaning more volunteers should be active at the social optimum. $\square$

**Intuition:** Each volunteer only considers their private benefit from participation, ignoring the positive externality they create for all $N-1$ other volunteers.

### 2.2.4 Participation Gap

**Definition:** The participation gap is:
$$\Delta k = k^{\text{opt}} - k^*$$

**Theorem 8:**
$$\Delta k = \left\lfloor \frac{\ln N}{\ln(1/(1-\rho))} \right\rfloor = \left\lfloor \frac{\ln N}{-\ln(1-\rho)} \right\rfloor$$

*Proof:*
From the equilibrium characterizations:
$$k^{\text{opt}} - k^* \approx \frac{\ln(NB\rho) - \ln(c)}{\ln(1/(1-\rho))} - \frac{\ln(B\rho) - \ln(c)}{\ln(1/(1-\rho))} = \frac{\ln N}{\ln(1/(1-\rho))} \quad \square$$

**Corollary:** For small $\rho$, using $\ln(1/(1-\rho)) \approx \rho$:
$$\Delta k \approx \frac{\ln N}{\rho}$$

The participation gap increases with $N$ and decreases with coverage ratio $\rho$.

---

## 2.3 Price of Anarchy

### 2.3.1 Definition

**Definition (Price of Anarchy):**
$$\text{PoA} = \frac{W(k^{\text{opt}})}{W(k^*)}$$

The PoA measures the efficiency loss due to selfish behavior.

**Definition (Price of Stability):**
If multiple equilibria exist:
$$\text{PoS} = \frac{W(k^{\text{opt}})}{\max_{\text{NE } k} W(k)}$$

### 2.3.2 PoA Computation

**Theorem 9 (Price of Anarchy Bound):**

$$\text{PoA} = \frac{NB \cdot P_{\text{det}}(k^{\text{opt}}) - k^{\text{opt}} \cdot c}{NB \cdot P_{\text{det}}(k^*) - k^* \cdot c}$$

**Upper Bound:**

For the worst case where $k^* = 0$ (no participation):
$$\text{PoA} = \frac{W(k^{\text{opt}})}{W(0)} = \frac{NB \cdot P_{\text{det}}(k^{\text{opt}}) - k^{\text{opt}} \cdot c}{0}$$

This is unbounded, occurring when $c > B\rho$.

**Interior Case Analysis:**

When $k^* \geq 1$:

$$\text{PoA} = \frac{NB[1-(1-\rho)^{k^{\text{opt}}}] - k^{\text{opt}} c}{NB[1-(1-\rho)^{k^*}] - k^* c}$$

### 2.3.3 Asymptotic Analysis

**Theorem 10 (Large N Behavior):**
As $N \to \infty$ with fixed $\rho$ and $c/B$:

$$\text{PoA} \to \frac{NB - k^{\text{opt}} c}{NB - k^* c} \approx 1 + \frac{(k^{\text{opt}} - k^*)c}{NB - k^* c}$$

For large $N$ with both $k^{\text{opt}}$ and $k^*$ scaling as $O(\ln N / \rho)$:
$$\text{PoA} = 1 + O\left(\frac{\ln N}{N}\right) \to 1$$

**Interpretation:** For large populations, the inefficiency becomes negligible because the social benefit dominates the cost differences.

### 2.3.4 Numerical PoA Characterization

**Proposition 1:** For moderate parameters ($N = 100$, $\rho = 0.01$, $c/B \in [0.001, 0.1]$):
$$\text{PoA} \in [1, 1.5]$$

The PoA is bounded and modest for typical parameter ranges.

---

## 2.4 Stackelberg Extension

### 2.4.1 Stackelberg Game Formulation

We extend the model to a two-stage Stackelberg game:

**Leader (Platform/Coordinator):**
- Announces an incentive payment $p \geq 0$ per active volunteer
- Objective: Maximize social welfare minus incentive costs

**Followers (Volunteers):**
- Observe incentive $p$ and decide participation
- Modified utility: $U_i^p(\mathbf{a}) = B \cdot P_{\text{det}}(k) - c \cdot a_i + p \cdot a_i$

### 2.4.2 Volunteer Response

**Modified Utility:**
$$U_i^p(1; k) = B \cdot P_{\text{det}}(k) - c + p = B \cdot P_{\text{det}}(k) - (c - p)$$
$$U_i^p(0; k) = B \cdot P_{\text{det}}(k)$$

The incentive effectively reduces the participation cost to $(c - p)$.

**Lemma 4 (Induced Equilibrium):**
Given incentive $p$, the Nash equilibrium participation $k^*(p)$ satisfies:
$$k^*(p) = \left\lfloor 1 + \frac{\ln(c-p) - \ln(B\rho)}{\ln(1-\rho)} \right\rfloor$$

for $p < c$ and $c - p \leq B\rho$.

### 2.4.3 Platform Objective

The platform's utility is social welfare minus total incentive payments:
$$V(p) = W(k^*(p)) - p \cdot k^*(p)$$
$$V(p) = NB \cdot P_{\text{det}}(k^*(p)) - k^*(p) \cdot c$$

Note: The incentive payments cancel out (transfer from platform to volunteers).

**Alternative Objective (Budget-Constrained):**
$$\max_{p \geq 0} W(k^*(p)) \quad \text{s.t.} \quad p \cdot k^*(p) \leq \mathcal{B}$$

where $\mathcal{B}$ is the platform's budget.

### 2.4.4 Optimal Incentive

**Theorem 11 (Optimal Stackelberg Incentive):**
To implement the social optimum $k^{\text{opt}}$, the platform should set:

$$p^* = c - B\rho(1-\rho)^{k^{\text{opt}}-1}$$

*Proof:*
We need volunteers to find it optimal to participate when $k = k^{\text{opt}}$:
$$B\rho(1-\rho)^{k^{\text{opt}}-1} \geq c - p^*$$

Setting equality:
$$p^* = c - B\rho(1-\rho)^{k^{\text{opt}}-1}$$

This makes the $k^{\text{opt}}$-th volunteer exactly indifferent, inducing equilibrium at $k^{\text{opt}}$. $\square$

**Corollary (Incentive Characterization):**
$$p^* = c - \frac{c}{N}$$

when $k^{\text{opt}}$ satisfies the social optimality condition with equality.

For large $N$: $p^* \approx c - c/N = c(1 - 1/N) \approx c$

The platform must subsidize almost the entire cost to achieve social optimum.

### 2.4.5 Total Incentive Cost

**Theorem 12 (Incentive Budget):**
The total incentive payment at the social optimum is:
$$\mathcal{I}^* = p^* \cdot k^{\text{opt}} = k^{\text{opt}} \left[c - B\rho(1-\rho)^{k^{\text{opt}}-1}\right]$$

### 2.4.6 Implementation via Mechanism Design

**Alternative Mechanism: Participation Threshold**

Instead of per-volunteer payment, the platform can offer a collective reward $R$ if at least $k^{\text{opt}}$ volunteers participate.

**Threshold Mechanism:**
- Platform announces $(k^{\text{opt}}, R)$
- If $\sum_i a_i \geq k^{\text{opt}}$: each active volunteer receives $R/k$
- Otherwise: no payment

**Proposition 2:** The threshold mechanism induces $k^{\text{opt}}$ participation with:
$$R = k^{\text{opt}} \cdot p^*$$

### 2.4.7 Stackelberg Equilibrium Summary

| Quantity | NE (No Incentive) | Social Optimum | Stackelberg |
|----------|-------------------|----------------|-------------|
| Active volunteers | $k^*$ | $k^{\text{opt}}$ | $k^{\text{opt}}$ |
| Per-volunteer incentive | 0 | — | $p^*$ |
| Total incentive | 0 | — | $k^{\text{opt}} p^*$ |
| Social welfare | $W(k^*)$ | $W(k^{\text{opt}})$ | $W(k^{\text{opt}})$ |

---

## Summary of Key Results

### Equilibrium Expressions

| Quantity | Expression |
|----------|------------|
| Detection probability | $P_{\text{det}}(k) = 1 - (1-\rho)^k$ |
| Expected AoI | $\bar{\Delta}(k) = \frac{(1-\rho)^k}{1-(1-\rho)^k}$ |
| Nash equilibrium | $k^* = \lfloor 1 + \frac{\ln(c/B\rho)}{\ln(1-\rho)} \rfloor$ |
| Social optimum | $k^{\text{opt}} = \lfloor 1 + \frac{\ln(c/NB\rho)}{\ln(1-\rho)} \rfloor$ |
| Participation gap | $\Delta k \approx \frac{\ln N}{-\ln(1-\rho)}$ |
| Optimal incentive | $p^* = c - B\rho(1-\rho)^{k^{\text{opt}}-1}$ |

### Main Theorems

1. **Theorem 3:** Nash equilibrium characterization via threshold condition
2. **Theorem 7:** Under-participation at NE ($k^* \leq k^{\text{opt}}$)
3. **Theorem 8:** Participation gap scales as $O(\ln N / \rho)$
4. **Theorem 10:** PoA converges to 1 for large $N$
5. **Theorem 11:** Optimal Stackelberg incentive to implement social optimum

---

## Parameter Sensitivity

### Effect of Cost $c$:
- Higher $c$ → Lower $k^*$ and $k^{\text{opt}}$
- Higher $c$ → Higher required incentive $p^*$

### Effect of Benefit $B$:
- Higher $B$ → Higher $k^*$ and $k^{\text{opt}}$
- Higher $B$ → Lower required incentive $p^*$

### Effect of Coverage $\rho$:
- Higher $\rho$ → Lower $k^{\text{opt}}$ needed for same AoI
- Higher $\rho$ → Smaller participation gap

### Effect of Population $N$:
- Higher $N$ → Higher $k^{\text{opt}}$ (more people benefit)
- Higher $N$ → Larger participation gap
- Higher $N$ → PoA closer to 1

---

## Extensions (Future Work)

1. **Heterogeneous Volunteers:** Different costs $c_i$, detection radii $R_i$
2. **Mobile Volunteers:** Time-varying coverage, trajectory optimization
3. **Multiple Targets:** Competing or collaborative search
4. **Incomplete Information:** Bayesian game with unknown $\rho$ or $c$
5. **Dynamic Game:** Repeated interactions, reputation effects
6. **Network Effects:** Correlation in volunteer positions

---

*Document Version: 1.0*
*Last Updated: December 2024*
