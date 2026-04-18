# Acceptance Criteria

In trajectory-based meta-heuristics, the acceptance criterion defines the mathematical rule determining whether the search transitions from a current solution $s$ to a generated candidate solution $s'$. Assuming an objective function $f(\cdot)$ subject to minimization, the criteria below are organized along two axes: the information they use to reach a decision (cost-only vs. structural/historical vs. population-level), and the determinism of the outcome (fully deterministic vs. stochastic). Let $\Delta f = f(s') - f(s)$.

---

## Greedy & Elitist Criteria

These criteria accept moves based solely on immediate, deterministic cost comparisons with no tolerance for objective deterioration. They form the bedrock of descent-based local search and serve as the acceptance core for tabu search.

### Only Improving (OI) ✅
The strictest greedy, elitist acceptance rule. The search trajectory is monotonically non-increasing, accepting a candidate only if it yields a strict objective reduction:

$$f(s') < f(s) \quad (\Delta f < 0)$$

Used as the inner loop of pure hill-climbing and steepest-descent local search.

### Improving and Equal (IE) ✅
A weakly elitist strategy that accepts solutions of equal or superior quality:

$$f(s') \le f(s) \quad (\Delta f \le 0)$$

Unlike OI, IE admits random walks across neutral plateaus (fitness-flat regions), mitigating premature stagnation in combinatorial landscapes with large degenerate regions.

### Tabu Criterion (TC) ❌
The operational acceptance core of Tabu Search. A candidate move $(s \to s')$ is **accepted** if and only if the move is not classified as tabu — i.e., the move attribute does not appear in the tabu list $\mathcal{T}$:

$$\text{Accept}(s') \iff \text{attr}(s \to s') \notin \mathcal{T}$$

The tabu list $\mathcal{T}$ holds recently applied move attributes (e.g., node identities, edge pairs) for a configurable tenure $|\mathcal{T}|$ to explicitly prohibit revisiting recently explored configurations. The Aspiration Criterion (below) acts as its override mechanism. Together, TC and AC define the full tabu search acceptance rule.

### Aspiration Criterion (AC) ✅
An override mechanism for Tabu Search that supersedes the TC rejection when a tabu-classified candidate achieves a strict new global best:

$$\text{Accept}(s') \iff f(s') < f(s^*)$$

Even if $\text{attr}(s \to s') \in \mathcal{T}$, the move is admitted because its quality strictly surpasses any previously discovered solution, preventing the tabu mechanism from blocking globally superior solutions.

### All Moves Accepted (AMA) ✅
An extreme, non-selective rule that unconditionally accepts every generated candidate:

$$P(\text{accept } s') = 1$$

Equivalent to a pure random walk. Used as a naive diversification baseline or as a stress test for operator libraries.

---

## Thermodynamic & Stochastic Criteria

These criteria introduce controlled stochasticity into the acceptance decision, permitting occasional uphill moves with a probability that depends on the magnitude of deterioration and a tunable control parameter. This allows the search to escape local optima basins while still being biased toward improvement.

### Boltzmann–Metropolis Criterion (BMC) ✅
The canonical thermodynamic acceptance rule, derived from statistical mechanics (Kirkpatrick et al., 1983). Improving moves are accepted deterministically; deteriorating moves are accepted with a probability governed by the Boltzmann distribution:

$$P(\text{accept } s') = \begin{cases} 1 & \text{if } \Delta f \le 0 \\ \exp\!\left(-\dfrac{\Delta f}{T}\right) & \text{if } \Delta f > 0 \end{cases}$$

The artificial temperature $T > 0$ is reduced over time via a cooling schedule. Common schedules include:
- **Geometric:** $T_{k+1} = \alpha T_k$, $\alpha \in (0.9, 0.999)$ — the standard default
- **Linear:** $T_k = T_0 - k \cdot \delta$ — uniform cooling
- **Hyperbolic:** $T_k = T_0 / (1 + k)$ — slower initial cooling
- **Logarithmic:** $T_k = T_0 / \ln(1 + k)$ — theoretically convergent but practically slow

The cooling schedule is the primary tuning lever of Simulated Annealing solvers.

### Cauchy (Lorentz) Simulated Annealing ❌
An alternative to BMC that replaces the Boltzmann (Gaussian-generating) distribution with a Cauchy (Lorentz) distribution, yielding heavier probability tails for accepting large deteriorations:

$$P(\text{accept } s') = \frac{1}{1 + (\Delta f / T)^2}$$

Because the Cauchy distribution has infinite variance, this criterion accepts large worsening moves far more frequently than BMC at the same temperature, dramatically increasing global exploration. It is particularly effective in landscapes with deep, widely-separated basins separated by high barriers.

### Adaptive Boltzmann–Metropolis Criterion (ABMC) ✅
An advanced BMC variant where the temperature schedule is not predetermined but updated dynamically based on real-time search feedback — such as the rolling variance of accepted $\Delta f$ values or a target acceptance rate $\hat{p}$:

$$T_{k+1} = T_k \cdot \exp\!\left(\frac{\hat{p} - p_k}{\beta}\right)$$

where $p_k$ is the empirical acceptance rate over a recent window and $\beta$ is a sensitivity constant. This removes the need to pre-tune the cooling schedule by closing the feedback loop between the temperature and the landscape geometry.

### Generalized Tsallis Simulated Annealing (GTSA) ✅
Grounded in non-extensive Tsallis statistical mechanics, GTSA replaces the exponential Boltzmann distribution with a $q$-exponential, enabling a heavier-tailed and tunable probability distribution for accepting deteriorating moves:

$$P(\text{accept } s') = \left[1 - (1-q)\frac{\Delta f}{T}\right]^{\frac{1}{1-q}}$$

The entropic parameter $q \in \mathbb{R}$ governs tail weight: $q \to 1$ recovers the standard Boltzmann distribution; $q > 1$ yields super-Gaussian tails (more permissive of large deteriorations); $q < 1$ yields sub-Gaussian tails (stricter). Tsallis SA exhibits faster convergence than standard SA on certain multi-modal landscapes where frequent long-range jumps are beneficial.

### Monte Carlo (MC) ✅
A static-temperature Metropolis rule: either fixes $T$ at a constant throughout the search (rather than cooling it), or applies a fixed parameterized probability $p$ to accept any worsening move regardless of $\Delta f$ magnitude. Used primarily as a controlled-temperature baseline or in outer-loop exploration phases where cooling is undesirable.

### Exponential Monte Carlo with Counter (EMCC) ✅
A lightweight alternative to full SA that modulates the acceptance probability of deteriorating moves via an exponentially decaying iteration counter $c$, eliminating the need for a formal temperature schedule:

$$P(\text{accept } s') = \exp\!\left(-\frac{\Delta f}{c}\right)$$

The counter $c$ decreases monotonically with search iterations, mechanically tightening the acceptance threshold over time. Computationally cheaper than SA because it avoids temperature-reheating or schedule tuning.

### Probabilistic Transition (PT) ✅
A state-dependent stochastic acceptance rule that defines acceptance probability via a Markov transition function $P(s \to s')$, explicitly conditioned on the current search state and its position in the trajectory history. Unlike BMC — which depends only on $\Delta f$ and $T$ — PT may incorporate features such as solution age, visit frequency, or distance from the global best to bias the Markov chain toward historically productive regions of the landscape.

### Fitness Proportional (FP) ✅
A stochastic rule where the acceptance probability is proportional to the relative fitness of $s'$ against the incumbent $s$:

$$P(\text{accept } s') = \frac{F(s')}{F(s) + F(s')}$$

where $F(\cdot)$ is a non-negative fitness transformation of $f$ (e.g., $F(s) = 1 / f(s)$ for minimization). Widely used in evolutionary local search and population selection mechanisms; it naturally biases acceptance toward superior candidates while admitting inferior ones at lower but non-zero probability.

---

## Deterministic Bounded-Deterioration Criteria

These criteria deterministically admit or reject a candidate based on an explicit, bounded tolerance for objective worsening — no probabilistic sampling is involved. They provide the predictability of greedy descent with a controlled margin of flexibility to escape shallow local optima.

### Threshold Accepting (TA) ✅
A deterministic analogue to Simulated Annealing (Dueck & Scheuer, 1990). A candidate is accepted if the objective degradation remains within a monotonically decreasing threshold $\tau$:

$$f(s') - f(s) \le \tau \quad \Leftrightarrow \quad \Delta f \le \tau$$

$\tau$ begins at a large initial value (permitting broad exploration) and is reduced toward zero as the search progresses, tightening the criterion to pure hill-climbing at convergence. Because acceptance is deterministic — no random sampling — two runs with identical inputs are fully reproducible.

### Record-to-Record Travel (RRT) ✅
A deterministic, deviation-bounded rule relative to the global best rather than the current incumbent (Dueck, 1993). A candidate is accepted if its cost is within a fixed scalar deviation $\delta$ of the best-known solution $s^*$:

$$f(s') \le f(s^*) + \delta$$

RRT exhibits a natural self-regulating property: as $s^*$ improves, the absolute acceptance ceiling $f(s^*) + \delta$ automatically tightens, eliminating the need to schedule the deviation parameter. $\delta$ is typically held constant or reduced with a simple linear schedule.

### Great Deluge (GD) ✅
A deterministic acceptance rule that uses an absolute cost boundary — the "water level" $W$ — rather than a relative threshold (Dueck, 1993). A candidate is accepted if and only if its cost remains below the current water level:

$$f(s') \le W$$

$W$ is initialized above the starting solution cost (for maximization problems, below it) and decreases monotonically at a fixed decay rate $\Delta W$ per iteration, forcing convergence by progressively narrowing the acceptance window.

### Non-Linear Great Deluge (NLGD) ✅
Extends GD by decaying the water level $W$ non-linearly — exponentially or logarithmically — rather than by a fixed scalar per step:

$$W_{k+1} = W_k \cdot e^{-\lambda}$$

The non-linear schedule enables rapid initial exploration (high $W$, broad acceptance) that asymptotically tightens as the search converges, closely mirroring the effective behaviour of geometric-cooling SA but with a fully deterministic acceptance test.

### Demon Algorithm (DA) ✅
A bounded-deterioration criterion that maintains an internal energy reserve $D \ge 0$ (the "demon"). Worsening moves are accepted if — and only if — the demon possesses sufficient energy to absorb the cost increase:

$$\Delta f \le D \;\Rightarrow\; \text{Accept};\quad D \leftarrow D - \Delta f$$

Improving moves replenish the demon's energy: $D \leftarrow D + |\Delta f|$. The demon therefore acts as a self-regulating, move-history-aware acceptor: it is strict when the search has been improving (low $D$) and permissive after a streak of deterioration has depleted it. Unlike SA, the demon's energy is not externally scheduled — it emerges organically from the search trajectory.

---

## Structural & Distance-Based Criteria

These criteria base the acceptance decision not solely on cost but on the structural distance or novelty of the candidate relative to the incumbent or best-known solution. They are particularly valuable in meta-heuristics operating on combinatorial landscapes where cost similarity does not imply structural similarity.

### Skewed Variable Neighborhood Search (SVNS) ✅
An acceptance criterion that permits worsening moves if the candidate is sufficiently far — in structural solution space — from the current incumbent (Hansen et al., 2006). This counteracts the tendency of pure cost-based criteria to cluster the trajectory in a narrow structural neighbourhood:

$$f(s') - \alpha \cdot \rho(s, s') < f(s)$$

where $\rho(s, s')$ is a structural distance metric over the solution space (e.g., number of differing edges, Hamming distance over assignment vectors) and $\alpha > 0$ is an asymmetry parameter scaling the distance bonus. Candidates that are both geometrically remote and only modestly worse than $s$ are admitted; candidates that are spatially adjacent and costly are rejected.

### Population-Distance Acceptance (PDA) ❌
A criterion for population-based frameworks (evolutionary algorithms, memetic algorithms, island models) that conditions acceptance on the structural novelty of the candidate relative to the current population $\mathcal{P}$, rather than solely against the incumbent:

$$\text{Accept}(s') \iff f(s') < \bar{f}(\mathcal{P}) \;\lor\; \rho(s', \mathcal{P}) > \rho_{\min}$$

where $\bar{f}(\mathcal{P})$ is the population mean cost and $\rho(s', \mathcal{P}) = \min_{p \in \mathcal{P}} \rho(s', p)$ is the minimum structural distance from $s'$ to any existing population member. A candidate that is below average cost is accepted unconditionally; a candidate that is above average but structurally unique (distant from all population members) is also admitted to maintain diversity. This prevents premature convergence without requiring a global temperature parameter.

---

## Memory & History-Based Criteria

These criteria evaluate acceptance against historical search data — a sliding window of past costs, a step counter, or a dynamically adjusted threshold based on acceptance/rejection streaks. They adapt the tolerance automatically to the recent search trajectory without requiring an externally scheduled parameter.

### Late Acceptance Hill-Climbing (LAHC) ✅
A memory-based criterion that compares the candidate against the cost of the solution visited exactly $L$ steps ago, rather than the current incumbent (Burke & Bykov, 2017):

$$f(s') \le f(s_{i-L})$$

A circular cost-history array of length $L$ is maintained. When the search is in a flat or slowly improving region, $f(s_{i-L}) \approx f(s)$ and the criterion behaves like IE; when the search has been declining rapidly, the historical cost acts as a looser threshold that permits temporary uphill steps. The array length $L$ is the sole tuning parameter, and its value implicitly governs both the acceptance temperature and the effective search radius.

### Step Counting Hill Climbing (SCHC) ✅
A discrete memory criterion that replaces the continuous history array of LAHC with a single static cost bound $B$ that is refreshed every $L$ accepted steps:

$$f(s') \le B; \quad B \leftarrow f(s) \text{ every } L \text{ steps}$$

During the $L$-step window, $B$ is held fixed and acts as a flat acceptance ceiling. Upon expiry, $B$ is updated to the current incumbent, resetting the cycle. SCHC is strictly simpler than LAHC (one variable vs. an array of length $L$) while achieving comparable performance on many VRP benchmarks, making it attractive for embedded or memory-constrained implementations.

### Old Bachelor Acceptance (OBA) ✅
A non-monotone, adaptive threshold that dynamically adjusts based on the recent acceptance or rejection trajectory (Hu et al., 1995). If a move is accepted, the threshold is **tightened** (mimicking high selectivity after a recent success); if a streak of consecutive rejections occurs, the threshold is **progressively relaxed** to force trajectory escape:

$$\tau_{k+1} = \begin{cases} \tau_k - \delta_{\text{tight}} & \text{if last move accepted} \\ \tau_k + \delta_{\text{relax}} & \text{if last move rejected} \end{cases}$$

The acceptance rule is then $f(s') - f(s) \le \tau_k$. OBA is self-regulating: a long stagnation streak will automatically lower the bar, while a run of improvements will raise it, avoiding the manual specification of cooling schedules. The two adjustment deltas $\delta_{\text{tight}}$ and $\delta_{\text{relax}}$ are its primary tuning parameters.

---

## Multi-Objective & Ensemble Criteria

These criteria operate on vector-valued objective functions or aggregate decisions from multiple independent criteria. They are the acceptance backbone of multi-objective meta-heuristics, Pareto-archive managers, and ensemble solvers.

### Pareto Dominance (PD) ✅
The fundamental strict acceptance criterion for multi-objective landscapes. A candidate $s'$ replaces the incumbent $s$ if and only if it strictly dominates it across all objectives $i = 1, \dots, m$:

$$f_i(s') \le f_i(s)\; \forall i \quad \land \quad f_j(s') < f_j(s) \text{ for at least one } j$$

In practice, Pareto-based meta-heuristics maintain a non-dominated archive and accept a candidate if it is non-dominated with respect to the archive, adding it to the archive and removing any solutions it dominates.

### $\epsilon$-Dominance ($\epsilon$-Dom) ✅
A relaxed Pareto criterion that accelerates convergence by admitting candidates whose objective improvements are mathematically significant but numerically negligible. A candidate $s'$ $\epsilon$-dominates $s$ if:

$$(1 - \epsilon) f_i(s') \le f_i(s) \quad \forall i$$

This prevents infinitesimally marginal improvements from stalling the search or cluttering the Pareto archive. The parameter $\epsilon > 0$ sets the granularity of the dominance check; larger $\epsilon$ coarsens the Pareto front, accelerating convergence at the cost of archive precision.

### Stochastic Pareto Acceptance (SPA) ❌
A probabilistic relaxation of strict Pareto dominance designed for noisy or uncertain multi-objective landscapes. Rather than requiring strict dominance, SPA accepts $s'$ with a probability inversely proportional to the degree of inferiority:

$$P(\text{accept } s') = \exp\!\left(-\frac{\max_i \max(0,\, f_i(s') - f_i(s))}{T}\right)$$

The parameter $T$ controls the stringency; as $T \to 0$, SPA recovers deterministic Pareto dominance. SPA is used in multi-objective simulated annealing variants where evaluations are stochastic and exact dominance comparisons are unreliable.

### Tournament Acceptance ✅
The candidate $s'$ is evaluated against a randomly sampled subset $\mathcal{C} \subset \mathcal{P}$ of $\kappa$ competitors drawn from the current population or archive. It is accepted into the active trajectory if it strictly outperforms a required fraction of those competitors:

$$|\{c \in \mathcal{C} : f(s') < f(c)\}| \ge \kappa_{\min}$$

Tournament acceptance is the standard mechanism in genetic algorithms and memetic algorithms for survivor selection, balancing selection pressure (via $\kappa$) and population diversity (via random competitor sampling).

### Ensemble Move Acceptance (EMA) ✅
A meta-decision architecture that evaluates a candidate move through a portfolio of heterogeneous criteria concurrently — for example, BMC, GD, and IE simultaneously. The final acceptance decision is aggregated via configurable logical ensemble rules:

- **G-AND (Strict Consensus):** Accept only if all criteria agree — conservative, high-precision
- **G-OR (Authority Rule):** Accept if any single criterion votes positively — aggressive, high-diversity
- **G-VOT (Majority Vote):** Accept if a strict majority of criteria are satisfied — balanced
- **G-PVO (Probabilistic Vote):** Accept with probability proportional to the fraction of satisfied criteria — smooth interpolation between G-OR and G-AND

EMA avoids the brittleness of any single criterion while leveraging the complementary strengths of a diverse portfolio. The ensemble composition itself can be subject to adaptive reweighting (e.g., via Thompson Sampling over the criteria) to favour the best-performing criterion on the current instance.

---
