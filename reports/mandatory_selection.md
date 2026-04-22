# Mandatory Node Selection Strategies

Domain-specific filtering algorithms that determine which nodes *must* be obligatorily visited on a given operational day in multi-period combinatorial optimization (such as the Vehicle Routing Problem with Profits and the Inventory Routing Problem). These strategies serve two distinct roles: pure **overflow prevention** (guaranteeing no node exceeds its physical capacity) and **economic prioritization** (selecting the most profitable subset given routing constraints). Some strategies operate purely in the temporal or fill-level domain; others exploit spatial routing structure to select nodes that yield maximum density or synergy.

---

## Static Baselines

### Regular Selection ✅
A static, periodic scheduling policy operating entirely in the time domain, independent of stochastic accumulation states. Collection is triggered for all eligible nodes on operational day $t$ if:

$$t \equiv 1 \pmod{X}$$

for a fixed frequency $X$ (e.g., every 3 days). Computationally trivial but highly inefficient in stochastic environments; serves as a worst-case baseline for fixed-route comparisons.

### K-Means Geographic Sector Selection ✅
A static, spatially-aware cyclic scheduling policy that pre-partitions the node set into $G$ geographic sectors via $K$-means clustering on bin coordinates, then mandates exactly one sector per operational day on a round-robin rotation. On day $t$ the active sector index is:

$$\sigma(t) = (t - 1) \bmod G$$

and all nodes assigned to sector $\sigma(t)$ are obligatorily collected. The clustering minimises within-sector spatial variance, ensuring co-collected nodes are geographically proximate and naturally compressing per-day routing distance relative to a random or index-based partition. This formalises the dominant real-world practice of dividing service areas into fixed geographic zones assigned to specific weekdays, making it the most operationally realistic static baseline. Unlike Regular Selection (which mandates all nodes simultaneously every $X$ days), the sector rotation guarantees a bounded, predictable daily workload of approximately $n/G$ nodes. Spatially blind to fill-level dynamics and incapable of reacting to stochastic accumulation states; serves as the primary benchmark for strategies exploiting fill information or routing synergy.

### Staggered Regular Selection ✅
A static, temporally distributed variant of Regular Selection that eliminates the all-or-nothing load spikes of the base policy by assigning each node $i$ (0-indexed) a fixed phase offset:

$$\phi_i = i \bmod X$$

where $X$ is the collection period. Node $i$ is mandated on operational day $t$ if and only if:

$$(t - 1) \bmod X = \phi_i$$

The uniform distribution of offsets across $\{0, \ldots, X-1\}$ guarantees that on every day exactly $\lfloor n/X \rfloor$ or $\lceil n/X \rceil$ nodes are scheduled, converting the global period $X$ into $n$ independent per-node schedules that collectively cover the horizon without overlap. Crucially, phase assignment is derived from array position rather than spatial coordinates, making this strategy computationally free and parameter-identical to Regular Selection (sharing the same $X$ via `threshold`) while producing a fundamentally different load profile. Serves as the canonical load-balanced temporal baseline, isolating the cost of ignoring fill-level state from the cost of ignoring spatial structure.

### Bernoulli Trial Random Selection ✅
A stochastic null baseline in which each eligible node $i$ is independently mandated via a Bernoulli trial with fixed probability $p \in [0,1]$:

$$X_i \sim \text{Bernoulli}(p), \quad i = 1, \ldots, n$$

The realised selection set $S = \{i : X_i = 1\}$ has cardinality distributed as $|S| \sim \text{Binomial}(n, p)$, with $\mathbb{E}[|S|] = np$ and $\text{Var}[|S|] = np(1-p)$. Unlike fixed-$K$ random selection, the Bernoulli formulation decouples individual bin decisions: no bin's inclusion affects any other's probability, faithfully modelling a memoryless per-node assessment. This variability in set size is intentional — it reflects the stochasticity inherent in ignoring all state information entirely. Any deterministic strategy that fails to significantly outperform Bernoulli selection at matched expected cardinality ($K = np$) has a validity concern, making this the essential sanity-check baseline for the full strategy suite. The probability $p$ is supplied via the shared `threshold` parameter and is clipped to $[0, 1]$.

---

## Overflow Minimization and Risk Management

### Last-Minute Selection ✅
A reactive, fill-level-driven strategy that triggers an obligatory collection when a node's current fill ratio breaches a predefined critical threshold $\tau \in [0,1]$:

$$\frac{c_i}{Q_{\max}} \ge \tau$$

where $c_i$ is the current fill level and $Q_{\max}$ the maximum physical capacity. Acts as the primary overflow prevention backstop but suffers from spatial myopia — it considers no neighboring nodes and ignores routing synergy.

### Deadline-Driven Selection ✅
A deterministic temporal strategy that projects the exact operational deadline for each node via:

$$d^* = \left\lfloor\frac{Q_{\max} - c}{\mu}\right\rfloor$$

where $\mu$ is the mean daily accumulation rate. Any node whose deadline $d^*$ falls within the solver's look-ahead horizon is immediately mandated, preventing guaranteed overflows while remaining fully deterministic and parameter-free.

### Look-Ahead Selection ✅
A predictive, simulation-driven strategy that synchronizes multi-day collections. It identifies critically full nodes $S_{\text{crit}}$ and simulates their time-to-overflow to find the nearest required service day $d_{\text{next}}$. All unassigned nodes $j \notin S_{\text{crit}}$ whose fill level projects to overflow before the vehicle returns are mandated:

$$c_j + \mu_j \cdot d_{\text{next}} \ge Q_{\max}$$

By propagating the scheduling horizon from the most urgent node outward, this strategy naturally clusters co-incident collections, reducing fleet deployment frequency.

### Linear Service Level Prediction ✅
A statistical confidence-bound strategy for strict SLA adherence. It projects the worst-case future fill level over horizon $D$ using a linear accumulation variance approximation:

$$w_{\text{future}} = w_{\text{current}} + D\mu + D k\sigma$$

If this upper confidence bound breaches $Q_{\max}$, the node is mandated for immediate collection. The confidence factor $k$ governs the risk tolerance: larger $k$ triggers earlier, more conservative collections.

### Stochastic Regret Selection ✅
A statistically rigorous strategy based on Expected Overflow Regret (EOR). Let $s_i = Q_{\max} - c_i$ be remaining capacity; assume daily accumulation $X \sim \mathcal{N}(\mu, \sigma^2)$ with standard score $Z = (s_i - \mu)/\sigma$. The closed-form expected overflow (the regret of deferral) is:

$$\mathbb{E}[\max(0, X - s_i)] = \sigma\phi(Z) + (\mu - s_i)(1 - \Phi(Z))$$

Collection is mandated if this expectation exceeds an acceptable risk parameter $\gamma$. Unlike threshold-based strategies, this formulation directly quantifies the economic loss of inaction.

### Multi-Day Overflow Probability Selection ✅
A stochastic look-ahead strategy modeling accumulation over a $K$-day temporal horizon. Leveraging i.i.d. variance scaling ($\sigma_K = \sigma\sqrt{K}$), the strategy mandates collection if the tail probability of the $K$-day cumulative accumulation exceeding remaining capacity $Q_{\max} - c$ breaches an acceptable statistical risk threshold, naturally imposing tighter collection requirements on high-variance nodes.

### Conditional Value-at-Risk (CVaR) Selection ✅
A risk-averse paradigm evaluating tail risk of capacity violations. Future fill level $F \sim \mathcal{N}(c + \mu, \sigma^2)$; surplus variable $X = F - Q_{\max}$. Rather than thresholding the expected overflow, the strategy mandates collection if the expected overflow in the worst $(1-\alpha)$ fraction of outcomes — $\text{CVaR}_\alpha(\max(0,X))$ — exceeds a critical tolerance. CVaR is strictly more conservative than EOR-based selection, making it appropriate for penalty-heavy SLA regimes.

### Wasserstein Distributionally Robust Selection ✅
A robust optimization paradigm replacing Gaussian point-estimate assumptions with a Wasserstein-1 ambiguity ball of radius $\epsilon$ centered on the empirical demand distribution. Using the duality results of Mohajerin Esfahani and Kuhn (2018), it optimizes against the worst-case distribution within the ball. For the ReLU overflow loss function, the worst-case expectation simplifies to the nominal expectation plus $\epsilon$, yielding a tractable, distribution-free risk bound that requires no parametric assumption on the demand distribution.

### MIP Multiple-Knapsack Selection (Overflow-Minimizing Variant) ✅
An exact 0/1 MILP formulation explicitly engineered to minimize expected overflow losses across a look-ahead scenario tree. Models the homogeneous fleet as $K$ knapsacks of capacity $Q$. For each bin $i$ and scenario $s$ with probability $\pi_s$, let $o_i^{(s)} = \max(0, w_i^{(s)} - 100\%)$ be the projected overflow if not collected. The solver selects binary collection vector $x_i \in \{0,1\}$ to minimize:

$$\min_x \sum_i (1 - x_i)\!\left[\sum_s \pi_s \cdot o_i^{(s)} \cdot \hat{m}_i + P_i \cdot \Pr[\text{any overflow}_i]\right]$$

Lexicographic count constraints ($\sum_i x_{i,k} \ge \sum_i x_{i,k+1} \;\forall k$) eliminate fleet symmetry from the branch-and-bound search space.

---

## Profit Maximization and Economic Strategies

### Revenue Threshold Selection ✅
A value-based heuristic prioritizing nodes by the absolute monetary value of their current contents. Let $V$ be total volumetric capacity, $\rho$ waste density, $R_{\text{kg}}$ unit revenue. A node is mandated if its expected revenue exceeds a profitability threshold $\tau_{\text{rev}}$:

$$\left(\frac{c_i}{Q_{\max}}\right) V \rho R_{\text{kg}} > \tau_{\text{rev}}$$

### Profit-per-Kilometer (Spatial ROI) Selection ✅
A routing-aware economic heuristic that normalizes expected node value by a proxy for marginal insertion cost. The spatial Return on Investment is computed as $\text{Score}_i = r_i / (2d_{0,i})$, deprioritizing highly valuable but isolated nodes. A node is mandated only when this economic-spatial ratio exceeds a configured threshold, naturally avoiding costly detours for marginal profit.

### Fractional Knapsack (Density-Greedy) Selection ✅
A high-speed $\frac{1}{2}$-approximation of the profit-maximizing multiple-knapsack problem. Nodes are sorted by net-profit density (profit per unit mass) and greedily packed into vehicle capacities. The $\frac{1}{2}$-approximation guarantee is maintained by returning the maximum between the greedy-packed set and the single most profitable node that fits independently.

### Lagrangian Reduced-Cost Selection ✅
A relaxation-based strategy evaluating the LP relaxation of the profit-maximizing multiple-knapsack problem. The dual variable (shadow price) $\lambda^*$ of the binding global capacity constraint is extracted. Nodes with strictly positive Lagrangian reduced cost are mandated:

$$\bar{c}_i = (r_i - \text{cost} \cdot d_i) - \lambda^* m_i > 0$$

This provides an economically grounded prioritization: mandated nodes are exactly those whose marginal value exceeds the capacity's shadow price, aligning individual collection decisions with the system-level economic optimum.

### Filter-and-Fan Selection ✅
A meta-heuristic selection strategy adapted from Glover (1998) that balances economic value with capacity constraints through a two-phase search. 
1. **Filter Phase:** Evaluates all unassigned nodes using a composite score that amplifies separable net profit by spatial urgency: 
   $$\text{Score}_i = (r_i - 2d_{0,i} \cdot C) \cdot \left(1 + \frac{c_i}{Q_{\max}}\right)$$
   The top $k$ scoring nodes are retained to form an initial seed set (the filter beam).
2. **Fan Phase:** Systematically explores an add/remove neighborhood around the seed set up to a maximum search depth. Because the net-profit objective is mathematically separable, marginal move evaluations are $\mathcal{O}(1)$. The algorithm greedily accepts additions that improve total profit and ejects nodes that degrade it. 

A fail-safe mechanism then deterministically forces the inclusion of any node breaching the critical fill threshold $\tau$, ensuring operational safety alongside economic optimization.

---

## Spatial and Synergistic Routing Strategies

### Spatial Synergy Selection ✅
A geometric, density-aware strategy that amortizes fixed travel costs by co-locating collections. Critically full nodes $S_{\text{crit}}$ are identified via a severe threshold $\tau_{\text{crit}}$. A moderately full node $j$ is mandated if it lies within Euclidean radius $R$ of any critical node:

$$S_{\text{syn}} = \left\{j \;\middle|\; \frac{c_j}{Q_{\max}} \ge \tau_{\text{syn}} \land \exists\, i \in S_{\text{crit}} : d_{i,j} \le R\right\}$$

The final selection set is $S_{\text{crit}} \cup S_{\text{syn}}$. The bipartite threshold structure prevents trivially nearby low-fill nodes from being included.

### Clarke-Wright Savings Selection ✅
A spatial synergy filter derived from the Clarke-Wright VRP heuristic. Pairwise spatial savings are computed as $s_{ij} = d_{0,i} + d_{0,j} - d_{i,j}$. A node is mandated only if it satisfies a minimum fill requirement *and* exhibits strictly positive savings with at least one other critically full node, suppressing isolated routing in favor of cluster formations.

### Route-Cluster Synergy Selection ❌
A routing-state-aware strategy that conditions node selection on the *current partial route structure* rather than purely on fill levels or pairwise distances. For each unassigned node $i$, the marginal insertion cost into the nearest active route $R^*$ is computed:

$$\Delta C_i = \min_{(u,v) \in R^*} (d_{u,i} + d_{i,v} - d_{u,v})$$

A node is mandated if its fill level exceeds a moderate threshold *and* its marginal insertion cost relative to its expected revenue satisfies $\Delta C_i / r_i \le \epsilon_{\text{route}}$. Unlike static spatial synergy (which ignores current fleet positioning), this strategy adapts dynamically to where vehicles actually are — mandating nodes that the active fleet can absorb cheaply on its current trajectory.

### Set-Cover (Hub) Selection ✅
A geometric aggregation strategy that maps required visits to a Minimum Set Cover formulation. It defines a universe $U$ of critically full nodes and seeks a minimum-cardinality hub set such that every node in $U$ lies within service radius $R$ of a chosen hub. Solved via a greedy algorithm providing a theoretically guaranteed $\ln(|U|)$ approximation ratio.

### Submodular Facility Location Selection ✅
A subset selection strategy maximizing a submodular coverage objective with diminishing returns:

$$f(S) = \sum_{i \in \text{Bins}} \max\!\left(0,\; r_i - \alpha \min_{j \in S \cup \{0\}} d_{i,j}\right)$$

Maximized under a cardinality budget via the Lazy Greedy algorithm (Minoux, 1978), which exploits a max-priority queue of marginal gains to avoid redundant computations while maintaining a $(1 - 1/e)$ approximation guarantee.

### Supermodular Synergy Selection ✅
A clustering strategy maximizing an objective with increasing returns (supermodularity), balancing expected revenue against a lower-bound TSP tour length approximation:

$$f(S) = \sum_{i \in S} r_i - 2\alpha \sum_{i \in S} \min_{j \in S \setminus \{i\}} d(i,j)$$

The marginal routing cost decreases as the cluster grows, enabling localized clusters to collectively cross the viability threshold. A modified greedy approach with continuous re-evaluation of non-positive candidates allows the algorithm to construct dense, jointly profitable clusters that no node could justify individually.

---

## Simulation and Decision-Theoretic Strategies

### One-Step Rollout (Approximate DP) Selection ✅
A simulation-based look-ahead formalizing the decision as an optimal stopping problem. For each node, the expected future reward of two actions — "Collect Today" vs. "Defer to Tomorrow" — is evaluated. The "Defer" trajectory is simulated over finite horizon $H$ using a base heuristic (e.g., Last-Minute selection) and discounted by factor $\gamma$. A node is mandated if the expected discounted reward of immediate collection strictly dominates deferral.

### Multi-Step Rollout Selection ✅
Extends the one-step rollout by simulating the *full* multi-day trajectory rather than a single-step comparison. For each candidate collection set $S$, a simulation policy is rolled out over horizon $H$ to estimate total expected routing cost and overflow penalties:

$$V(S) = \mathbb{E}\left[\sum_{t=1}^H \gamma^t \left(C_{\text{route},t}(S) + C_{\text{overflow},t}(S)\right)\right]$$

The set $S^* = \arg\min_S V(S)$ is selected, with the expectation approximated via Monte Carlo sampling of demand trajectories. Multi-step rollout captures temporal dependencies that one-step lookahead misses — for instance, choosing to collect a node today to prevent a costly detour in three days — at the cost of higher computational overhead.

### Whittle Index (Restless Multi-Armed Bandit) Selection ✅
A state-of-the-art reinforcement learning baseline modelling the system as a Restless Multi-Armed Bandit (RMAB). Each node is an independent MDP with two actions: active (collect) or passive (accumulate). The Whittle Index is the exact "subsidy for passivity" $m$ at which the decision-maker is indifferent between the two actions. Computed via Value Iteration over a discretized state space, nodes are ranked by their indices and the top $K$ are selected, naturally balancing urgency and economic value over infinite horizons.

### Value Function Approximation (VFA) Selection ❌
An approximate dynamic programming strategy that replaces the exact Value Iteration of Whittle Index computation with a parameterized function approximator $\hat{V}_\theta(s_i)$ — for instance, a linear combination of fill ratio, accumulation rate, time since last collection, and depot distance. The parameters $\theta$ are fitted offline via temporal difference learning or regression on simulated rollout data. At deployment, each node is scored by $\hat{V}_\theta$, and nodes whose VFA score exceeds a threshold are mandated. VFA trades the exact optimality guarantee of Whittle for dramatically lower per-instance computation.

### Pareto-Front (Non-Dominated) Selection ✅
A multi-objective combinatorial strategy discarding scalar weighting in favor of strict Pareto dominance. Each node is evaluated in a two-dimensional objective space: Urgency (expected days to overflow) and Routing Efficiency (spatial distance to depot). Nodes on the non-dominated Pareto frontier are mandated, ensuring no node is excluded if it is simultaneously more urgent *and* cheaper to route than any alternative.

---

## Adaptive and Ensemble Dispatchers

### Combined (Ensemble) Selection ✅
A meta-selection architecture that evaluates multiple independent base strategies concurrently and aggregates their discrete selection outputs via logical operators:

$$S_{\text{final}} = \bigcup_{k \in K} S_k \quad \text{(Logical OR — aggressive union)}$$
$$S_{\text{final}} = \bigcap_{k \in K} S_k \quad \text{(Logical AND — conservative intersection)}$$

The OR mode is appropriate when any single strategy's signal is considered actionable; the AND mode is appropriate when unanimous agreement is required before commitment.

### Portfolio (Ensemble) Dispatcher ✅
A macro-level dispatcher that aggregates decisions from multiple *heterogeneous* strategies (e.g., CVaR, Whittle, Savings) via the same logical OR/AND aggregation as the Combined strategy. The architectural distinction from Combined Selection is the strategy composition: Portfolio Dispatcher is designed for strategies with fundamentally different information sources (risk, spatial, economic) whose outputs are complementary, whereas Combined Selection typically aggregates strategies from the same domain using varied parameterizations.

### Contextual Thompson Sampling Dispatcher ✅
A Multi-Armed Bandit meta-dispatcher that dynamically selects among candidate selection strategies (e.g., exact MIP vs. fast greedy) based on current environment state. A Bayesian Beta-Bernoulli posterior $(\alpha_k, \beta_k)$ is maintained for each strategy $k$. At each operational day, a success probability is sampled $\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$ and modulated by a temperature parameter. The strategy with the highest sampled value is dispatched. Posteriors are updated multiplicatively based on the resulting routing cost improvement over the rolling baseline, continuously re-weighting toward strategies that work best on current instance geometry and fill-level distributions.

### Entropy-Regularized Adaptive Selection ❌
A meta-dispatcher that adds an explicit exploration bonus to prevent premature strategy specialization. The selection score for each strategy $k$ at day $t$ is:

$$\text{Score}_k = \bar{R}_k - \beta_t H(\pi_k)$$

where $\bar{R}_k$ is the empirical mean reward of strategy $k$, $H(\pi_k)$ is the entropy of its selection distribution over nodes (low entropy = highly deterministic, high entropy = diverse), and $\beta_t$ is an annealing exploration coefficient. Strategies producing diverse, spread-out selections are preferred early in the search horizon (high $\beta_t$); concentrated, high-precision strategies are preferred as the horizon closes ($\beta_t \to 0$).

### Learned Imitation Selection ✅
A machine-learning heuristic that bypasses the computational bottleneck of exact MILP solvers. A pre-trained classification model (Random Forest or Neural Network) extracts a localized state vector per node: fill ratio, accumulation rate, demand variance, spatial distance, net revenue. A forward pass predicts the probability that the exact solver would select the node, and nodes exceeding a threshold probability are mandated. Unlike the VFA strategy (which approximates the value function), Learned Imitation directly imitates the exact solver's binary selection decision, making it faster but less interpretable.

---
