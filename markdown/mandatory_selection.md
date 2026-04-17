# Mandatory Bin Selection Strategies

These are domain-specific filtering algorithms utilized in multi-period combinatorial optimization (such as the Vehicle Routing Problem with Profits) to explicitly mandate which nodes must obligatorily be visited by the routing policy on a given operational day.

## Static & Reactive Baselines

### Regular Selection
A static, periodic scheduling policy that operates entirely in the time domain, independent of stochastic accumulation states. It strictly mandates collection based on a fixed frequency $X$ (e.g., every 3 days). A collection is triggered for all eligible nodes on the current operational day $t$ if:
  $$t \equiv 1 \pmod{X}$$
  While computationally trivial, it is highly inefficient in stochastic environments, serving primarily as a worst-case baseline for fixed-route comparisons.

### Last Minute Selection
A reactive, capacity-driven strategy that triggers an obligatory collection strictly when a node's current fill state exceeds a predefined critical threshold. Let $c_i$ be the current fill level of node $i$, $Q_{\max}$ be the maximum physical capacity, and $\tau \in [0, 1]$ be the trigger threshold. A node is selected if its fill ratio breaches the threshold:
  $$\frac{c_i}{Q_{\max}} \ge \tau$$
  *Implementation Note:* This acts as the baseline for avoiding catastrophic overflow, but naturally suffers from spatial myopia as it ignores neighboring nodes.

---

## Overflow Minimization & Risk Management

### Deadline-Driven Selection
A deterministic temporal strategy that projects the exact operational deadline for each node. By computing the expected days until guaranteed overflow using the floor function $d^* = \lfloor(Q_{\max} - c) / \mu\rfloor$, it strictly obligates the collection of any node whose deadline $d^*$ is less than or equal to the solver's current look-ahead horizon.

### Look-Ahead Selection
A predictive, simulation-driven strategy designed to naturally synchronize multi-day collections. It identifies the subset of critically full bins $S_{\text{crit}}$ and simulates their time-to-overflow to find the nearest required service day, $d_{\text{next}}$. It then evaluates all unassigned nodes $j \notin S_{\text{crit}}$ with accumulation rate $\mu_j$, mandating their collection if they are mathematically guaranteed to overflow before the vehicle returns to that region:
  $$c_j + \mu_j \cdot d_{\text{next}} \ge Q_{\max}$$

### Linear Service Level Prediction
A statistical confidence-bound strategy designed for strict Service Level Agreement (SLA) adherence. It projects the worst-case future fill level of a node over a defined horizon $D$ using a linear approximation of the accumulation variance:
  $$w_{\text{future}} = w_{\text{current}} + D\mu + D k\sigma$$
  If this projected upper confidence bound breaches the maximum capacity limit, the node is immediately mandated for collection.

### Stochastic Regret Selection
A statistically rigorous strategy based on Expected Overflow Regret (EOR). Let $s_i = Q_{\max} - c_i$ be the remaining capacity of node $i$. Assuming the daily accumulation follows a normal distribution $X \sim \mathcal{N}(\mu, \sigma^2)$, the regret of deferring collection is the expected value of the overflow. The strategy computes the closed-form expectation using the standard normal probability density function $\phi$ and cumulative distribution function $\Phi$, evaluating standard score $Z = \frac{s_i - \mu}{\sigma}$:
  $$\mathbb{E}[\max(0, X - s_i)] = \sigma \phi(Z) + (\mu - s_i)(1 - \Phi(Z))$$
  Collection is obligated if this mathematical expectation exceeds an acceptable risk parameter $\gamma$.

### Multi-Day Overflow Probability Selection
A stochastic look-ahead strategy that models the evolution of waste accumulation over a temporal horizon of $K$ days. Leveraging the property that variance scales linearly with time under i.i.d. assumptions, the strategy computes the dynamic standard deviation $\sigma_K = \sigma \sqrt{K}$. It mandates collection if the cumulative tail probability of the accumulation exceeding the remaining capacity, $P(\text{Accum} \ge Q_{\max} - c)$, breaches an acceptable statistical risk threshold.

### Conditional Value-at-Risk (CVaR) Selection
A risk-averse selection paradigm that evaluates the tail risk of capacity violations. Assuming the future fill level $F$ follows a Gaussian distribution $F \sim \mathcal{N}(c + \mu, \sigma^2)$, this strategy evaluates the surplus variable $X = F - Q_{\max}$. Rather than simply thresholding the expected value, it mandates collection if the expected overflow strictly in the worst $(1 - \alpha)$ fraction of outcomes—mathematically defined as $\text{CVaR}_\alpha(\max(0, X))$—exceeds a defined critical tolerance.

### Wasserstein Distributionally Robust Selection
A robust optimization paradigm that replaces point-estimate Gaussian assumptions with a Wasserstein-1 ambiguity ball of radius $\epsilon$ centered on the empirical distribution. Utilizing the duality results of Mohajerin Esfahani and Kuhn (2018), it optimizes against the worst-case probability distribution within this ball. For the specific ReLU/Max loss function characterizing overflow volume, the worst-case expectation simplifies exactly to the nominal expectation plus the radius $\epsilon$, providing a highly tractable, distribution-free risk bound.

### MIP Multiple-Knapsack Selection (Overflow-Minimizing Variant)
An exact 0/1 mixed-integer programming (MILP) multiple-knapsack formulation. Unlike profit-maximizing variants, this architecture is explicitly engineered to minimize expected overflow losses across a look-ahead horizon, evaluated via a stochastic scenario tree. It models the homogeneous fleet as $K$ knapsacks of capacity $Q$. For every bin $i$ and future scenario $s$ with probability $\pi_s$, let $o_i^{(s)} = \max(0, w_i^{(s)} - 100\%)$ be the projected overflow fraction if the bin is *not* collected today. The solver selects a binary collection vector $x_i \in \{0,1\}$ to minimize the total expected waste lost plus a fixed bin overflow penalty $P_i$:

$$
\min \sum_{i} (1 - x_i) \left[ \sum_s \pi_s \cdot o_i^{(s)} \cdot \hat{m}_i + P_i \cdot \Pr[\text{any overflow}_i] \right]
$$

To drastically reduce the branch-and-bound search space caused by fleet symmetry, the formulation injects lexicographic count constraints ($\sum_i x_{i,k} \ge \sum_i x_{i,k+1} \quad \forall k$).

---

## Profit Maximization & Economic Strategies

### Revenue Threshold Selection
A value-based economic heuristic prioritizing nodes based on the absolute monetary value of their current contents. Let $V$ be the total volumetric capacity of the node, $\rho$ be the waste density, and $R_{\text{kg}}$ be the unit revenue per kilogram. The strategy computes the expected current revenue and triggers an obligatory visit if it surpasses a profitability threshold $\tau_{\text{rev}}$:
  $$\left( \frac{c_i}{Q_{\max}} \right) V \rho R_{\text{kg}} > \tau_{\text{rev}}$$

### Profit-per-Kilometer (Spatial ROI) Selection
A routing-aware economic heuristic that moves beyond pure revenue thresholding. This strategy normalizes the expected monetary value of a node by a proxy for its marginal insertion cost. By computing the spatial Return on Investment (ROI) as $Score_i = r_i / (2 d_{0,i})$, it actively deprioritizes highly valuable but highly isolated nodes, triggering an obligatory visit only when this economic-spatial ratio strictly exceeds a configured threshold.

### Fractional Knapsack (Density-Greedy) Selection
A high-speed heuristic approximation of the multiple-knapsack formulation designed to maximize profit. It sorts unassigned nodes by their net-profit density (profit per unit mass) and greedily packs them into vehicle capacities. To ensure mathematical rigor, it guarantees a $\frac{1}{2}$-approximation of the optimal solution by returning the maximum between the greedy-packed set and the single most profitable node that fits independently.

### Lagrangian Reduced-Cost Selection
A relaxation-based strategy that evaluates the LP relaxation of the profit-maximizing multiple-knapsack problem. It extracts the dual variable (shadow price) $\lambda^*$ associated with the binding global capacity constraint. Nodes are then evaluated based on their Lagrangian reduced costs:
  $$\bar{c}_i = (r_i - \text{cost} \cdot d_i) - \lambda^* m_i$$
  Any node exhibiting a strictly positive marginal contribution ($\bar{c}_i > 0$) is flagged for obligatory collection.

---

## Spatial & Synergistic Routing Strategies

### Spatial Synergy Selection
A geometric, density-aware routing strategy that utilizes the underlying spatial graph to amortize fixed travel costs. It operates under a bipartite thresholding model. First, it isolates a set of critically full nodes $S_{\text{crit}}$ using a severe threshold $\tau_{\text{crit}}$ (e.g., $0.90$). Then, it defines a moderate synergy threshold $\tau_{\text{syn}}$ (e.g., $0.60$). A node $j$ is pulled into the obligatory set if it is moderately full *and* lies within a strict Euclidean distance radius $R$ of any critically full node $i$:
  $$S_{\text{syn}} = \left\{ j \;\middle|\; \frac{c_j}{Q_{\max}} \ge \tau_{\text{syn}} \land \exists i \in S_{\text{crit}} \text{ s.t. } d_{i,j} \le R \right\}$$
  The final selection set is the union $S_{\text{crit}} \cup S_{\text{syn}}$.

### Clarke-Wright Savings Selection
A spatial synergy filter derived from the foundational Clarke-Wright vehicle routing heuristic. It evaluates pairwise spatial savings defined as $s_{ij} = d_{0,i} + d_{0,j} - d_{i,j}$. A node is flagged for obligatory collection only if it satisfies a minimum fill requirement *and* exhibits a strictly positive spatial saving ($s_{ij} > 0$) with at least one other critically full node, ensuring isolated routing is mathematically suppressed in favor of cluster formations.

### Set-Cover (Hub) Selection
A geometric aggregation strategy that maps the required visits to a Minimum Set Cover formulation. It defines a universe $U$ of critically full nodes and seeks to select a minimum-cardinality set of "hub" nodes such that every node in $U$ is within a specified Euclidean service radius of a chosen hub. It is solved using a standard greedy heuristic, providing a theoretically guaranteed $\ln(|U|)$ approximation ratio.

### Submodular Facility Location Selection
A mathematically rigorous subset selection strategy that maximizes a submodular coverage objective. To capture the diminishing returns of routing density, it evaluates the objective:
  $$f(S) = \sum_{i \in \text{Bins}} \max\left(0, r_i - \alpha \min_{j \in S \cup \{0\}} d_{i,j}\right)$$
  The maximization is executed under a cardinality budget using the Lazy Greedy algorithm (Minoux, 1978), which leverages a max-priority queue to avoid redundant marginal gain computations while maintaining a $(1 - 1/e)$ approximation guarantee.

### Supermodular Synergy Selection
A clustering strategy that maximizes an objective function exhibiting increasing returns (supermodularity). It balances expected revenue against a lower bound approximation of the Traveling Salesperson Problem (TSP) tour length:
  $$f(S) = \sum_{i \in S} r_i - 2\alpha \sum_{i \in S} \min_{j \in S \setminus \{i\}} d(i, j)$$
  Because the marginal cost of routing decreases as the cluster grows, the algorithm employs a modified greedy approach with continuous re-evaluation of non-positive candidates, allowing localized clusters to cross the threshold of viability collectively.

---

## Multi-Objective & Advanced Look-Ahead Strategies

### Pareto-Front (Non-Dominated) Selection
A multi-objective combinatorial strategy that discards scalar weighting in favor of strict Pareto dominance. It evaluates every node in a 2D objective space: minimizing the expected days to overflow (Urgency) and minimizing the spatial distance to the depot (Routing Efficiency). The strategy mandates the collection of all nodes residing on the non-dominated Pareto frontier, ensuring no node is left behind if it is strictly more urgent and cheaper to route than the alternatives.

### One-Step Rollout (Approximate Dynamic Programming) Selection
A simulation-based look-ahead algorithm that formalizes the decision process as an optimal stopping problem. For each node, it computes the expected future reward of two immediate actions: "Collect Today" versus "Defer to Tomorrow". The "Defer" trajectory is evaluated by simulating the node's stochastic accumulation over a finite temporal horizon $H$, applying a base heuristic (e.g., Last-Minute selection) and a discount factor $\gamma$ to future actions. A node is selected if the expected discounted reward of immediate collection strictly dominates deferral.

### Whittle Index (Restless Multi-Armed Bandit) Selection
A state-of-the-art reinforcement learning baseline that models the entire system as a Restless Multi-Armed Bandit (RMAB). Each node is an independent Markov Decision Process (MDP) with two actions: active (collect) or passive (accumulate). The strategy computes the Whittle Index—the exact "subsidy for passivity" $m$ at which the decision-maker is mathematically indifferent between the two actions. Using a discretized state space and Value Iteration, nodes are ranked by their computed indices, and the top $K$ nodes are selected, naturally balancing urgency and economic value over infinite horizons.

---

## Meta-Algorithmic & Dispatcher Strategies

### Combined (Ensemble) Selection
A meta-selection architecture that concurrently evaluates multiple independent base strategies (e.g., Spatial Synergy and Revenue Threshold) and aggregates their discrete outputs using logical operators. This permits the engineering of complex, multi-objective collection triggers through set operations, computing either the conservative intersection (Logical AND) or the aggressive union (Logical OR) of the generated routing requirements:
  $$S_{\text{final}} = \bigcup_{k \in K} S_k \quad \text{or} \quad S_{\text{final}} = \bigcap_{k \in K} S_k$$

### Portfolio (Ensemble) Dispatcher
A macro-level dispatcher designed to aggregate the decisions of multiple, highly heterogeneous independent selection strategies (e.g., CVaR, Whittle, Savings) to form a robust consensus. Like the Combined strategy, the dispatcher can be configured mathematically as either a conservative logical intersection requiring total agreement, or an aggressive logical union requiring only a single strategy's nomination.

### Contextual Thompson Sampling Dispatcher
A meta-selection architecture modeled as a Multi-Armed Bandit (MAB). Instead of statically assigning a selection strategy, it dynamically chooses the optimal strategy (e.g., exact MIP vs. fast greedy) per environment state. It maintains a Bayesian Beta-Bernoulli posterior for each candidate strategy, sampling a success probability $\theta \sim \text{Beta}(\alpha, \beta)$ modulated by a temperature parameter. Strategies that consistently yield lower operational routing costs are sampled and dispatched with higher probability over successive epochs.

### Learned Imitation Selection
A machine-learning-driven heuristic that bypasses the computational bottleneck of exact MILP solvers. Utilizing a pre-trained classification model (e.g., Random Forest or Neural Network), it extracts a localized state vector for each node (fill ratio, accumulation rate, variance, spatial distance, and net revenue). The model conducts a forward pass to predict the continuous probability of the exact solver selecting the node, thresholding the output to trigger an obligatory visit.

---
