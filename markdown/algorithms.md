# Mandatory Bin Selection Strategies

These are domain-specific filtering algorithms utilized in multi-period combinatorial optimization (such as the Vehicle Routing Problem with Profits) to explicitly mandate which nodes must obligatorily be visited by the routing policy on a given operational day.

### 1. Static & Reactive Baselines

- **Regular Selection:** A static, periodic scheduling policy that operates entirely in the time domain, independent of stochastic accumulation states. It strictly mandates collection based on a fixed frequency $X$ (e.g., every 3 days). A collection is triggered for all eligible nodes on the current operational day $t$ if: $$t \equiv 1 \pmod{X}$$ While computationally trivial, it is highly inefficient in stochastic environments, serving primarily as a worst-case baseline for fixed-route comparisons.

- **Last Minute Selection:** A reactive, capacity-driven strategy that triggers an obligatory collection strictly when a node’s current fill state exceeds a predefined critical threshold. Let $c_i$ be the current fill level of node $i$, $Q_{\max}$ be the maximum physical capacity, and $\tau \in [0, 1]$ be the trigger threshold. A node is selected if its fill ratio breaches the threshold: $$\frac{c_i}{Q_{\max}} \ge \tau$$ *Implementation Note:* This acts as the baseline for avoiding catastrophic overflow, but naturally suffers from spatial myopia as it ignores neighboring nodes.

### 2. Overflow Minimization & Risk Management

- **Deadline-Driven Selection:** A deterministic temporal strategy that projects the exact operational deadline for each node. By computing the expected days until guaranteed overflow using the floor function $d^* = \lfloor(Q_{\max} - c) / \mu\rfloor$, it strictly obligates the collection of any node whose deadline $d^*$ is less than or equal to the solver’s current look-ahead horizon.

- **Look-Ahead Selection:** A predictive, simulation-driven strategy designed to naturally synchronize multi-day collections. It identifies the subset of critically full bins $S_{\text{crit}}$ and simulates their time-to-overflow to find the nearest required service day, $d_{\text{next}}$. It then evaluates all unassigned nodes $j \notin S_{\text{crit}}$ with accumulation rate $\mu_j$, mandating their collection if they are mathematically guaranteed to overflow before the vehicle returns to that region: $$c_j + \mu_j \cdot d_{\text{next}} \ge Q_{\max}$$

- **Linear Service Level Prediction:** A statistical confidence-bound strategy designed for strict Service Level Agreement (SLA) adherence. It projects the worst-case future fill level of a node over a defined horizon $D$ using a linear approximation of the accumulation variance: $$w_{\text{future}} = w_{\text{current}} + D\mu + D k\sigma$$ If this projected upper confidence bound breaches the maximum capacity limit, the node is immediately mandated for collection.

- **Stochastic Regret Selection:** A statistically rigorous strategy based on Expected Overflow Regret (EOR). Let $s_i = Q_{\max} - c_i$ be the remaining capacity of node $i$. Assuming the daily accumulation follows a normal distribution $X \sim \mathcal{N}(\mu, \sigma^2)$, the regret of deferring collection is the expected value of the overflow. The strategy computes the closed-form expectation using the standard normal probability density function $\phi$ and cumulative distribution function $\Phi$, evaluating standard score $Z = \frac{s_i - \mu}{\sigma}$: $$\mathbb{E}[\max(0, X - s_i)] = \sigma \phi(Z) + (\mu - s_i)(1 - \Phi(Z))$$ Collection is obligated if this mathematical expectation exceeds an acceptable risk parameter $\gamma$.

- **Multi-Day Overflow Probability Selection:** A stochastic look-ahead strategy that models the evolution of waste accumulation over a temporal horizon of $K$ days. Leveraging the property that variance scales linearly with time under i.i.d. assumptions, the strategy computes the dynamic standard deviation $\sigma_K = \sigma \sqrt{K}$. It mandates collection if the cumulative tail probability of the accumulation exceeding the remaining capacity, $P(\text{Accum} \ge Q_{\max} - c)$, breaches an acceptable statistical risk threshold.

- **Conditional Value-at-Risk (CVaR) Selection:** A risk-averse selection paradigm that evaluates the tail risk of capacity violations. Assuming the future fill level $F$ follows a Gaussian distribution $F \sim \mathcal{N}(c + \mu, \sigma^2)$, this strategy evaluates the surplus variable $X = F - Q_{\max}$. Rather than simply thresholding the expected value, it mandates collection if the expected overflow strictly in the worst $(1 - \alpha)$ fraction of outcomes—mathematically defined as $\text{CVaR}_\alpha(\max(0, X))$—exceeds a defined critical tolerance.

- **Wasserstein Distributionally Robust Selection:** A robust optimization paradigm that replaces point-estimate Gaussian assumptions with a Wasserstein-1 ambiguity ball of radius $\epsilon$ centered on the empirical distribution. Utilizing the duality results of Mohajerin Esfahani and Kuhn (2018), it optimizes against the worst-case probability distribution within this ball. For the specific ReLU/Max loss function characterizing overflow volume, the worst-case expectation simplifies exactly to the nominal expectation plus the radius $\epsilon$, providing a highly tractable, distribution-free risk bound.

- **MIP Multiple-Knapsack Selection (Overflow-Minimizing Variant):** An exact 0/1 mixed-integer programming (MILP) multiple-knapsack formulation. Unlike profit-maximizing variants, this architecture is explicitly engineered to minimize expected overflow losses across a look-ahead horizon, evaluated via a stochastic scenario tree. It models the homogeneous fleet as $K$ knapsacks of capacity $Q$. For every bin $i$ and future scenario $s$ with probability $\pi_s$, let $o_i^{(s)} = \max(0, w_i^{(s)} - 100\%)$ be the projected overflow fraction if the bin is *not* collected today. The solver selects a binary collection vector $x_i \in \{0,1\}$ to minimize the total expected waste lost plus a fixed overflow occurrence penalty $P_i$: $$\min \sum_{i} (1 - x_i) \left[ \sum_s \pi_s \cdot o_i^{(s)} \cdot \hat{m}_i + P_i \cdot \Pr[\text{any overflow}_i] \right]$$ To drastically reduce the branch-and-bound search space caused by fleet symmetry, the formulation injects lexicographic count constraints ($\sum_i x_{i,k} \ge \sum_i x_{i,k+1} \quad \forall k$).

### 3. Profit Maximization & Economic Strategies

- **Revenue Threshold Selection:** A value-based economic heuristic prioritizing nodes based on the absolute monetary value of their current contents. Let $V$ be the total volumetric capacity of the node, $\rho$ be the waste density, and $R_{\text{kg}}$ be the unit revenue per kilogram. The strategy computes the expected current revenue and triggers an obligatory visit if it surpasses a profitability threshold $\tau_{\text{rev}}$: $$\left( \frac{c_i}{Q_{\max}} \right) V \rho R_{\text{kg}} > \tau_{\text{rev}}$$

- **Profit-per-Kilometer (Spatial ROI) Selection:** A routing-aware economic heuristic that moves beyond pure revenue thresholding. This strategy normalizes the expected monetary value of a node by a proxy for its marginal insertion cost. By computing the spatial Return on Investment (ROI) as $Score_i = r_i / (2 d_{0,i})$, it actively deprioritizes highly valuable but highly isolated nodes, triggering an obligatory visit only when this economic-spatial ratio strictly exceeds a configured threshold.

- **Fractional Knapsack (Density-Greedy) Selection:** A high-speed heuristic approximation of the multiple-knapsack formulation designed to maximize profit. It sorts unassigned nodes by their net-profit density (profit per unit mass) and greedily packs them into vehicle capacities. To ensure mathematical rigor, it guarantees a $\frac{1}{2}$-approximation of the optimal solution by returning the maximum between the greedy-packed set and the single most profitable node that fits independently.

- **Lagrangian Reduced-Cost Selection:** A relaxation-based strategy that evaluates the LP relaxation of the profit-maximizing multiple-knapsack problem. It extracts the dual variable (shadow price) $\lambda^*$ associated with the binding global capacity constraint. Nodes are then evaluated based on their Lagrangian reduced costs: $$\bar{c}_i = (r_i - \text{cost} \cdot d_i) - \lambda^* m_i$$ Any node exhibiting a strictly positive marginal contribution ($\bar{c}_i > 0$) is flagged for obligatory collection.

### 4. Spatial & Synergistic Routing Strategies

- **Spatial Synergy Selection:** A geometric, density-aware routing strategy that utilizes the underlying spatial graph to amortize fixed travel costs. It operates under a bipartite thresholding model. First, it isolates a set of critically full nodes $S_{\text{crit}}$ using a severe threshold $\tau_{\text{crit}}$ (e.g., $0.90$). Then, it defines a moderate synergy threshold $\tau_{\text{syn}}$ (e.g., $0.60$). A node $j$ is pulled into the obligatory set if it is moderately full *and* lies within a strict Euclidean distance radius $R$ of any critically full node $i$: $$S_{\text{syn}} = \left\{ j \;\middle|\; \frac{c_j}{Q_{\max}} \ge \tau_{\text{syn}} \land \exists i \in S_{\text{crit}} \text{ s.t. } d_{i,j} \le R \right\}$$ The final selection set is the union $S_{\text{crit}} \cup S_{\text{syn}}$.

- **Clarke-Wright Savings Selection:** A spatial synergy filter derived from the foundational Clarke-Wright vehicle routing heuristic. It evaluates pairwise spatial savings defined as $s_{ij} = d_{0,i} + d_{0,j} - d_{i,j}$. A node is flagged for obligatory collection only if it satisfies a minimum fill requirement *and* exhibits a strictly positive spatial saving ($s_{ij} > 0$) with at least one other critically full node, ensuring isolated routing is mathematically suppressed in favor of cluster formations.

- **Set-Cover (Hub) Selection:** A geometric aggregation strategy that maps the required visits to a Minimum Set Cover formulation. It defines a universe $U$ of critically full nodes and seeks to select a minimum-cardinality set of “hub” nodes such that every node in $U$ is within a specified Euclidean service radius of a chosen hub. It is solved using a standard greedy heuristic, providing a theoretically guaranteed $\ln(|U|)$ approximation ratio.

- **Submodular Facility Location Selection:** A mathematically rigorous subset selection strategy that maximizes a submodular coverage objective. To capture the diminishing returns of routing density, it evaluates the objective: $$f(S) = \sum_{i \in \text{Bins}} \max\left(0, r_i - \alpha \min_{j \in S \cup \{0\}} d_{i,j}\right)$$ The maximization is executed under a cardinality budget using the Lazy Greedy algorithm (Minoux, 1978), which leverages a max-priority queue to avoid redundant marginal gain computations while maintaining a $(1 - 1/e)$ approximation guarantee.

- **Supermodular Synergy Selection:** A clustering strategy that maximizes an objective function exhibiting increasing returns (supermodularity). It balances expected revenue against a lower bound approximation of the Traveling Salesperson Problem (TSP) tour length: $$f(S) = \sum_{i \in S} r_i - 2\alpha \sum_{i \in S} \min_{j \in S \setminus \{i\}} d(i, j)$$ Because the marginal cost of routing decreases as the cluster grows, the algorithm employs a modified greedy approach with continuous re-evaluation of non-positive candidates, allowing localized clusters to cross the threshold of viability collectively.

### 5. Multi-Objective & Advanced Look-Ahead Strategies

- **Pareto-Front (Non-Dominated) Selection:** A multi-objective combinatorial strategy that discards scalar weighting in favor of strict Pareto dominance. It evaluates every node in a 2D objective space: minimizing the expected days to overflow (Urgency) and minimizing the spatial distance to the depot (Routing Efficiency). The strategy mandates the collection of all nodes residing on the non-dominated Pareto frontier, ensuring no node is left behind if it is strictly more urgent and cheaper to route than the alternatives.

- **One-Step Rollout (Approximate Dynamic Programming) Selection:** A simulation-based look-ahead algorithm that formalizes the decision process as an optimal stopping problem. For each node, it computes the expected future reward of two immediate actions: “Collect Today” versus “Defer to Tomorrow”. The “Defer” trajectory is evaluated by simulating the node’s stochastic accumulation over a finite temporal horizon $H$, applying a base heuristic (e.g., Last-Minute selection) and a discount factor $\gamma$ to future actions. A node is selected if the expected discounted reward of immediate collection strictly dominates deferral.

- **Whittle Index (Restless Multi-Armed Bandit) Selection:** A state-of-the-art reinforcement learning baseline that models the entire system as a Restless Multi-Armed Bandit (RMAB). Each node is an independent Markov Decision Process (MDP) with two actions: active (collect) or passive (accumulate). The strategy computes the Whittle Index—the exact “subsidy for passivity” $m$ at which the decision-maker is mathematically indifferent between the two actions. Using a discretized state space and Value Iteration, nodes are ranked by their computed indices, and the top $K$ nodes are selected, naturally balancing urgency and economic value over infinite horizons.

### 6. Meta-Algorithmic & Dispatcher Strategies

- **Combined (Ensemble) Selection:** A meta-selection architecture that concurrently evaluates multiple independent base strategies (e.g., Spatial Synergy and Revenue Threshold) and aggregates their discrete outputs using logical operators. This permits the engineering of complex, multi-objective collection triggers through set operations, computing either the conservative intersection (Logical AND) or the aggressive union (Logical OR) of the generated routing requirements: $$S_{\text{final}} = \bigcup_{k \in K} S_k \quad \text{or} \quad S_{\text{final}} = \bigcap_{k \in K} S_k$$

- **Portfolio (Ensemble) Dispatcher:** A macro-level dispatcher designed to aggregate the decisions of multiple, highly heterogeneous independent selection strategies (e.g., CVaR, Whittle, Savings) to form a robust consensus. Like the Combined strategy, the dispatcher can be configured mathematically as either a conservative logical intersection requiring total agreement, or an aggressive logical union requiring only a single strategy’s nomination.

- **Contextual Thompson Sampling Dispatcher:** A meta-selection architecture modeled as a Multi-Armed Bandit (MAB). Instead of statically assigning a selection strategy, it dynamically chooses the optimal strategy (e.g., exact MIP vs. fast greedy) per environment state. It maintains a Bayesian Beta-Bernoulli posterior for each candidate strategy, sampling a success probability $\theta \sim \text{Beta}(\alpha, \beta)$ modulated by a temperature parameter. Strategies that consistently yield lower operational routing costs are sampled and dispatched with higher probability over successive epochs.

- **Learned Imitation Selection:** A machine-learning-driven heuristic that bypasses the computational bottleneck of exact MILP solvers. Utilizing a pre-trained classification model (e.g., Random Forest or Neural Network), it extracts a localized state vector for each node (fill ratio, accumulation rate, variance, spatial distance, and net revenue). The model conducts a forward pass to predict the continuous probability of the exact solver selecting the node, thresholding the output to trigger an obligatory visit.

# Route Construction Algorithms

## Exact Stochastic and Decomposition Solvers

While meta-heuristics and neural models offer computational efficiency, exact solvers are the gold standard for establishing optimal baselines and providing theoretical guarantees. This work utilizes two primary exact frameworks for benchmarking and solving the state-space challenges of multi-period routing.

- **Integer L-shaped Benders Decomposition (ILS-BD):** An exact algorithmic framework extending classical Benders decomposition, engineered for Stochastic Mixed-Integer Linear Programs (SMILP) where second-stage (recourse) decisions involve discrete variables. Because standard continuous LP duality cannot formulate Benders cuts for integer subproblems, this architecture dynamically generates integer optimality cuts (Laporte and Louveaux, 1993) at integer-feasible nodes of the master tree. Let $S^k$ be the index set of binary master variables $x_i$ taking value 1 at iteration $k$, and $L$ be a valid lower bound on the expected recourse cost $Q(x)$. The global optimality cut isolates scenario-dependent routing: $$\theta \ge (Q(x^k) - L) \left( \sum_{i \in S^k} x_i - \sum_{i \notin S^k} x_i - |S^k| + 1 \right) + L$$

- **Logic-Based Benders Decomposition (LBBD):** A generalization of the Benders method that removes the requirement for the subproblem to be a continuous linear program. By modeling routing subproblems via alternative paradigms—such as Constraint Programming (CP)—LBBD derives exact bounding cuts through logical inference rather than extreme rays of a dual polyhedron. Let $f(x, y)$ be the combinatorial subproblem cost. The master problem utilizes a bounding function $B_k(x)$ inferred from the CP solver’s proof of optimality or infeasibility, adding cuts of the form: $$\theta \ge B_k(x) \quad \text{where} \quad B_k(x^k) = \min_{y \in Y(x^k)} f(x^k, y)$$

- **Exact Stochastic Dynamic Programming (ESDP):** A rigorous mathematical framework for solving multi-stage stochastic routing challenges by evaluating Bellman’s principle of optimality backward through the temporal horizon. By exhaustively evaluating the exact state-space transitions, ESDP computes the optimal policy for every system state $S_t$. The value function $V_t(S_t)$ is solved recursively: $$V_t(S_t) = \min_{x_t \in \mathcal{X}(S_t)} \left\{ C(S_t, x_t) + \gamma \mathbb{E}_{\omega_t} \left[ V_{t+1}\big(\mathcal{T}(S_t, x_t, \omega_t)\big) \right] \right\}$$ While providing a flawless theoretical baseline, its practical execution is bottlenecked by the curse of dimensionality.

- **Progressive Hedging (PH):** A scenario-based decomposition strategy that tackles complexity by temporarily relaxing non-anticipativity constraints. It optimizes each probabilistic scenario $s \in \mathcal{S}$ independently and applies an augmented Lagrangian penalty to penalize deviations from a consensus policy $z$. At iteration $k$, the scenario subproblem minimizes: $$\min_{x_s} \left\{ p_s f_s(x_s) + w_s^{(k)T} x_s + \frac{\rho}{2} \| x_s - z^{(k-1)} \|^2 \right\}$$ The consensus $z^{(k)} = \sum_{s \in \mathcal{S}} p_s x_s^{(k)}$ and multipliers $w_s^{(k+1)} = w_s^{(k)} + \rho(x_s^{(k)} - z^{(k)})$ are iteratively updated until convergence.

- **Scenario Tree Extensive Form (ST-EF):** The foundational baseline approach for solving finite stochastic routing models. It constructs the full probabilistic scenario tree and formulates the Deterministic Equivalent Problem (DEP) as a single, monolithic MILP: $$\min_{x, y_s} \sum_{s \in \mathcal{S}} p_s \left( c^T x + q_s^T y_s \right)$$ $$\text{s.t.} \quad A x \le b, \quad T_s x + W_s y_s \le h_s \quad \forall s \in \mathcal{S}$$ While circumventing algorithmic decomposition, its memory footprint scales exponentially.

- **Constraint Programming with Boolean Satisfiability (CP-SAT):** An inference architecture integrating finite-domain CP with modern Boolean Satisfiability (SAT) engines. Through lazy clause generation, complex spatial constraints and vehicle capacities are translated into SAT clauses dynamically. For instance, if routing node $i$ to $j$ under capacity violations causes conflict, a nogood clause $\neg(x_i = v_1 \land x_j = v_2)$ is injected into the SAT solver, frequently outperforming classical continuous linear relaxations in highly constrained spaces.

- **Branch-and-Bound (BB):** The foundational search algorithm for exact MILP optimization. BB systematically partitions the feasible decision space into smaller subproblems (branching) and computes continuous LP relaxation lower bounds $\text{LB} \le Z^*$ at each node. If a node’s lower bound strictly exceeds the cost of the best-known integer incumbent solution ($\text{LB} \ge \text{UB}$), or if the subproblem is mathematically infeasible, the subtree is safely pruned. This implicitly exhausts the search space, guaranteeing global optimality without explicit enumeration.

- **Branch-and-Cut (BC):** An exact algorithm that enhances the classical Branch-and-Bound framework by embedding cutting plane methods directly into the search tree. At each node, the solver evaluates the fractional LP solution $\tilde{x}$. If $\tilde{x} \notin \text{conv}(\mathcal{X})$, the algorithm dynamically generates valid inequalities (cutting planes) $\alpha^T x \le \beta$ that strictly separate $\tilde{x}$ from the true integer polyhedron. This aggressively tightens the continuous relaxations and significantly reduces the exponential size of the search tree.

- **Branch-and-Price (BP):** A specialized exact algorithm integrating Branch-and-Bound with column generation, primarily deployed for Set Partitioning formulations featuring an exponentially large variable space. Instead of enumerating all possible routing variables, the Restricted Master Problem (RMP) is solved using a limited subset of columns. A pricing subproblem—typically modeled as an Elementary Shortest Path Problem with Resource Constraints (ESPPRC)—is then solved to dynamically generate new route columns $j$ exhibiting negative reduced costs: $$\bar{c}_j = c_j - \sum_{i \in V} \pi_i a_{ij} - \mu < 0$$ where $\pi_i$ represent the dual prices of the node-covering constraints.

- **Branch-and-Price-and-Cut (BPC):** The state-of-the-art exact framework for integer multicommodity flow and routing problems. BPC unifies column generation and cutting planes within a branch-and-bound tree. To combat severe problem symmetry—where standard fractional branching yields negligible bound improvements—BPC dynamically separates advanced valid inequalities, such as Subset-Row Cuts (SRCs) or lifted cover inequalities derived from 0-1 knapsack constraints. The pricing subproblem is systematically modified to incorporate the dual variables of these generated cuts, ensuring the linear programming relaxation remains mathematically tight.

- **Smart Waste Collection - Two-Commodity Flow (SWC-TCF):** An MILP formulation adapted from the exact two-commodity network flow model (Baldacci et al., 2004). It utilizes continuous flow variables $y_{ij}$ (waste load) and $y'_{ij}$ (empty vehicle capacity) to replace exponential subtour elimination constraints. For vehicle capacity $Q$ and binary routing variables $x_{ij}$, the flow conservation guarantees valid routes: $$y_{ij} + y'_{ij} = Q x_{ij} \quad \forall (i,j) \in A$$ $$\sum_{j \in V} y_{ji} - \sum_{j \in V} y_{ij} = q_i \quad \forall i \in V$$ This dictates the optimal subset of bins and sequences to maximize $\sum_i (R \cdot q_i) - C \sum_{i,j} d_{ij} x_{ij}$, subject to multi-period SLA thresholding.

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  \## Matheuristics
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  \## Meta-Heuristics

  These are high-level frameworks used to find or select a heuristic that may provide a sufficiently good solution to an optimization problem.

  \- **Adaptive Large Neighborhood Search (ALNS)**: A domain-specific meta-heuristic utilizing ruin and recreate logic.

  \- **Artificial Bee Colony (ABC)**: A relabeled swarm-based meta-heuristic.

  \- **Evolution Strategy Mu Plus Lambda (ES-MPL)**: A canonical evolutionary meta-heuristic.

  \- **Evolution Strategy Mu, Kappa, Lambda (ES-MKL)**: A foundational evolutionary meta-heuristic introducing lifespan bounds.

  \- **Evolution Strategy Mu, Lambda (ES-MCL)**: A foundational evolutionary meta-heuristic focused on offspring selection.

  \- **Differential Evolution (DE)**: A canonical population-based meta-heuristic for continuous optimization.

  \- **Fast Iterative Localized Optimization (FILO)**: A domain-specific routing heuristic based on spatial granularity.

  \- **Firefly Algorithm (FA)**: A relabeled swarm meta-heuristic.

  \- **Genetic Algorithm (GA)**: A canonical evolutionary meta-heuristic.

  \- **GENIUS**: A deterministic local search meta-heuristic utilizing generalized insertion (GENI) and unstringing operators to evaluate non-adjacent vertex insertions via segment reversals.

  \- **Guided Local Search (GLS)**: A penalty-based canonical meta-heuristic.

  \- **Harmony Search (HS)**: A relabeled meta-heuristic isomorphic to Evolution Strategies.

  \- **Hybrid Genetic Search (HGS)**: A memetic algorithm combining GA with local search.

  \- **Hybrid Genetic Search with Ruin and Recreate (HGS-RR)**: A memetic algorithm intersecting population paradigms with trajectory operators.

  \- **Hybrid Volleyball Premier League (HVPL)**: A memetic algorithm combining swarm and trajectory principles.

  \- **Iterated Local Search (ILS)**: A canonical trajectory-based meta-heuristic.

  \- **K-Sparse Ant Colony Optimization (KS-ACO)**: An algorithmic regularization of the canonical ACO swarm meta-heuristic.

  \- **Knowledge Guided Local Search (KGLS)**: A domain-specific execution of Guided Local Search.

  \- **League Championship Algorithm (LCA)**: A relabeled evolutionary meta-heuristic.

  \- **Memetic Algorithm (MA)**: A general paradigm combining population search with local refinement.

  \- **Particle Swarm Optimization (PSO)**: A foundational continuous swarm archetype.

  \- **Particle Swarm Optimization Memetic Algorithm (PSOMA)**: A hybrid meta-heuristic combining swarm and local trajectory search.

  \- **Quantum Differential Evolution (QDE)**: A population-based meta-heuristic using chaotic distributions.

  \- **Reactive Tabu Search (RTS)**: A canonical memory-based meta-heuristic.

  \- **Simulated Annealing (SA)**: A canonical meta-heuristic based on statistical mechanics.

  \- **Simulated Annealing Neighborhood Search (SANS)**: A hybrid meta-heuristic combining SA with multiple neighborhood structures.

  \- **Sine Cosine Algorithm (SCA)**: A relabeled swarm meta-heuristic.

  \- **Slack Induction by String Removal (SISR)**: A domain-specific spatial meta-heuristic.

  \- **Soccer League Competition (SLC)**: A relabeled evolutionary meta-heuristic.

  \- **Tabu Search (TS):** A foundational trajectory-based meta-heuristic employing short-term memory to escape local optima.

  \- **Variable Neighborhood Search (VNS)**: A canonical meta-heuristic for trajectory diversification.

  \- **Volleyball Premier League (VPL)**: A relabeled evolutionary meta-heuristic.
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Hyper-Heuristics

Hyper-heuristics operate at a higher level of abstraction than classical meta-heuristics. Rather than searching the solution space directly, they search the space of heuristics, intelligently managing a portfolio of Low-Level Heuristics (LLHs) to adapt to the problem’s topological landscape during execution.

- **Hidden Markov Model + Great Deluge Hyper-Heuristic (HMM-GD-HH):** A state-of-the-art online-learning selection hyper-heuristic modeled as an Input-Output Hidden Markov Model (IOHMM). It treats the underlying search space topology as a set of hidden states $S$ (e.g., improving, stagnating, escaping) that expand dynamically at a rate of $\mathcal{O}(\sqrt{\log t})$. The model takes LLHs as inputs $u_t$ and maps multi-objective solution changes ($\Delta$ Profit, $\Delta$ Cost) to a discrete observation alphabet $o_t$.

  1.  **Action Selection:** At iteration $t$, the controller selects the LLH $u^*$ that maximizes a dynamic balance between expected normalized profit $\bar{P}(u)$ and the transition entropy $H$ (to enforce exploration), annealed by $\alpha_t$: $$u^* = \arg\max_u \left[ \bar{P}(u) - \alpha_t \sum_{s'} P(s' | u) \ln(P(s' | u) + \epsilon) \right]$$
  2.  **Belief Update:** The hidden state belief vector is updated using a Stochastic Online Expectation-Maximization (EM) approximation of the Forward algorithm: $$b_t(j) \propto B_{u_t, j, o_t} \sum_i b_{t-1}(i) A_{u_t, i, j}$$
  3.  **Acceptance:** Independent of the HMM’s learning phase, candidate solutions are strictly filtered through a Great Deluge criterion adapted for maximization, where the water level $W_t$ monotonically rises: $$f(s') \ge W_t \quad \text{where} \quad W_{t+1} = W_t + \lambda |f(s^*)|$$

- **Genetic Programming Hyper-Heuristic (GPHH):** A generative hyper-heuristic that evolves a mathematical scoring function for constructive routing (heuristic generation).

  - Represented as a GP expression tree, the function evaluates candidate insertions using continuous local tactical features such as `node_profit`, `distance_to_route`, `insertion_cost`, and `remaining_capacity`.
  - To resolve the exponential computational bottleneck of full constructive evaluation, it employs a K-Nearest Neighbors (K-NN) candidate list to restrict evaluations to the spatial endpoints of existing routes, mathematically reducing the time complexity from $\mathcal{O}(N^2 R)$ to $\mathcal{O}(N K R)$.
  - Fitness is rigorously evaluated via true generalization—calculating the average normalized profit across structurally distinct spatial training environments, heavily penalized by a parsimony coefficient to mitigate tree bloat.
  - The evolutionary process employs mutually exclusive deep genetic operators (80% subtree crossover, 20% point mutation or Ephemeral Random Constant perturbation) governed by strict Koza-style depth limits.

- **Guided Indicators Hyper-Heuristic (GIHH):** An adaptive selection hyper-heuristic utilizing Episodic Weight Updates via Roulette Wheel selection to manage a Multi-Objective Pareto Archive (ARCH).

  - Operator selection weights are dynamically guided by two primary performance indicators: **ScoreA** (Quality Reward), which tracks the frequency an operator’s offspring successfully enters the non-dominated Pareto front, and **ScoreB** (Directional Reward), which tracks objective-space bias (e.g., favoring revenue maximization versus cost minimization).
  - The algorithm incorporates a rigorous theoretical correction for multi-objective balancing, resolving the “Equation 25 Paradox” from Chen et al. (2018).
  - It triggers directional weight updates strictly when an operator’s bias is opposite to the current segment’s dominant deviation, aggressively enforcing equilibrium across the Pareto frontier.
  - Additionally, structurally identical clones are explicitly rejected during archive evaluation to prevent archive explosion vulnerabilities.

- **Hidden Markov Model + Great Deluge Hyper-Heuristic (HMM-GD-HH):** An online-learning selection hyper-heuristic that mathematically treats the sequence of applied Low-Level Heuristics (LLHs) as a Markov chain.

  - Rather than assigning a single deterministic state, the system maintains a probability belief distribution over three hidden states (improving, stagnating, escaping) using the Forward Algorithm.
  - At each search iteration, the belief vector $P(S_t | O_{1:t})$ is updated via the observation likelihood of the normalized profit change $\Delta_{\text{norm}}$.
  - The LLH selection probabilities are then dynamically computed as the belief-weighted mixture of the per-state emission matrices.
  - Independent of the HMM’s learning phase, candidate solutions are strictly filtered through a Great Deluge acceptance criterion, which deterministically accepts candidate solutions whose profit exceeds a linearly rising water level, eliminating the need for temperature parameter tuning.

- **Reinforcement Learning - Great Deluge Hyper-Heuristic (RL-GD-HH):** An adaptive parameter controller that integrates a utility-based Reinforcement Learning mechanism with a Great Deluge (GD) acceptance criterion, strictly mapping to the architecture of Ozcan et al. (2010).

  - The algorithm maintains a scalar utility $u_i$ bounded by an upper limit (e.g., $U_{\max} = 40$) for each heuristic, consistently selecting the operator with the highest utility while employing random tie-breaking.
  - It applies an additive reward ($u_i \leftarrow \min(U_{\max}, u_i + r)$) strictly upon strict objective improvement.
  - Neutral or worsening moves are explicitly punished using configurable adaptation variants: subtractive (RL1: $u \leftarrow \max(lb, u - p)$), divisional (RL2: $u \leftarrow \lfloor u/2 \rfloor$), or root (RL3: $u \leftarrow \lfloor\sqrt{u}\rfloor$).
  - Worsening candidate solutions are gated by the Great Deluge water level, which linearly updates from the initial solution quality $f_0$ toward a predefined quality lower-bound over the exact search budget.

- **Sequence-based Selection Hyper-Heuristic (SS-HH):** An online-learning hyper-heuristic based on Markov chain principles that constructs and evaluates variable-length *sequences* of Low-Level Heuristics (LLHs) prior to application. Rigorously modeled after Kheiri (2014), the selection engine is driven by two dynamically updated matrices: a Transition Matrix (`TMatrix`) encoding the success probability of executing heuristic $j$ immediately after heuristic $i$, and an Acceptance-Strategy Matrix (`ASMatrix`) dictating whether to extend the current heuristic sequence ($AS=0$) or terminate and apply it to the solution ($AS=1$). Sequential transitions are sampled via roulette-wheel selection. Upon yielding a new global best solution, all involved matrix entries receive a continuous proportional reward scaled by the normalized objective improvement ($\Delta_{\text{norm}}$). Finally, candidate sequences are evaluated through a time-decaying threshold acceptance criterion, allowing temporary objective deterioration within mathematically bound limits.

- **Hyper-Heuristic Ant Colony Optimization (HH-ACO):** A swarm-based meta-algorithmic controller operating strictly within the heuristic space. Instead of traversing spatial customer graphs, artificial ants construct fixed-length sequences of local search and ruin-recreate operators (e.g., 2-opt, relocate, shaw removal). The transition probability from operator $i$ to operator $j$ is governed by the Ant Colony System (ACS) pseudo-random proportional rule, mathematically balancing aggressive exploitation (via a $q_0$ probability threshold selecting $\arg\max [\tau_{ij}^\alpha \eta_{ij}^\beta]$) and proportional exploration (via roulette-wheel sampling). The underlying pheromone matrix $\tau_{ij}$ is bounded by MAX-MIN Ant System (MMAS) constraints ($\tau_{\min}$, $\tau_{\max}$) and incorporates an evaporation rate $\rho$ to continuously reinforce highly synergistic sequences of algorithmic operators.

- **Hyper-heuristic with Unstringing/Stringing and Local search K-opt (HULK):** An advanced, domain-specific operator management architecture tailored for dynamic routing landscapes, mapping directly to the framework proposed by Müller and Bonilha (2022). The controller governs a tripartite pool of operators: Unstringing (generalized destruction), Stringing (generalized greedy/regret reconstruction), and Local Search (intra/inter-route $k$-opt and swaps). Operator selection is handled by an $\epsilon$-greedy Adaptive Operator Selector that tracks the running historical success (average objective score and application frequency) of each operator within a sliding memory window. The selection weights are continuously recalibrated using a dual-decay learning rate ($\text{weight} \leftarrow (1 - \text{lr}) \cdot \text{weight} \cdot \text{decay} + \text{lr} \cdot \text{avg\_score}$) to ensure the algorithm rapidly exploits high-performing operators without permanent stagnation. Acceptance of candidate moves is strictly governed by a Simulated Annealing thermodynamic criterion, which computes the transition probability $P = \exp(\Delta f / T)$ as the artificial temperature $T$ cools over successive epochs.

------------------------------------------------------------------------

# Acceptance Criteria

In trajectory-based meta-heuristics, the acceptance criterion defines the mathematical rule determining whether the search transitions from a current solution $s$ to a generated candidate solution $s'$. Assuming an objective function $f(\cdot)$ subject to minimization, the following rigorous acceptance rules are formally defined:

## Strict & Elitist Criteria

- **Only Improving (OI):** The strictest form of greedy, elitist move acceptance. The search trajectory is monotonically non-increasing, accepting a candidate only if it yields a strict objective reduction: $f(s') < f(s)$.
- **Improving and Equal (IE):** A weakly elitist strategy that accepts solutions of equal or superior quality ($f(s') \le f(s)$). Unlike OI, this allows the algorithm to perform random walks across neutral plateaus (valleys) in the objective landscape, preventing premature stagnation in flat regions.
- **Aspiration Criterion (AC):** Primarily utilized in Tabu Search architectures. It serves as an override mechanism that accepts a mathematically forbidden (tabu) candidate if its objective value strictly surpasses the global best-known solution: $f(s') < f(s^*)$.
- **All Moves Accepted (AMA):** A naive baseline or extreme diversification mechanism (equivalent to a pure random walk) where every generated candidate is deterministically accepted regardless of objective deterioration: $P(\text{accept } s') = 1$.

## Thermodynamic & Stochastic Criteria

- **Boltzmann / Metropolis Criterion (BMC):** A stochastic thermodynamic criterion originally derived from statistical mechanics. Improving moves ($\Delta f = f(s') - f(s) \le 0$) are accepted deterministically. Deteriorating moves ($\Delta f > 0$) are accepted with a probability defined by the Boltzmann distribution, which decays as the artificial temperature parameter $T$ cools over time: $P(\text{accept } s') = \exp\left(-\frac{\Delta f}{T}\right)$.
- **Adaptive Boltzmann Metropolis (ABMC):** An advanced variant of BMC where the temperature parameter $T$ or the cooling schedule is not strictly monotonic. Instead, it dynamically adapts based on real-time landscape feedback, such as recent variance in objective values or a target acceptance rate.
- **Generalized Tsallis Simulated Annealing (GTSA):** Grounded in non-extensive Tsallis statistical mechanics, this criterion replaces the standard exponential Boltzmann distribution with a generalized $q$-exponential function. This permits a heavier-tailed probability distribution for accepting deteriorating moves, enhancing global exploration: $P(\text{accept } s') = \left[1 - (1-q)\frac{\Delta f}{T}\right]^{\frac{1}{1-q}}$.
- **Monte Carlo (MC):** A foundational stochastic acceptance rule. In practice, it either acts as a static-temperature Metropolis acceptance or applies a fixed, parameterized probability $p$ to tolerate specific worsening moves without a time-dependent schedule.
- **Exponential Monte Carlo Counter (EMCC):** A stochastic mechanism that modulates the acceptance probability of deteriorating moves using an exponentially decaying counter or tolerance phase. It serves as a computationally lighter alternative to full simulated annealing, evaluating bounds via $P(\text{accept } s') = \exp(-\Delta f / c)$ where $c$ is a dynamic step-counter.
- **Probabilistic Transition (PT):** A generic stochastic rule defining acceptance via a dynamic Markov transition matrix or state-dependent function $P(s \to s')$, often utilized to bias the search trajectory toward historically promising sub-regions.
- **Fitness Proportional (FP):** A stochastic acceptance model (often utilized in evolutionary local search) where the probability of transitioning to $s'$ is directly proportional to a normalized transformation of its fitness relative to the incumbent $s$ or a local pool: $P(\text{accept } s') = \frac{F(s')}{F(s) + F(s')}$.

## Threshold & Bounded-Deterioration Criteria

- **Threshold Accepting (TA):** A deterministic analogue to Simulated Annealing. It explicitly bounds the allowable deterioration using a threshold parameter $\tau$. A candidate is accepted if the objective degradation is within the permissible bound: $f(s') - f(s) \le \tau$. The threshold $\tau$ is monotonically decreased toward zero as the search progresses.
- **Record-to-Record Travel (RRT):** A deterministic, bounded-deviation rule relative to the global optimum. Instead of comparing the candidate strictly to the current solution, RRT accepts $s'$ if its cost is within a fixed scalar margin $\delta$ of the best-known solution $s^*$: $f(s') \le f(s^*) + \delta$.
- **Great Deluge (GD):** A deterministic, non-elitist criterion that utilizes an absolute cost ceiling known as the “water level” $W$. A candidate is accepted strictly if its objective value remains submerged below this boundary ($f(s') \le W$). To force convergence, $W$ is monotonically decreased at a specific decay rate $\Delta W$ at each iteration.
- **Non-Linear Great Deluge (NLGD):** Extends the standard GD algorithm by decaying the water boundary $W$ non-linearly (e.g., exponentially or logarithmically) rather than by a fixed scalar. This facilitates rapid initial exploration that asymptotically tightens as the search converges.
- **Demon Algorithm (DA):** A deterministic, bounded-deterioration strategy. It utilizes a conceptual “demon” possessing an energy credit $D$. Worsening moves ($\Delta f > 0$) are accepted strictly if $\Delta f \le D$, after which the demon’s energy is depleted ($D \leftarrow D - \Delta f$). Improving moves ($\Delta f < 0$) replenish the demon’s energy.
- **Skewed Variable Neighborhood Search (SVNS):** An extension of VNS that permits acceptance of deteriorating moves if the candidate solution is located at a sufficient structural distance from the incumbent. It evaluates acceptance using a distance metric $\rho(s, s')$ and an asymmetry parameter $\alpha$: $f(s') - \alpha \cdot \rho(s, s') < f(s)$.

## Memory & History-Based Criteria

- **Late Acceptance Hill-Climbing (LAHC):** A threshold-based memory criterion that mitigates the need for explicit cooling schedules. It maintains a finite circular array of the costs from the last $L$ iterations. A candidate is accepted if it is better than or equal to the cost encountered exactly $L$ steps ago, naturally adapting the threshold to the current region of the search landscape: $f(s') \le f(s_{i-L})$.
- **Step Counting Hill Climbing (SCHC):** A discrete memory-based criterion that utilizes a single static cost bound $B$. The bound remains fixed for a predefined step-limit $L$, allowing the search to explore locally. A candidate is accepted if $f(s') \le B$. Once $L$ steps are exhausted, the bound is explicitly updated to match the current incumbent’s cost ($B \leftarrow f(s)$).
- **Old Bachelor Acceptance (OBA):** A dynamic, non-monotone threshold strategy. It utilizes an acceptance threshold that automatically adjusts based on recent search history. If a move is accepted, the threshold is aggressively tightened (mimicking high standards); if a series of moves are rejected, the threshold is gradually relaxed (lowering standards) to force diversification and escape local minima.

## Multi-Objective & Ensemble Criteria

- **Pareto Dominance (PD):** The foundational strict acceptance criterion for multi-objective landscapes. A candidate $s'$ is accepted (or supersedes $s$) if it strictly dominates the incumbent, meaning $f_i(s') \le f_i(s)$ for all objectives $i$, and $f_j(s') < f_j(s)$ for at least one objective $j$.
- **Epsilon Dominance ($\epsilon$-Dominance):** A relaxed multi-objective criterion that accelerates convergence. A candidate $s'$ is accepted if it $\epsilon$-dominates the incumbent, requiring that $(1-\epsilon)f_i(s') \le f_i(s)$ for all minimized objectives $i$, preventing mathematically negligible improvements from stalling the search or cluttering Pareto archives.
- **Tournament Acceptance:** The candidate $s'$ is evaluated not just against the incumbent $s$, but against a randomized localized pool of solutions. It is accepted into the active trajectory if it mathematically outranks a defined subset of those competitors.
- **Ensemble Move Acceptance (EMA):** A meta-decision architecture that evaluates a candidate move through a portfolio of heterogeneous criteria (e.g., SA, GD, and IE concurrently). The final acceptance decision is aggregated using logical ensemble rules, such as G-AND (strict consensus/minority rule), G-OR (authority rule where a single positive vote accepts), G-VOT (majority vote), or G-PVO (probabilistic voting based on criteria confidence).

------------------------------------------------------------------------

# Route Improvement and Refinement Strategies

Route improvement algorithms operate on pre-existing, structurally feasible routing configurations to systematically reduce operational costs or maximize net profit. This taxonomy categorizes the implemented algorithms ranging from fast localized topological operators to exact mathematical decompositions and machine-learning-augmented orchestrators.

### 1. Constructive & Augmentation Heuristics

These algorithms dynamically insert unassigned or highly profitable nodes into an active routing structure, modifying vehicle capacities and route boundaries.

- **Cheapest Insertion Augmentation:** A greedy constructive heuristic. For every unassigned node $k$ and valid adjacent edge $(i, j)$ in active routes, it computes the marginal insertion penalty $\Delta C = d_{i,k} + d_{k,j} - d_{i,j}$. For profit-aware environments, it evaluates net marginal profit (Revenue - $\Delta C$). The node yielding the absolute maximum economic or spatial gain is iteratively committed.
- **Regret-$k$ Insertion Augmentation:** An advanced constructive heuristic mitigating spatial shortsightedness. For each unassigned node $i$, it computes insertion costs across its $k$ best viable positions. The “regret” penalty for failing to insert the node into its optimal position is $\text{Regret}_i = \sum_{j=2}^k (\Delta C_{i,j} - \Delta C_{i,1})$. Nodes with maximum regret are prioritized for insertion.
- **Profitable Detour Augmentation:** A spatially-constrained heuristic for the Vehicle Routing Problem with Profits (VRPP). For active edge $(u, v)$ and unassigned bin $b$, if the spatial detour $\Delta d = d_{u,b} + d_{b,v} - d_{u,v}$ satisfies a proportional threshold ($\Delta d \le \epsilon \cdot d_{u,v}$), the net economic gain $\Delta \text{Profit} = (R \cdot w_b) - (C \cdot \Delta d)$ is calculated. Candidates with positive marginal utility are greedily assimilated.
- **Path-Induced Synergistic Pickup:** An opportunistic filter exploiting fleet physical trajectories. If an unassigned bin physically resides along the exact geometric shortest-path sequence between consecutive scheduled stops $(u, v)$, and slack capacity exists, the bin is seamlessly assimilated at zero marginal routing distance.

### 2. Intra-Route Intensification (TSP Optimizers)

These exact or highly-optimized heuristic solvers treat individual vehicle routes as isolated Traveling Salesperson Problems (TSP), strictly optimizing the sequence of visits without altering route assignments.

- **Steepest 2-opt Refinement:** A deterministic intensification operator. It exhaustively maps the 2-opt neighborhood (edge crossing removals) for every individual route independently. Executing a steepest-descent trajectory, it commits to the permutation yielding the absolute maximum cost reduction until a strict intra-route local minimum is reached.
- **Dynamic Programming (DP) Route Reoptimization:** An exact intra-route solver utilizing the foundational Held-Karp dynamic programming algorithm. It guarantees the absolute optimal sequence for any subset of nodes assigned to a vehicle. Given its $\mathcal{O}(|V|^2 2^{|V|})$ complexity, it is strictly gated by a hard cardinality limit (e.g., $|V| \le 20$).
- **Fast TSP Refinement:** A high-speed, scalable local search heuristic. It systematically isolates individual routes, strips depot connections, and delegates the isolated clusters to a highly optimized C++ TSP backend, ensuring each vehicle sequence is locally optimal before evaluating complex inter-route exchanges.
- **Lin-Kernighan-Helsgaun (LKH) Refinement:** The premier state-of-the-art heuristic for the TSP. It dynamically constructs highly complex, variable $k$-opt sequential exchanges. To constrain the exponential search space, the Helsgaun variant restricts evaluations to the algorithmic $\alpha$-nearness 1-tree, heavily prioritizing highly probable edges.

### 3. Inter-Route & Topological Local Search

These algorithms operate across multiple vehicle routes simultaneously, performing complex structural recombinations to balance load distributions and optimize spatial boundaries.

- **Classical Local Search:** A deterministic trajectory-based refinement wrapping a suite of fundamental operators (e.g., 3-opt, 4-opt, swap-star). It systematically evaluates localized edge permutations and drives the multi-route solution to a strict local minimum via iterative or steepest-descent improvement.
- **Steepest Node Exchange:** An exhaustive steepest-descent operator computing the exact objective delta for all possible pairwise spatial swaps ($i \leftrightarrow j$) across both intra- and inter-route neighborhoods, irrevocably committing to the single optimal permutation per iteration.
- **Or-opt Refinement (Steepest & Iterative):** An aggressive generalization of localized routing. It models the extraction and optimal re-insertion of contiguous sequence chains (lengths $L \in \{1, 2, 3\}$). The steepest variant rigorously maps the entire neighborhood before execution, while the iterative variant utilizes a first-improvement trajectory for rapid convergence.
- **Cross-Exchange Local Search:** An advanced inter-route trajectory operator generalizing swaps and 2-opt\* moves. It exchanges a contiguous segment (bounded by length $L$) from route $r_1$ with a segment from $r_2$, effectively breaking and reconnecting four distinct edges simultaneously to facilitate massive structural recombinations.

### 4. Trajectory Meta-Heuristics

These frameworks introduce stochasticity or memory structures to accept objective deterioration, deliberately navigating the search trajectory out of deep local optima basins.

- **Simulated Annealing (SA) Refinement:** A stochastic trajectory meta-heuristic utilizing topological perturbations. Move acceptance is strictly governed by the Boltzmann-Metropolis thermodynamic criterion ($P = \exp(\Delta f / T)$), permitting controlled objective deterioration as an artificial temperature parameter cools over successive epochs.
- **Guided Local Search (GLS):** A penalty-based meta-heuristic. It monitors the sequence of edges utilized in incumbent solutions and penalizes edges that are frequently traversed but globally suboptimal. Operators evaluate moves based on an augmented utility function: $U(i, j) = c_{ij} / (1 + p_{ij})$, forcing diversification into unexplored topological regions.
- **Randomized Local Search:** A stochastic search bypassing exhaustive steepest descent. It utilizes a predefined probability distribution to randomly sample and apply an operator (e.g., Or-opt, cross-exchange) at each iteration, introducing chaotic permutations to escape shallow local optima.

### 5. Large Neighborhood Search (Destroy & Repair)

Algorithms that aggressively dismantle substantial portions of the routing topology and intelligently reconstruct them to bridge disparate regions of the solution space.

- **Ruin and Recreate (LNS):** A foundational destroy-and-repair meta-heuristic. It disrupts the incumbent by applying spatial or randomized removal operators, subsequently rebuilding via greedy or regret insertion. Acceptance of candidate states is controlled via pluggable thermodynamic criteria to decouple search trajectories from strict monotonic improvement.
- **Adaptive Large Neighborhood Search (ALNS):** A highly sophisticated meta-heuristic where operator selection is governed dynamically by a Multi-Armed Bandit utilizing Thompson Sampling. The bandit maintains a Bayesian posterior of success for each ruin (e.g., Shaw, worst) and recreate operator, biasing selection toward pairs historically yielding the highest objective improvements.

### 6. Exact & Matheuristic Solvers

Hybrid architectures combining the speed of local search with the rigorous mathematical bounding capabilities of Mixed-Integer Linear Programming (MILP).

- **Set-Partitioning (Pool-Restricted Exact):** A matheuristic that automatically constructs a massive pool of candidate routes through diverse generation strategies (e.g., LNS perturbations, Held-Karp sequence optimization, mandatory singletons). Following canonical deduplication, a restricted Set Partitioning MILP model is solved to global optimality over the pool.
- **Set-Partitioning Polish:** An exact mathematical refinement that acts strictly on a pre-supplied pool of high-quality routes, utilizing a commercial MILP solver (e.g., Gurobi) to extract the optimal combination of routes while enforcing capacity and visit constraints.
- **Branch-and-Price (B&P) Refinement:** An exact column-generation matheuristic applied as a localized improver. It formulates the routing state as a Set Partitioning Master Problem. To iteratively populate the Restricted Master Problem with improving routes, it solves an Exact Resource-Constrained Shortest Path Problem (RCSPP) utilizing $ng$-route relaxation and Lagrangian bounding, guaranteeing integrality via Ryan-Foster branching.
- **Fix-and-Optimize Matheuristic:** A decomposition-based exact strategy. It permanently “fixes” (locks) a subset of high-performing routes, while nodes belonging to the remaining “free” routes are passed to an MILP solver. The solver exactly re-optimizes this restricted sub-MIP, circumventing the intractability of solving the global graph simultaneously.

### 7. Neural & Meta-Algorithmic Orchestrators

Advanced algorithmic management systems that sequence operators or utilize deep learning to map the combinatorial search space.

- **Learned Route Improver (Neural Operator):** A machine-learning-augmented operator utilizing a pre-trained Graph Neural Network (GNN) to bypass the $\mathcal{O}(n^k)$ computational bottleneck of exhaustive $k$-opt evaluations. Using multi-layer perceptron (MLP) node/edge encoders, a neural “move head” directly predicts expected objective improvements of topological permutations, executing the highest-scoring moves sequentially.
- **Multi-Phase Composition:** A meta-algorithmic architecture pipelining multiple distinct refinement strategies into a sequential execution graph. It constructs complex life-cycles—e.g., executing a greedy augmentation phase, followed by inter-route SA local search, concluding with an exact LKH polish—while tracking granular metrics across discrete transitions.
