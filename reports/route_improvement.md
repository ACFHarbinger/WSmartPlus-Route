# Route Improvement and Refinement Strategies

Route improvement algorithms operate on pre-existing, structurally feasible routing configurations to systematically reduce operational cost or maximize net profit. The taxonomy below progresses from fast, localized topological operators to exact mathematical reformulations and machine-learning-augmented orchestrators. Algorithms within each category share the same fundamental mode of interaction with the solution — what they differ in is neighborhood scope, acceptance mechanism, and the mathematical guarantee (if any) attached to the output.

---

## Dynamic Node Insertion

These algorithms augment an existing feasible routing topology by inserting currently unassigned or highly profitable nodes, dynamically modifying route assignments without destroying the overall structural skeleton.

### Cheapest Insertion Augmentation ✅
A greedy constructive heuristic. For every unassigned node $k$ and valid insertion position between adjacent nodes $(i, j)$ in active routes, the marginal detour penalty is computed:

$$\Delta C_{k,ij} = d_{i,k} + d_{k,j} - d_{i,j}$$

In profit-aware environments, the effective gain is $\Delta G_{k,ij} = r_k - \Delta C_{k,ij}$. The node–position pair yielding the maximum economic or spatial gain is committed greedily, and the process iterates until no profitable insertion remains. Time complexity per iteration: $\mathcal{O}(|U| \cdot |V_{\text{active}}|)$.

### Regret-$k$ Insertion Augmentation ✅
An advanced constructive heuristic that mitigates the spatial myopia of pure greedy insertion. For each unassigned node $i$, insertion costs are evaluated across its $k$ best viable positions. The regret value — the opportunity cost of not placing $i$ in its optimal position — is:

$$\text{Regret}_i = \sum_{j=2}^k \bigl(\Delta C_{i,j} - \Delta C_{i,1}\bigr)$$

Nodes with maximum regret are prioritized: the algorithm first commits nodes that would suffer the greatest cost increase if deferred. This prevents the greedy order from systematically under-serving spatially isolated nodes with limited insertion opportunities.

### Profitable Detour Augmentation ✅
A spatially-constrained heuristic for the Vehicle Routing Problem with Profits (VRPP). For an active edge $(u, v)$ and an unassigned node $b$, the spatial detour is computed as $\Delta d = d_{u,b} + d_{b,v} - d_{u,v}$. A node is inserted only if the detour satisfies a proportional threshold ($\Delta d \le \epsilon \cdot d_{u,v}$) and the net economic gain is positive:

$$\Delta \text{Profit} = r_b - C \cdot \Delta d > 0$$

The threshold $\epsilon$ controls the geometric reach of insertion; lower values enforce tighter route adherence while higher values permit broader detours for high-revenue nodes.

### Path-Induced Synergistic Pickup ✅
An opportunistic zero-cost insertion filter exploiting fleet physical trajectories. If an unassigned node physically lies along the geometric shortest-path sequence between consecutive scheduled stops $(u, v)$ — that is, $d_{u,b} + d_{b,v} = d_{u,v}$ — and sufficient vehicle capacity exists, the node is assimilated at zero marginal routing cost. Particularly effective in dense urban topologies where many nodes lie geometrically co-linear.

---

## Intra-Route Intensification

These solvers treat each individual vehicle route as an isolated Traveling Salesperson Problem, optimizing the visit sequence strictly within a single route without altering inter-route node assignments. They are used as a polish step following construction or inter-route restructuring.

### Steepest 2-Opt Refinement ✅
A deterministic intensification operator. It exhaustively evaluates the complete 2-opt neighborhood — all possible edge-crossing removals — for each route independently. It commits to the single permutation yielding the maximum cost reduction and repeats until no improving 2-opt move exists, guaranteeing convergence to a strict 2-opt local minimum. Time complexity per pass: $\mathcal{O}(|R|^2)$ per route.

### Or-Opt Sequence Relocation ✅
A targeted intra-route improvement operator that extracts contiguous node chains of length $L \in \{1, 2, 3\}$ and reinserts them at every other position within the same route. It is strictly more powerful than 2-opt for sequences of length $\ge 2$ because it permits non-reversing relocations that 2-opt cannot achieve. The steepest variant evaluates the full neighborhood before committing; the first-improvement variant commits upon the first discovered improvement.

### Dynamic Programming (DP) Exact Reoptimization ✅
An exact intra-route solver applying the Held-Karp dynamic programming algorithm. It guarantees the globally optimal node sequence for any route assigned to a single vehicle. Executing in $\mathcal{O}(|R|^2 \cdot 2^{|R|})$ time and space, it is gated by a hard cardinality limit (typically $|R| \le 20$) and used selectively on short routes or as a post-processing polish on the final solution.

### Fast TSP Refinement ✅
A high-speed, scalable local search heuristic. Individual routes are isolated, stripped of depot connections, and delegated to a highly optimized C++ TSP backend for intra-sequence optimization. The backend applies a combination of 2-opt, 3-opt, or Or-opt moves using efficient data structures (neighbor lists, don't-look bits) to guarantee local optimality before the route is reintegrated into the multi-route solution.

### Lin-Kernighan-Helsgaun (LKH) Refinement ✅
The premier heuristic for the TSP sub-problem. LKH dynamically constructs variable-depth $k$-opt sequential exchanges, beginning from a promising 2-opt or 3-opt improving move and recursively extending the exchange chain until no further gain is achievable. The Helsgaun extension restricts the candidate set at each step using $\alpha$-nearness bounds derived from the minimum 1-tree, containing the otherwise exponential search space while enabling 5-opt and partition moves that standard local search cannot reach. Produces near-optimal solutions on instances with thousands of nodes.

---

## Inter-Route Topological Operators

These algorithms operate simultaneously across multiple vehicle routes, performing structural recombinations that balance load, reallocate nodes between vehicles, and optimize spatial boundaries. They are the primary mechanism for escaping the intra-route local optima left by single-route intensifiers.

### 2-opt* (Inter-Route Edge Exchange) ✅
The canonical inter-route extension of 2-opt (Potvin & Rousseau, 1995). Two routes $R_a$ and $R_b$ are selected. An edge $(u, u')$ in $R_a$ and an edge $(v, v')$ in $R_b$ are simultaneously removed and replaced with cross-route links $(u, v)$ and $(u', v')$, effectively swapping the route tails:

$$\Delta C = d_{u,v} + d_{u',v'} - d_{u,u'} - d_{v,v'}$$

Unlike intra-route 2-opt, this move does not reverse any sub-tour — both new segments inherit the orientation of their respective parent routes. Capacity feasibility is checked on both recombined routes before commitment. 2-opt* is often the first inter-route operator applied because its $\mathcal{O}(|V|^2)$ neighborhood is fast to evaluate and frequently yields large improvements.

### Node Relocation (Inter-Route Relocate) ✅
Removes a single node $u$ from its current route $R_a$ and inserts it at its cheapest feasible position in a different route $R_b \ne R_a$. The marginal cost delta is:

$$\Delta C = (d_{p_u, n_u} - d_{p_u, u} - d_{u, n_u}) + (d_{v, u} + d_{u, n_v} - d_{v, n_v})$$

where $p_u, n_u$ are the predecessor and successor of $u$ in $R_a$, and $v, n_v$ are the insertion position in $R_b$. Node Relocation is the single-node analogue of Or-opt applied inter-route, and is typically the highest-frequency improving move in CVRP local search. A full pass evaluating all $(u, v)$ pairs runs in $\mathcal{O}(|V|^2)$.

### Steepest Node Exchange ✅
An exhaustive steepest-descent operator computing the exact objective delta for all possible pairwise node swaps ($i \leftrightarrow j$) across both intra- and inter-route neighborhoods. It irrevocably commits to the single optimal transposition per iteration. The inter-route capacity constraint $\sum_{k \in R_a} q_k - q_i + q_j \le Q$ is checked before each candidate swap is scored.

### Or-Opt Refinement (Steepest and Iterative) ✅
A generalization of intra-route relocation applied across the full multi-route solution. Contiguous chains of length $L \in \{1, 2, 3\}$ are extracted from one route and reinserted into any feasible position across all routes — including the source route — providing both intra- and inter-route chain relocation in a single operator. The steepest variant maps the entire neighborhood exhaustively before committing; the iterative first-improvement variant exploits the first discovered positive delta for faster convergence.

### Cross-Exchange Local Search ✅
An advanced inter-route operator generalizing 2-opt* to contiguous segment swaps. It exchanges a segment of up to $L$ consecutive nodes from route $R_a$ with a segment from route $R_b$, simultaneously severing and reconnecting four distinct edges. This enables massive structural recombinations — effectively performing a constrained subtour exchange — that 2-opt* (which swaps only tails) cannot produce.

### Swap-Star ✅
A refined inter-route swap that addresses the geometric suboptimality of blind position-preserving swaps. Rather than naively swapping nodes $u \in R_a$ and $v \in R_b$ in-place, Swap-Star evaluates the globally optimal insertion position for $u$ within $R_b$ and for $v$ within $R_a$, selecting the pair that maximizes the net combined cost improvement. It produces strictly better solutions than standard pairwise swap at the cost of an $\mathcal{O}(|R|)$ inner loop per candidate pair.

### Classical Local Search ✅
A deterministic trajectory-based refinement wrapping a suite of inter-route operators (3-opt, 4-opt, Or-opt, Swap-Star). It evaluates localized edge permutations systematically, driving the multi-route solution to a strict local minimum via iterative or steepest-descent improvement. Serves as the intensification backbone within VNS, ALNS, and memetic algorithm frameworks.

---

## Trajectory Meta-Heuristics

These frameworks introduce controlled stochasticity or structured memory into the local search loop to accept objective deterioration deliberately, navigating the trajectory out of deep local optima basins that pure greedy descent cannot escape.

### Variable Neighborhood Descent (VND) ✅
A deterministic meta-heuristic that systematically sequences through a predefined ordered set of neighborhood structures $\mathcal{N}_1, \mathcal{N}_2, \dots, \mathcal{N}_k$. Starting from $\mathcal{N}_1$, local search is applied until a local minimum is reached. The search then transitions to $\mathcal{N}_2$ to attempt improvement in the larger or differently structured neighborhood. If an improvement is found, the method resets to $\mathcal{N}_1$; if $\mathcal{N}_k$ is exhausted without improvement, the current solution is declared the VND local optimum:

$$s^* \leftarrow \text{LocalSearch}_{\mathcal{N}_l}(s); \quad l \leftarrow \begin{cases} 1 & \text{if } f(s^*) < f(s) \\ l + 1 & \text{otherwise} \end{cases}$$

VND is both an effective standalone improver and the standard intensification engine within Variable Neighborhood Search (VNS). A typical VRP VND sequence applies: relocate $\to$ 2-opt* $\to$ swap $\to$ Or-opt $\to$ cross-exchange.

### Tabu Search (TS) Refinement ❌
A memory-augmented trajectory meta-heuristic that escapes local optima by explicitly prohibiting recently applied moves for a configurable tenure $|\mathcal{T}|$. At each iteration, the best non-tabu neighbor is selected — even if it worsens the objective — and committed. The tabu list $\mathcal{T}$ is maintained as a circular queue of recently applied move attributes (e.g., node pairs, edge identities):

$$s \leftarrow \arg\min_{s' \in \mathcal{N}(s) \setminus \mathcal{T}} f(s')$$

An Aspiration Criterion overrides the tabu status of any move that achieves a strict new global best $f(s') < f(s^*)$. Reactive Tabu Search adapts the tenure $|\mathcal{T}|$ dynamically: it increases tenure upon detecting cycling (repeated solution visits) and decreases it during stagnation. TS consistently reaches high-quality VRP solutions because its prohibition of revisits systematically diversifies the trajectory beyond the immediate local minimum.

### Iterated Local Search (ILS) ✅
A meta-heuristic that alternates between a perturbation phase — which escapes a local optimum by disrupting the solution beyond the reach of the incumbent local search — and an intensification phase — which re-optimizes from the perturbed state. The standard ILS loop is:

$$s_0 \to \text{LocalSearch}(s_0) \to \text{Perturb}(s^*) \to \text{LocalSearch}(s') \to \text{Accept?}(s', s^*)$$

The perturbation strength is the critical tuning parameter: too weak and the restart falls back to the same basin; too strong and it behaves as a random restart. A Double-Bridge 4-opt move is the canonical VRP perturbation because it creates a topological change that no 2-opt, 3-opt, or Or-opt operator can undo. The acceptance criterion (comparing $s'$ against $s^*$) may be OI, IE, or any stochastic criterion from the Acceptance Criteria taxonomy.

### Simulated Annealing (SA) Refinement ✅
A stochastic trajectory meta-heuristic using thermodynamic perturbations. At each iteration, a neighbor $s'$ is generated via a randomly selected topological operator (2-opt, Or-opt, relocate). Acceptance is governed by the Boltzmann–Metropolis criterion, permitting controlled objective deterioration as an artificial temperature $T$ cools:

$$P(\text{accept } s') = \begin{cases} 1 & \text{if } \Delta f \le 0 \\ \exp\!\left(-\dfrac{\Delta f}{T}\right) & \text{if } \Delta f > 0 \end{cases}$$

The geometric cooling schedule $T_{k+1} = \alpha T_k$ is standard; reheat schedules (periodically raising $T$) can be added to re-diversify upon stagnation. SA's primary advantage over TS is parameter simplicity — a single temperature governs all acceptance — at the cost of less directed diversification.

### Guided Local Search (GLS) ✅
A penalty-based meta-heuristic that escapes local optima by augmenting the objective function with learned edge penalties. It monitors which edges appear frequently in incumbent solutions that are locally but not globally optimal, assigning a penalty $p_{ij}$ to each. The modified objective penalizes over-exploited edges:

$$U(i,j) = \frac{c_{ij}}{1 + p_{ij}}$$

After each local search convergence, the penalties of the most profitable (most-used but costly) edges are incremented, redirecting the subsequent local search away from previously explored topological regions. GLS is particularly effective in symmetric VRP instances where suboptimal solutions share a small number of highly favoured but costly edges.

### Randomized Local Search (RLS) ✅
A stochastic search that bypasses exhaustive steepest descent by randomly sampling an operator and an application point at each iteration. The sampling distribution over operators (Or-opt, cross-exchange, 2-opt*) may be uniform or weighted by historical improvement rates. RLS introduces sufficient stochasticity to escape shallow local optima while remaining significantly faster than full neighborhood evaluation — making it the preferred inner loop for high-iteration-count meta-heuristics operating under strict time limits.

---

## Large Neighborhood Search (Destroy and Repair)

These algorithms escape local optima by aggressively dismantling a large portion of the current routing topology and intelligently reconstructing it, bridging the trajectory to topologically distant regions of the solution space that local operators cannot reach in a single step.

### Ruin and Recreate (LNS) ✅
The foundational destroy-and-repair meta-heuristic. At each iteration, a ruin operator ejects a subset $U$ of nodes from the current solution $s$ (via random, worst, Shaw, or spatial removal), and a recreate operator reinserts them into the remaining partial solution $s'$ (via greedy, regret, or GENI insertion). The candidate $s''$ is then evaluated against $s$ using a pluggable acceptance criterion — decoupling the exploration amplitude (controlled by the ruin operator) from the acceptance tolerance (controlled by the criterion). This decoupling is LNS's primary architectural advantage over local search: the move scale is not bounded by neighborhood size.

### Adaptive Large Neighborhood Search (ALNS) ✅
A highly adaptive LNS variant where operator selection is governed dynamically by a Multi-Armed Bandit with Thompson Sampling. The bandit maintains a Bayesian Beta posterior $\text{Beta}(\alpha_i, \beta_i)$ of success for each ruin–recreate operator pair, sampling a success probability $\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$ at each iteration and selecting the operator pair with the highest sampled value. Posterior parameters are updated multiplicatively after each application based on the quality of the resulting solution. ALNS outperforms static LNS on heterogeneous instances where no single operator is uniformly dominant — a common condition in multi-period and profit-maximizing VRP variants.

---

## Exact Solvers

These algorithms evaluate restricted or localized state spaces using rigorous mathematical bounding, guaranteeing absolute optimality for the sub-problems presented to them. Applied to sub-problems rather than the full instance, they act as powerful polish steps that heuristic operators cannot replicate.

### Set-Partitioning Polish ✅
An exact mathematical refinement operating on a pre-supplied pool of high-quality routes. It formulates the optimal route combination as a Set-Partitioning MILP:

$$\min \sum_r c_r x_r \quad \text{s.t.} \quad \sum_r a_{ir} x_r = 1 \; \forall i \in V, \quad x_r \in \{0,1\}$$

solved to global optimality by a commercial MILP solver (e.g., Gurobi, CPLEX). Particularly effective as a final-stage combinator after ALNS or LNS has generated a diverse route pool, because the set-partitioning problem over a fixed pool is far smaller than the full VRP.

### Branch-and-Price (B&P) Refinement ✅
An exact column-generation solver applied as a localized improver. The routing state is formulated as a Set Partitioning Master Problem (RMP). The pricing sub-problem — an Exact Resource-Constrained Shortest Path Problem (ESPPRC) solved with $ng$-route relaxation and Lagrangian bounding — iteratively generates improving route columns with negative reduced cost:

$$\bar{c}_r = c_r - \sum_{i \in V} \pi_i a_{ir} - \mu < 0$$

Integrality is enforced via Ryan-Foster branching. Applied to a restricted sub-graph (e.g., the routes and nodes involved in the most recent LNS destruction), B&P refinement achieves provable optimality over the affected sub-problem without the intractability of global B&P.

---

## Matheuristics

Hybrid architectures that explicitly combine the trajectory speed and exploration logic of heuristic search with the mathematical bounding power of MILP formulations, inheriting the best properties of both paradigms.

### Set-Partitioning Pool Construction ✅
A matheuristic that automatically builds a massive pool of candidate routes through diverse generation strategies — LNS perturbations, Held-Karp sub-sequence optimization, and mandatory singleton routes. After canonical deduplication, a restricted Set-Partitioning MILP is solved to global optimality over the heuristically generated pool. The quality of the final solution depends directly on the diversity of the pool: a heterogeneous pool from multiple solvers and perturbation seeds consistently outperforms a pool from a single trajectory.

### Fix-and-Optimize Matheuristic ✅
A decomposition-based exact strategy that permanently locks a subset of high-quality routes as binary constants, then passes the nodes belonging to the remaining "free" routes to an MILP sub-solver for exact re-optimization:

$$\min_{x \in \{0,1\}^{|F|}} f(x, \bar{x}) \quad \text{s.t.} \quad x_r = \bar{x}_r \; \forall r \in \text{Fixed}, \quad \text{capacity, covering constraints}$$

The fixed/free partition is selected based on route quality metrics (cost per demand, geographic compactness) and rotated across iterations, allowing the MILP to exactly re-optimize different sub-problems from a globally promising context.

### MIP Large Neighborhood Search (Exact Ruin and Recreate) ✅
A matheuristic variant of LNS where the repair phase delegates reconstruction to an exact MILP sub-solver rather than a greedy heuristic. A spatial or randomized ruin operator destroys a portion of the solution; the fragmented nodes are then re-inserted into the remaining locked routes by a sub-MIP that is solved to proven optimality:

$$\min_{y} \sum_{i \in U} \sum_{r \in \mathcal{R}_{\text{free}}} c_{ir} y_{ir} \quad \text{s.t.} \quad \sum_r y_{ir} = 1 \; \forall i \in U, \quad \text{capacity on } \mathcal{R}_{\text{free}}$$

This bridges rapid combinatorial exploration (the LNS outer loop) with mathematical exactness (the sub-MIP repair), consistently achieving tighter solutions than heuristic recreate at moderate computational overhead.

---

## Neural Algorithms

These approaches integrate trained deep learning architectures directly into the improvement or operator-selection process, learning from the topology of routing instances rather than applying generic mathematical rules.

### Learned Route Improver (Neural Operator) ✅
A GNN-augmented operator that bypasses the $\mathcal{O}(n^k)$ complexity of exhaustive $k$-opt evaluation. Multi-layer MLP node and edge encoders embed the current routing state into a latent representation. A neural "move head" directly predicts the expected objective improvement $\hat{\Delta f}$ for each candidate topological exchange, executing only the highest-scoring moves without enumerating the full neighborhood. The GNN is trained offline on solved instances and generalizes to new topologies of the same structural class.

### Neural Operator Selector (Neural Orchestrator) ✅
A deep neural policy that governs the high-level operator selection loop rather than executing moves itself. At each iteration, it extracts a state vector encoding the optimality gap, search progress, stagnation count, and recent operator performance history, and produces a probability distribution over the available operator suite via a policy-gradient-trained network. Unlike ALNS's Thompson Sampling bandit — which tracks only aggregate historical success — the neural orchestrator conditions its selection on the current search geometry, learning operator-to-landscape mappings that static bandit mechanisms cannot represent.

---

## Meta-Algorithmic Orchestrators

Advanced management systems that sequence, govern, or dynamically weight the execution of entire improvement algorithms rather than individual low-level operators. They operate at the strategy level, treating complete algorithms as atomic actions.

### Multi-Phase Composition ✅
A meta-algorithmic architecture that pipelines multiple distinct refinement strategies into a sequential execution graph. Complex life-cycles are constructed explicitly — for example: greedy augmentation $\to$ inter-route SA local search $\to$ exact LKH polish $\to$ set-partitioning combination — while tracking granular performance metrics (improvement per second, accepted-move rates, operator utilization) across each discrete transition. Phase boundaries are triggered by configurable conditions such as iteration limits, stagnation thresholds, or improvement-rate drops.

### Adaptive Ensemble Route Improver (Meta-ALNS) ✅
A sophisticated orchestrator managing a portfolio of complete high-level route improvement algorithms (LKH, SA, Ruin-Recreate, Fix-and-Optimize) as atomic bandit arms. An Exponential Moving Average (EMA) of historical objective improvements tracks the recent performance of each algorithm:

$$\text{EMA}_i \leftarrow (1 - \lambda) \cdot \text{EMA}_i + \lambda \cdot \Delta f_i$$

Selection is driven by Roulette Wheel sampling over the EMA-normalized weights, continuously re-weighting the portfolio toward the currently most effective strategy. Unlike the low-level ALNS bandit — which operates over destroy/repair operator pairs — Meta-ALNS operates over full algorithmic strategies, enabling coarser but more powerful macro-level adaptation to the current phase of the search.

---
