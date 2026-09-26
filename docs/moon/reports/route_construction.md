# Routing Solution Algorithms

This document catalogues the full algorithmic suite available for routing optimization, spanning four distinct abstraction tiers: **Exact and Decomposition Solvers** provide provably optimal solutions with worst-case exponential cost; **Matheuristics** hybridize MILP formulations with heuristic search for strong bounded solutions; **Meta-Heuristics** guide combinatorial search trajectories through stochastic and evolutionary mechanisms; and **Hyper-Heuristics** operate above the search layer, managing and sequencing entire portfolios of lower-level algorithms.

---

## Exact and Decomposition Solvers

Exact solvers are the gold standard for establishing provably optimal baselines. Their practical applicability is bounded by the problem's state-space complexity, but they serve as indispensable sub-solvers and benchmarking tools within larger hybrid frameworks.

### Branch-and-Bound (BB) ✅
The foundational search algorithm for exact MILP optimization. BB partitions the feasible decision space into smaller subproblems (branching) and computes continuous LP relaxation lower bounds $\text{LB} \le Z^*$ at each node. If a node's lower bound exceeds the best-known incumbent ($\text{LB} \ge \text{UB}$), or the subproblem is infeasible, the subtree is pruned. This implicitly exhausts the search space, guaranteeing global optimality without explicit enumeration.

### Branch-and-Cut (BC) ✅
Enhances BB by embedding cutting plane generation directly into the search tree. At each node, the fractional LP solution $\tilde{x}$ is evaluated. If $\tilde{x} \notin \text{conv}(\mathcal{X})$, the algorithm generates valid inequalities (cutting planes) $\alpha^\top x \le \beta$ separating $\tilde{x}$ from the integer polyhedron, aggressively tightening the LP relaxation and reducing the search tree exponentially.

### Branch-and-Price (BP) ✅
Integrates Branch-and-Bound with column generation, deployed for Set Partitioning formulations with exponentially many route variables. The Restricted Master Problem (RMP) is solved on a limited column subset. A pricing subproblem — typically an ESPPRC (Elementary Shortest Path Problem with Resource Constraints) — generates improving route columns with negative reduced cost:

$$\bar{c}_j = c_j - \sum_{i \in V} \pi_i a_{ij} - \mu < 0$$

### Branch-and-Price-and-Cut (BPC) ✅
The state-of-the-art exact framework for routing MIPs. BPC unifies column generation and cutting planes within the B&B tree. To combat symmetry — where fractional branching yields negligible bound improvements — BPC dynamically separates advanced valid inequalities such as Subset-Row Cuts (SRCs) and lifted cover inequalities from 0-1 knapsack relaxations. The pricing subproblem is modified to incorporate dual variables from generated cuts, keeping the LP relaxation tight throughout.

### Constraint Programming with Boolean Satisfiability (CP-SAT) ✅
Integrates finite-domain CP with modern SAT engines via lazy clause generation. Complex spatial constraints and vehicle capacities are translated dynamically into SAT clauses. When routing node $i$ to $j$ under capacity violation causes conflict, a nogood clause $\neg(x_i = v_1 \land x_j = v_2)$ is injected into the SAT solver, frequently outperforming continuous LP relaxations on highly constrained instances.

### Integer L-Shaped Benders Decomposition (ILS-BD) ✅
An exact framework for Stochastic MILPs where second-stage recourse decisions involve discrete variables. Standard LP duality cannot generate Benders cuts for integer subproblems, so this architecture dynamically produces integer optimality cuts (Laporte & Louveaux, 1993) at integer-feasible master tree nodes. Let $S^k$ be the index set of master variables taking value 1 at iteration $k$:

$$\theta \ge (Q(x^k) - L)\!\left(\sum_{i \in S^k} x_i - \sum_{i \notin S^k} x_i - |S^k| + 1\right) + L$$

### Logic-Based Benders Decomposition (LBBD) ✅
Removes the requirement that the Benders subproblem be a continuous LP. By modelling routing subproblems via Constraint Programming, LBBD derives exact bounding cuts through logical inference rather than dual rays. The master problem employs a bounding function $B_k(x)$ inferred from the CP solver's proof of optimality or infeasibility:

$$\theta \ge B_k(x), \quad B_k(x^k) = \min_{y \in Y(x^k)} f(x^k, y)$$

### Exact Stochastic Dynamic Programming (ESDP) ✅
Solves multi-stage stochastic routing by evaluating Bellman's optimality principle backward through the temporal horizon. The value function $V_t(S_t)$ is computed recursively over all system states:

$$V_t(S_t) = \min_{x_t \in \mathcal{X}(S_t)} \left\{ C(S_t, x_t) + \gamma\, \mathbb{E}_{\omega_t}\!\left[V_{t+1}\!\left(\mathcal{T}(S_t, x_t, \omega_t)\right)\right] \right\}$$

Provides a theoretically perfect baseline but is bottlenecked by the curse of dimensionality in large state spaces.

### Progressive Hedging (PH) ✅
A scenario-based decomposition that relaxes non-anticipativity constraints. Each probabilistic scenario $s \in \mathcal{S}$ is optimized independently, with an augmented Lagrangian penalty enforcing consensus toward a shared policy $z$:

$$\min_{x_s} \left\{ p_s f_s(x_s) + w_s^{(k)\top} x_s + \frac{\rho}{2}\|x_s - z^{(k-1)}\|^2 \right\}$$

The consensus $z^{(k)} = \sum_s p_s x_s^{(k)}$ and multipliers $w_s^{(k+1)} = w_s^{(k)} + \rho(x_s^{(k)} - z^{(k)})$ are updated iteratively until convergence.

### Scenario Tree Extensive Form (ST-EF) ✅
The baseline approach for finite stochastic routing models. Constructs the full probabilistic scenario tree and formulates the Deterministic Equivalent Problem (DEP) as a single monolithic MILP:

$$\min_{x, y_s} \sum_{s \in \mathcal{S}} p_s (c^\top x + q_s^\top y_s) \quad \text{s.t.} \quad Ax \le b,\; T_s x + W_s y_s \le h_s \;\forall s$$

Memory footprint scales exponentially with scenario count; used primarily for small-scale stochastic benchmarks.

### Smart Waste Collection — Two-Commodity Flow (SWC-TCF) ✅
An MILP adapted from the two-commodity network flow model (Baldacci et al., 2004). Continuous flow variables $y_{ij}$ (waste load) and $y'_{ij}$ (empty vehicle capacity) replace exponential subtour elimination constraints. For vehicle capacity $Q$ and binary routing variables $x_{ij}$, the flow conservation is:

$$y_{ij} + y'_{ij} = Q x_{ij} \;\forall (i,j) \in A, \quad \sum_j y_{ji} - \sum_j y_{ij} = q_i \;\forall i$$

Maximizes $\sum_i R \cdot q_i - C \sum_{ij} d_{ij} x_{ij}$ subject to multi-period SLA thresholding.

---

## Matheuristics

Matheuristics hybridize the speed of heuristic trajectory search with the mathematical bounding power of exact MILP formulations, inheriting the best properties of both paradigms.

### ILS-RVND-SP (Iterated Local Search + Randomized VND + Set Partitioning) ✅
A gold-standard hybridization (Subramanian et al., 2013) operating in two tightly coupled phases.

**Route Generation.** ILS escapes local optima while a Randomized Variable Neighborhood Descent (RVND) performs local search. Unlike standard VND, RVND explores neighborhoods $\mathcal{N} = \{N_1, \dots, N_k\}$ in a random sequence. On any improvement, the sequence is immediately reshuffled and reset, continuing until no operator yields improvement.

**Exact Selection.** The diverse route pool $\Omega$ generated by ILS-RVND is passed to a Set Partitioning ILP:

$$\max \sum_{r \in \Omega} P_r x_r \quad \text{s.t.} \quad \sum_{r \in \Omega} a_{ir} x_r \le 1 \;\forall i \in V,\; x_r \in \{0,1\}$$

### POPMUSIC ✅
A scalable decomposition architecture (Taillard & Voss, 2002). Partitions massive instances into overlapping subproblems via a KD-Tree spatial index: a seed node's $K$ closest route centroids are identified in $\mathcal{O}(K \log N)$ and re-optimized using a targeted local solver (ALNS or HGS). If the local solve yields cost reduction, the global solution is updated. The overlapping structure ensures boundary consistency without global re-optimization.

### Kernel Search (KS) ✅
A restricted variable-space framework (Angelelli et al., 2010). The LP relaxation of the global MILP identifies variables with strictly positive fractional values as the initial kernel $\mathcal{K}_0$. Remaining variables are sorted by reduced cost into buckets $\mathcal{B}_1, \dots, \mathcal{B}_m$. At iteration $i$, a restricted MILP over $\mathcal{K}_{i-1} \cup \mathcal{B}_i$ is solved; any $\mathcal{B}_i$ variable used in the optimal solution is permanently absorbed:

$$\mathcal{K}_i = \mathcal{K}_{i-1} \cup \left\{j \in \mathcal{B}_i \;\middle|\; x_j^* > 0\right\}$$

### Adaptive Kernel Search (AKS) ✅
Extends KS (Guastaroba et al., 2017) with runtime-adaptive variable fixing. The sub-MILP solve time $t_{\text{solve}}$ classifies each bucket as EASY, NORMAL, or HARD relative to threshold $t_{\text{easy}}$. For HARD instances, LP-near-optimal variables are aggressively fixed:

$$x_j = 1 \;\forall j \in \mathcal{K} \;\text{ where }\; \tilde{x}_j \ge 1 - \epsilon$$

For EASY instances, fixing is bypassed and the kernel expands freely to chase tighter global bounds.

### Local Branching (LB) ✅
An exact matheuristic (Fischetti & Lodi, 2003) that restricts the MIP search space via a Hamming-distance constraint from the incumbent $\bar{x}$. Let $B_1 = \{j \mid \bar{x}_j = 1\}$ and $B_0 = \{j \mid \bar{x}_j = 0\}$:

$$\sum_{j \in B_0} x_j + \sum_{j \in B_1}(1 - x_j) \le k$$

Restricting $k$ prunes the tree dramatically. If the sub-MIP reaches its time limit without improvement, the neighborhood is diversified ($k \leftarrow k + 2$) or the exhausted region is permanently excluded.

### Local Branching with VNS (LB-VNS) ✅
Embeds Local Branching within a Variable Neighborhood Search framework (Hansen et al., 2006). Three phases cycle iteratively: **Shaking** forces a starting point at the boundary of $N_k$ (equality constraint); **Local Search** runs a tight LB solve ($k_{LS} = 4$) from the shaken point; **Neighborhood Change** resets $k$ to $k_{\min}$ on improvement (intensification) or expands $k \leftarrow k + k_{\text{step}}$ on failure (diversification).

### Relaxation Enforced Neighborhood Search (RENS) ✅
A primal LNS variant (Berthold, 2009) requiring no initial feasible incumbent. Solves the LP relaxation to obtain $\tilde{x}$, then fixes all naturally integer variables to their LP bounds:

$$x_j = \tilde{x}_j \;\forall j : \tilde{x}_j \in \{0,1\}$$

The exact solver optimizes only the remaining fractional variables, rapidly establishing a high-quality upper bound.

### Relaxation Induced Neighborhood Search (RINS) ❌
Similar to Local Branching, but fixes variables based on LP-integer agreement rather than Hamming distance from the incumbent. All variables where the continuous LP relaxation $\tilde{x}_j$ exactly matches the integer incumbent $\bar{x}_j$ are fixed; the exact solver explores only the remaining unfixed subspace, concentrating computation at the most uncertain decision boundaries.

### Restricted Master Heuristic (Heuristic Column Generation) ❌
Addresses the computational bottleneck of exact Branch-and-Price by replacing the exact pricing subproblem with a fast meta-heuristic (Tabu Search, ILS) that rapidly generates route columns with negative reduced costs. The exact MILP solver is then invoked exclusively on the Restricted Master Problem to optimally select the column combination, separating the hard combinatorial pricing from the exact selection.

### Learning-to-Branch / Learning-Augmented Matheuristics ❌
The frontier of ML-integrated matheuristics. Graph Neural Networks score and predict the outcome of strong-branching tests, filtering non-promising branches without executing the LP solver at each candidate. The search tree is traversed orders of magnitude faster while retaining mathematical rigor, because the GNN amortizes the branching cost over training instances.

### Cluster-First Route-Second (CF-RS) ✅
A foundational two-stage paradigm (Fisher & Jaikumar, 1981). **Clustering:** the geographic space is partitioned into angular sectors with seed nodes per sector. Unassigned nodes $i$ are allocated to seeds $k$ via either a greedy approximation ($C_{ik} = d(0,i) + d(i,k) - d(0,k)$) or an exact Generalized Assignment Problem (GAP) MILP:

$$\max \sum_i \sum_k (R \cdot w_i - C \cdot C_{ik}) x_{ik} \quad \text{s.t.} \quad \sum_k x_{ik} \le 1 \;\forall i,\; \sum_i w_i x_{ik} \le Q \;\forall k$$

**Routing:** locked clusters are solved as independent TSPs via PSO with Enhanced Edge Recombination (EER) crossover and 2-opt mutation.

---

## Meta-Heuristics

High-level algorithmic frameworks that guide search trajectories through stochastic, memory-based, or population-driven mechanics. Organized below by primary search mechanism: trajectory-based (single-solution), evolutionary/population-based, and swarm-based.

### Trajectory-Based Methods

#### Simulated Annealing (SA) ✅
A canonical stochastic trajectory meta-heuristic based on statistical mechanics. The transition probability from state $s$ to neighbor $s'$ with $\Delta E = f(s') - f(s)$ is governed by the Metropolis criterion:

$$P(s \to s') = \begin{cases} 1 & \text{if } \Delta E \le 0 \\ \exp\!\left(-\dfrac{\Delta E}{T_k}\right) & \text{if } \Delta E > 0 \end{cases}$$

Temperature $T_k$ is reduced via a geometric cooling schedule $T_{k+1} = \alpha T_k$, $\alpha \in [0.8, 0.99]$, collapsing the algorithm into greedy descent at convergence.

#### Tabu Search (TS) ✅
A memory-augmented trajectory meta-heuristic. A Tabu List $\mathcal{T}_L$ of recent move attributes (e.g., node pairs, edge identities) explicitly prohibits revisiting recently explored configurations for a configurable tenure. At each step, the best non-tabu neighbor is selected — even if it worsens the objective. An Aspiration Criterion overrides the tabu status if a move achieves a new global best.

#### Reactive Tabu Search (RTS) ✅
Extends TS with dynamic tenure adaptation. A hash map tracks objective-state revisitations; if a configuration recurs, the tenure is expanded ($T \leftarrow T \cdot \gamma$) to force diversification. If no cycles occur over a defined epoch, the tenure is relaxed to permit localized intensification.

#### Iterated Local Search (ILS) ✅
A canonical trajectory meta-heuristic defined by a four-step Markov chain: `GenerateInitial` → `LocalSearch` → `Perturbation` → `AcceptanceCriterion`. The search escapes local optima $s^*$ via a perturbation function (e.g., double-bridge 4-opt move) to yield $s'$, then re-applies local search:

$$s'' \leftarrow \text{LocalSearch}(\text{Perturbation}(s^*))$$

Acceptance of $s''$ is governed by any pluggable acceptance criterion.

#### Variable Neighborhood Search (VNS) ✅
Formalizes diversification through nested, dynamically expanding neighborhoods $\mathcal{N}_1, \dots, \mathcal{N}_{k_{\max}}$. A stochastic shaking phase ($s' \leftarrow \text{Random}(N_k(s))$) is followed by deterministic local search ($s'' \leftarrow \text{LocalSearch}(s')$). If $s''$ improves upon the incumbent, the search recenters on $s''$ and resets to $k=1$; otherwise, the radius expands ($k \leftarrow k+1$) until $k_{\max}$ is reached.

#### Guided Local Search (GLS) ✅
A penalty-based trajectory meta-heuristic. It monitors expensive, frequently traversed edges in incumbent solutions and penalizes them to force diversification. The augmented cost $h(s)$ evaluated by local search is:

$$h(s) = f(s) + \lambda \sum_{i=1}^M p_i I_i(s), \quad \text{util}(i) = \frac{c_i}{1 + p_i}$$

Upon reaching a local optimum, the feature with maximum utility is penalized, redirecting subsequent local search to unexplored topological regions.

#### Guided Fast Local Search (GFLS) ❌
An acceleration architecture that integrates Fast Local Search (FLS) within the Guided Local Search (GLS) meta-heuristic. Instead of exhaustively evaluating the $\mathcal{O}(N^2)$ or $\mathcal{O}(N^3)$ neighborhood at each iteration, GFLS maintains an active neighborhood bit-mask. Nodes are evaluated only if their activation bit is set to True. When the GLS controller penalizes an edge, the activation bits of the incident nodes are flipped back to True, explicitly directing the subsequent local descent (e.g., 2-opt, Relocate) strictly toward the penalized features. If a node's full neighborhood is evaluated without yielding an improving move, it is deactivated. This dramatically reduces computational overhead by pruning dormant search regions while preserving the exact trajectory escape properties of GLS.

#### Adaptive Large Neighborhood Search (ALNS) ✅
A destroy-and-repair meta-heuristic where operator selection is governed by a multi-armed bandit. Operator weights $w_j$ are updated via exponential smoothing with reward $c$ (scaled by improvement quality) and smoothing factor $\rho$:

$$p_j = \frac{w_j}{\sum_{k \in \Omega} w_k}, \quad w_{j,t+1} = (1 - \rho)w_{j,t} + \rho c$$

The bandit continuously biases selection toward historically successful destroy-repair pairs, with acceptance governed by a Simulated Annealing criterion.

#### Large Neighborhood Search (LNS) ✅
The non-adaptive predecessor of ALNS. A single fixed destroy operator ejects a subset of nodes $U$ from the incumbent; a fixed repair operator reinserts them. Move acceptance is typically governed by a pluggable criterion (SA, TA, GD). LNS is distinguished from local search by its large move size — the destroyed neighborhood may encompass 10–40% of the solution — bridging topologically distant regions unreachable by edge-exchange operators.

#### Slack Induction by String Removal (SISR) ✅
A domain-specific meta-heuristic for heavily constrained routing (time windows, capacity). SISR removes sequences (strings) of spatially proximate vertices, probabilistically inducing capacity "slack" into the remaining routes. Let $\bar{c}$ be the average removal string length; the slack enables tight recompilation of the destroyed route segments without capacity infeasibility during repair.

#### GENIUS ✅
A deterministic local search utilizing Generalized Insertion (GENI) and Unstringing operators. Instead of standard adjacency insertions, GENIUS evaluates non-adjacent vertex placements by executing localized segment reversals. The net spatial cost of inserting vertex $v$ using reference nodes $i, j, k, l$ is:

$$\Delta C = d(i,v) + d(v,j) - d(i,k) - d(l,j) + d(k,l)$$

#### Fast Iterative Localized Optimization (FILO) ✅
A domain-specific routing heuristic based on spatial granularity. Instead of evaluating the full objective for every perturbation, FILO restricts edge-exchange evaluations to a spatially restricted $k$-NN subgraph, reducing neighborhood evaluation from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log K)$.

### Population and Evolutionary Methods

#### Genetic Algorithm (GA) ✅
A canonical evolutionary meta-heuristic simulating natural selection. A population of routing solutions (chromosomes) is evaluated via a fitness function $f(x)$. Parents are selected via binary tournament or roulette wheel. Specialized crossover operators (Ordered Crossover, Edge Recombination) maintain topological validity. Mutation introduces spontaneous swaps or inversions with probability $p_m$.

#### Hybrid Genetic Search (HGS) ✅
A state-of-the-art memetic algorithm (Vidal, 2022) combining a GA with aggressive local search intensification. To preserve diversity in highly constrained routing spaces, HGS employs a bi-criteria biased fitness:

$$BF(p) = \mathit{fit}(p) + \left(1 - \frac{nc_{pop}}{N}\right)\Delta(p)$$

where $nc_{pop}$ counts population clones and $\Delta(p)$ measures normalized Hamming distance to nearest population neighbors. Every crossover offspring immediately undergoes a steepest-descent TSP solver before population re-entry.

#### HGS with Ruin and Recreate (HGS-RR) ✅
Replaces HGS's static mutation operators with dynamic ALNS-style destroy-and-repair. Large-scale topological defects from crossover are disrupted and repaired using regret-insertion heuristics, rather than weak localized edge swaps.

#### Memetic Algorithm (MA) ✅
The general hybrid framework combining a population-based evolutionary strategy with trajectory-based local search. Each individual in the population undergoes local search intensification (e.g., VND, 2-opt, SA) after recombination, ensuring offspring improve toward the local basin floor before competing for survival. MAs consistently outperform pure GAs on VRP benchmarks due to the local search gradient. Key specialized variants:

- **MA with Dual Population (MA-DP):** Maintains two interacting sub-populations — one elitist (preserving high-quality solutions) and one diverse (enforcing novelty) — exchanging individuals periodically via a migration policy.
- **MA with Island Model (MA-IM):** Partitions the population into geographically isolated sub-populations (islands) that evolve independently and exchange individuals via migration events at fixed intervals, preventing global premature convergence.
- **MA with Tolerance-Based Selection (MA-TS):** Replaces strict elitist survivor selection with a tolerance-bounded criterion: offspring are retained even if slightly inferior to parents, up to a configurable fitness tolerance $\epsilon$. This mitigates premature convergence without sacrificing selection pressure.

#### Differential Evolution (DE) ✅
A population-based continuous meta-heuristic. For target vector $x_{i,t}$, a mutant vector is constructed via:

$$v_{i,t} = x_{r1,t} + F \cdot (x_{r2,t} - x_{r3,t})$$

A trial vector $u_{i,t}$ is formed via binomial crossover with probability $CR$; it replaces the target if $f(u_{i,t}) \le f(x_{i,t})$. For discrete routing, positions are decoded via random-key encoding.

#### Quantum Differential Evolution (QDE) ✅
Extends DE by incorporating quantum superposition principles into the population update. Instead of real-valued mutation vectors, individuals are represented as quantum probability amplitudes that collapse to discrete states at evaluation. The quantum rotation gate updates each amplitude toward the global best with a rotation angle derived from the fitness landscape, enabling faster convergence on multi-modal routing landscapes.

#### Evolution Strategy — $(\mu + \lambda)$-ES ✅
Maintains a parent population of size $\mu$; generates $\lambda$ offspring via mutation (e.g., Gaussian perturbation). The next generation selects the best $\mu$ from the combined parent-offspring pool, enforcing strict elitism:

$$P_{t+1} \leftarrow \text{SelectBest}(\mu,\; P_t \cup \text{Offspring}_t)$$

#### Evolution Strategy — $(\mu, \kappa, \lambda)$-ES ✅
Mitigates immortal local optima by bounding individual lifespan to $\kappa$ generations. Any individual exceeding its lifespan is strictly discarded, forcing the search to continuously explore novel genetic material even at temporary objective cost.

#### Evolution Strategy — $(\mu, \lambda)$-ES ✅
A non-elitist strategy where the next generation is selected exclusively from the $\lambda$ offspring (requiring $\lambda \ge \mu$), discarding all parents. This naturally prevents population stagnation at local optima:

$$P_{t+1} \leftarrow \text{SelectBest}(\mu,\; \text{Offspring}_t)$$

### Swarm Intelligence-Based Methods

#### K-Sparse Ant Colony Optimization (KS-ACO) ✅
An algorithmic regularization of canonical ACO for dense graphs. The spatial graph is pre-processed into a sparse $k$-NN graph, restricting ant transition evaluations to the $K$ nearest geometric neighbors:

$$P_{ij}^k = \frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_{l \in N_i^k} [\tau_{il}]^\alpha [\eta_{il}]^\beta}$$

This reduces per-iteration complexity from $\mathcal{O}(V^2)$ to $\mathcal{O}(VK)$ without sacrificing solution quality on clustered routing instances.

#### Particle Swarm Optimization (PSO) ✅
A continuous swarm meta-heuristic. Particles track personal best $P_{\text{best}}$ and global swarm best $G_{\text{best}}$. Velocity and position update as:

$$v_i^{t+1} = w v_i^t + c_1 r_1 (P_{\text{best},i} - x_i^t) + c_2 r_2 (G_{\text{best}} - x_i^t), \quad x_i^{t+1} = x_i^t + v_i^{t+1}$$

For discrete routing, positions are decoded via random-key permutation encoding.

#### PSO Memetic Algorithm (PSOMA) ✅
Hybridizes PSO swarm mechanics with trajectory local search. Following velocity-position updates, discrete tour representations undergo periodic local search refinement (2-opt, swap). The locally optimized tour is inverse-mapped to update particle coordinates, transforming PSO into a macro-level diversification controller over intensified solutions.

#### PSO with Distance-Based Algorithm (PSODA) ✅
Augments standard PSO with a distance-based repulsion mechanism that explicitly prevents particle clustering in the search space. Particles that fall within a minimum pairwise distance threshold $d_{\min}$ of each other trigger a repulsion update, diversifying the swarm before local search collapse.

#### Artificial Bee Colony (ABC) ✅
Mimics honey bee foraging. Employed bees explore neighborhoods via:

$$v_{ij} = x_{ij} + \phi_{ij}(x_{ij} - x_{kj}), \quad \phi_{ij} \in [-1,1]$$

Onlooker bees select food sources proportional to fitness $P_i = F_i / \sum_n F_n$. Sources not improving after a `limit` of cycles are abandoned; scout bees reinitialize them randomly, preventing permanent stagnation.

#### Firefly Algorithm (FA) ✅
A swarm meta-heuristic based on bioluminescent attraction. Firefly $i$ moves toward brighter firefly $j$ with attractiveness decaying with distance $r_{ij}$:

$$\beta(r) = \beta_0 e^{-\gamma r_{ij}^2}, \quad x_i^{t+1} = x_i^t + \beta_0 e^{-\gamma r_{ij}^2}(x_j^t - x_i^t) + \alpha(\text{rand} - 0.5)$$

#### Harmony Search (HS) ✅
Inspired by musical improvisation. A Harmony Memory (HM) stores candidate solutions. New solutions are composed component-by-component: with probability HMCR a component is sampled from HM (and perturbed with probability PAR by $bw \cdot \epsilon$); otherwise it is generated randomly. HM is updated if the new harmony improves upon its worst member.

#### Sine Cosine Algorithm (SCA) ✅
A trigonometric swarm meta-heuristic. Agent position updates toward destination $P_i$ via switching sine/cosine functions:

$$x_i^{t+1} = \begin{cases} x_i^t + r_1 \sin(r_2)|r_3 P_i^t - x_i^t| & r_4 < 0.5 \\ x_i^t + r_1 \cos(r_2)|r_3 P_i^t - x_i^t| & r_4 \ge 0.5 \end{cases}$$

where $r_1$ decays over time to tighten the search radius.

#### League Championship Algorithm (LCA) ✅
Models a sports league competition. Each solution is a team competing in a round-robin schedule. Teams (solutions) update their strategies by learning from their wins and losses against opponents, with the update rule analogous to a social learning model: each team moves toward the strategy of the winning team and away from the strategy of the losing team, weighted by a randomized learning rate.

#### Volleyball Premier League (VPL) and Variants ✅
A population meta-heuristic modelling volleyball team competition and league structure. Teams are partitioned into leagues; intra-league competition drives local intensification while inter-league promotion/relegation drives diversification. Extended variants include:

#### Hybrid VPL (HVPL) ✅
Integrates a secondary local search phase within each competition round, hybridizing swarm competition with trajectory descent.

#### Soccer League Competition (SLC) ✅
A swarm meta-heuristic modelling professional football league dynamics. Players (solutions) belong to teams, which compete in weekly matches. Player positions are updated based on the scoring outcomes: players on winning teams reinforce their current strategy (exploitation), while players on losing teams undergo formation changes (exploration). The league table ranking drives selection pressure, with top-ranked teams' strategies acting as attractor basins for the rest of the population.

---

## Hyper-Heuristics

Hyper-heuristics operate at a higher abstraction level than classical meta-heuristics. Rather than directly searching the solution space, they search the *space of heuristics*, managing and sequencing portfolios of Low-Level Heuristics (LLHs) to adapt to the problem's topological landscape during execution.

### Genetic Programming Hyper-Heuristic (GP-HH) ✅
A generative hyper-heuristic that evolves a mathematical scoring function for constructive routing. Represented as a GP expression tree, the function evaluates candidate insertions using tactical features (`node_profit`, `distance_to_route`, `insertion_cost`, `remaining_capacity`). A K-NN candidate list reduces time complexity from $\mathcal{O}(N^2 R)$ to $\mathcal{O}(NKR)$. Fitness is evaluated via average normalized profit across structurally distinct training environments, penalized by a parsimony coefficient. Evolutionary operators: 80% subtree crossover, 20% point mutation or Ephemeral Random Constant perturbation, governed by Koza-style depth limits.

### Guided Indicators Hyper-Heuristic (GIHH) ✅
An adaptive selection hyper-heuristic utilizing episodic weight updates via roulette wheel selection over a Multi-Objective Pareto Archive. Operator selection is guided by two indicators: **ScoreA** (Quality Reward) — frequency of offspring entering the non-dominated front — and **ScoreB** (Directional Reward) — objective-space bias toward revenue maximization vs. cost minimization. Directional updates trigger only when an operator's bias opposes the current dominant deviation, enforcing Pareto equilibrium. Structurally identical clones are rejected from the archive to prevent explosion.

### Hidden Markov Model + Great Deluge Hyper-Heuristic (HMM-GD-HH) ✅
An online-learning selection hyper-heuristic modelled as an Input-Output Hidden Markov Model (IOHMM). Hidden states (improving, stagnating, escaping) expand dynamically at rate $\mathcal{O}(\sqrt{\log t})$. LLHs are taken as inputs $u_t$; multi-objective solution changes map to a discrete observation alphabet $o_t$.

1. **Action Selection.** Selects the LLH $u^*$ maximizing the balance between expected normalized profit and transition entropy, annealed by $\alpha_t$:
   $$u^* = \arg\max_u \!\left[\bar{P}(u) - \alpha_t \sum_{s'} P(s'|u)\ln(P(s'|u) + \epsilon)\right]$$
2. **Belief Update.** The hidden state belief vector is updated via stochastic online EM (Forward algorithm approximation):
   $$b_t(j) \propto B_{u_t,j,o_t} \sum_i b_{t-1}(i) A_{u_t,i,j}$$
3. **Acceptance.** Candidate solutions pass a Great Deluge criterion with monotonically rising water level $W_{t+1} = W_t + \lambda|f(s^*)|$.

### Reinforcement Learning — Great Deluge Hyper-Heuristic (RL-GD-HH) ✅
An adaptive parameter controller integrating a utility-based RL mechanism with Great Deluge acceptance (Ozcan et al., 2010). Each heuristic $i$ maintains a scalar utility $u_i \in [0, U_{\max}]$. The highest-utility operator is always selected (random tie-breaking). On strict improvement: $u_i \leftarrow \min(U_{\max}, u_i + r)$. On neutral or worsening moves, punishment is applied via one of three variants: subtractive (RL1: $u \leftarrow \max(lb, u - p)$), divisional (RL2: $u \leftarrow \lfloor u/2 \rfloor$), or root (RL3: $u \leftarrow \lfloor\sqrt{u}\rfloor$). Candidate solutions are gated by a linearly updating Great Deluge water level.

### Sequence-Based Selection Hyper-Heuristic (SS-HH) ✅
An online-learning hyper-heuristic (Kheiri, 2014) that constructs and evaluates variable-length LLH *sequences* before application. Two dynamically updated matrices drive selection: **TMatrix** — the success probability of executing LLH $j$ immediately after $i$ — and **ASMatrix** — whether to extend the current sequence ($AS=0$) or terminate and apply it ($AS=1$). On a new global best, all involved matrix entries receive a proportional reward scaled by the normalized objective improvement $\Delta_{\text{norm}}$. Candidate sequences are evaluated through a time-decaying threshold acceptance criterion.

### Hyper-Heuristic Ant Colony Optimization (HH-ACO) ✅
A swarm-based meta-algorithmic controller operating strictly within the heuristic space. Artificial ants construct fixed-length sequences of local search and ruin-recreate operators. The transition probability from operator $i$ to $j$ follows the ACS pseudo-random proportional rule, balancing exploitation (probability $q_0$ threshold selecting $\arg\max[\tau_{ij}^\alpha \eta_{ij}^\beta]$) and exploration (roulette wheel). The pheromone matrix $\tau_{ij}$ is bounded by MMAS constraints ($\tau_{\min}$, $\tau_{\max}$) with evaporation rate $\rho$ to reinforce highly synergistic operator sequences.

### Hyper-Heuristic with Unstringing/Stringing and Local Search K-opt (HULK) ✅
An advanced domain-specific operator management architecture (Müller & Bonilha, 2022). Governs a tripartite operator pool: Unstringing (generalized destruction), Stringing (greedy/regret reconstruction), and Local Search (intra/inter-route $k$-opt and swaps). An $\epsilon$-greedy Adaptive Operator Selector tracks each operator's historical success within a sliding memory window. Selection weights are recalibrated via a dual-decay learning rate:

$$\text{weight} \leftarrow (1 - \text{lr}) \cdot \text{weight} \cdot \text{decay} + \text{lr} \cdot \text{avg\_score}$$

Candidate move acceptance is governed by the Boltzmann–Metropolis criterion with the correct sign convention:

$$P = \exp\!\left(-\frac{\Delta f}{T}\right)$$

as temperature $T$ cools over successive epochs.

---
