# Route Construction Algorithms
## Exact Stochastic and Decomposition Solvers

While meta-heuristics and neural models offer computational efficiency, exact solvers are the gold standard for establishing optimal baselines and providing theoretical guarantees. This work utilizes two primary exact frameworks for benchmarking and solving the state-space challenges of multi-period routing.

- **Integer L-shaped Benders Decomposition (ILS-BD):** An exact algorithmic framework extending classical Benders decomposition, engineered for Stochastic Mixed-Integer Linear Programs (SMILP) where second-stage (recourse) decisions involve discrete variables. Because standard continuous LP duality cannot formulate Benders cuts for integer subproblems, this architecture dynamically generates integer optimality cuts (Laporte and Louveaux, 1993) at integer-feasible nodes of the master tree. Let $S^k$ be the index set of binary master variables $x_i$ taking value 1 at iteration $k$, and $L$ be a valid lower bound on the expected recourse cost $Q(x)$. The global optimality cut isolates scenario-dependent routing:
  $$\theta \ge (Q(x^k) - L) \left( \sum_{i \in S^k} x_i - \sum_{i \notin S^k} x_i - |S^k| + 1 \right) + L$$

- **Logic-Based Benders Decomposition (LBBD):** A generalization of the Benders method that removes the requirement for the subproblem to be a continuous linear program. By modeling routing subproblems via alternative paradigms—such as Constraint Programming (CP)—LBBD derives exact bounding cuts through logical inference rather than extreme rays of a dual polyhedron. Let $f(x, y)$ be the combinatorial subproblem cost. The master problem utilizes a bounding function $B_k(x)$ inferred from the CP solver's proof of optimality or infeasibility, adding cuts of the form:
  $$\theta \ge B_k(x) \quad \text{where} \quad B_k(x^k) = \min_{y \in Y(x^k)} f(x^k, y)$$

- **Exact Stochastic Dynamic Programming (ESDP):** A rigorous mathematical framework for solving multi-stage stochastic routing challenges by evaluating Bellman's principle of optimality backward through the temporal horizon. By exhaustively evaluating the exact state-space transitions, ESDP computes the optimal policy for every system state $S_t$. The value function $V_t(S_t)$ is solved recursively:
  $$V_t(S_t) = \min_{x_t \in \mathcal{X}(S_t)} \left\{ C(S_t, x_t) + \gamma \mathbb{E}_{\omega_t} \left[ V_{t+1}\big(\mathcal{T}(S_t, x_t, \omega_t)\big) \right] \right\}$$
  While providing a flawless theoretical baseline, its practical execution is bottlenecked by the curse of dimensionality.

- **Progressive Hedging (PH):** A scenario-based decomposition strategy that tackles complexity by temporarily relaxing non-anticipativity constraints. It optimizes each probabilistic scenario $s \in \mathcal{S}$ independently and applies an augmented Lagrangian penalty to penalize deviations from a consensus policy $z$. At iteration $k$, the scenario subproblem minimizes:
  $$\min_{x_s} \left\{ p_s f_s(x_s) + w_s^{(k)T} x_s + \frac{\rho}{2} \| x_s - z^{(k-1)} \|^2 \right\}$$
  The consensus $z^{(k)} = \sum_{s \in \mathcal{S}} p_s x_s^{(k)}$ and multipliers $w_s^{(k+1)} = w_s^{(k)} + \rho(x_s^{(k)} - z^{(k)})$ are iteratively updated until convergence.

- **Scenario Tree Extensive Form (ST-EF):** The foundational baseline approach for solving finite stochastic routing models. It constructs the full probabilistic scenario tree and formulates the Deterministic Equivalent Problem (DEP) as a single, monolithic MILP:
  $$\min_{x, y_s} \sum_{s \in \mathcal{S}} p_s \left( c^T x + q_s^T y_s \right)$$
  $$\text{s.t.} \quad A x \le b, \quad T_s x + W_s y_s \le h_s \quad \forall s \in \mathcal{S}$$
  While circumventing algorithmic decomposition, its memory footprint scales exponentially.

- **Constraint Programming with Boolean Satisfiability (CP-SAT):** An inference architecture integrating finite-domain CP with modern Boolean Satisfiability (SAT) engines. Through lazy clause generation, complex spatial constraints and vehicle capacities are translated into SAT clauses dynamically. For instance, if routing node $i$ to $j$ under capacity violations causes conflict, a nogood clause $\neg(x_i = v_1 \land x_j = v_2)$ is injected into the SAT solver, frequently outperforming classical continuous linear relaxations in highly constrained spaces.

- **Branch-and-Bound (BB):** The foundational search algorithm for exact MILP optimization. BB systematically partitions the feasible decision space into smaller subproblems (branching) and computes continuous LP relaxation lower bounds $\text{LB} \le Z^*$ at each node. If a node's lower bound strictly exceeds the cost of the best-known integer incumbent solution ($\text{LB} \ge \text{UB}$), or if the subproblem is mathematically infeasible, the subtree is safely pruned. This implicitly exhausts the search space, guaranteeing global optimality without explicit enumeration.

- **Branch-and-Cut (BC):** An exact algorithm that enhances the classical Branch-and-Bound framework by embedding cutting plane methods directly into the search tree. At each node, the solver evaluates the fractional LP solution $\tilde{x}$. If $\tilde{x} \notin \text{conv}(\mathcal{X})$, the algorithm dynamically generates valid inequalities (cutting planes) $\alpha^T x \le \beta$ that strictly separate $\tilde{x}$ from the true integer polyhedron. This aggressively tightens the continuous relaxations and significantly reduces the exponential size of the search tree.

- **Branch-and-Price (BP):** A specialized exact algorithm integrating Branch-and-Bound with column generation, primarily deployed for Set Partitioning formulations featuring an exponentially large variable space. Instead of enumerating all possible routing variables, the Restricted Master Problem (RMP) is solved using a limited subset of columns. A pricing subproblem—typically modeled as an Elementary Shortest Path Problem with Resource Constraints (ESPPRC)—is then solved to dynamically generate new route columns $j$ exhibiting negative reduced costs:
  $$\bar{c}_j = c_j - \sum_{i \in V} \pi_i a_{ij} - \mu < 0$$
  where $\pi_i$ represent the dual prices of the node-covering constraints.

- **Branch-and-Price-and-Cut (BPC):** The state-of-the-art exact framework for integer multicommodity flow and routing problems. BPC unifies column generation and cutting planes within a branch-and-bound tree. To combat severe problem symmetry—where standard fractional branching yields negligible bound improvements—BPC dynamically separates advanced valid inequalities, such as Subset-Row Cuts (SRCs) or lifted cover inequalities derived from 0-1 knapsack constraints. The pricing subproblem is systematically modified to incorporate the dual variables of these generated cuts, ensuring the linear programming relaxation remains mathematically tight.

- **Smart Waste Collection - Two-Commodity Flow (SWC-TCF):** An MILP formulation adapted from the exact two-commodity network flow model (Baldacci et al., 2004). It utilizes continuous flow variables $y_{ij}$ (waste load) and $y'_{ij}$ (empty vehicle capacity) to replace exponential subtour elimination constraints. For vehicle capacity $Q$ and binary routing variables $x_{ij}$, the flow conservation guarantees valid routes:
  $$y_{ij} + y'_{ij} = Q x_{ij} \quad \forall (i,j) \in A$$
  $$\sum_{j \in V} y_{ji} - \sum_{j \in V} y_{ij} = q_i \quad \forall i \in V$$
  This dictates the optimal subset of bins and sequences to maximize $\sum_i (R \cdot q_i) - C \sum_{i,j} d_{ij} x_{ij}$, subject to multi-period SLA thresholding.

---
## Matheuristics

Matheuristics represent a powerful hybrid class of algorithms that combine the speed of trajectory-based local search with the mathematical rigor of exact optimization models.

- **ILS-RVND-SP (Iterated Local Search + Randomized VND + Set Partitioning):** A "gold-standard" hybridization proposed by Subramanian et al. (2013). It operates in two tightly coupled phases:
  1. **Route Generation:** The algorithm utilizes ILS to escape local optima, combined with a Randomized Variable Neighborhood Descent (RVND) local search. Unlike standard VND, RVND explores a set of neighborhood operators $\mathcal{N} = \{N_1, N_2, \dots, N_k\}$ in a completely random sequence. If an operator yields a strict objective improvement ($f(s') > f(s)$), the sequence is immediately reshuffled and reset, continuing until no operator in $\mathcal{N}$ can find an improvement.
  2. **Exact Selection:** The diverse pool of high-quality feasible routes generated by the ILS-RVND phase is collected into a universe $\Omega$. An exact Set Partitioning (SP) model is then solved via Integer Linear Programming (ILP) to extract the absolute optimal combination of routes. For a maximization objective (like VRPP), it evaluates binary route variables $x_r$ with route profit $P_r$:
  $$\max \sum_{r \in \Omega} P_r x_r \quad \text{s.t.} \quad \sum_{r \in \Omega} a_{ir} x_r \le 1 \quad \forall i \in V, \quad x_r \in \{0,1\}$$
  where $a_{ir} = 1$ if route $r$ visits node $i$.

- **Partial Optimization Metaheuristic Under Special Intensification Conditions (POPMUSIC):** A state-of-the-art decomposition architecture (Taillard & Voss, 2002) that dynamically partitions a massive routing instance into highly localized, manageable subproblems. Rather than relying on exact BCP solvers, this implementation leverages a KD-Tree spatial index to construct proximity networks. By querying a seed node's position, it identifies the $K$ closest route centroids in $\mathcal{O}(K \log N)$ time, bypassing $\mathcal{O}(N^2)$ brute-force distance computations. The selected overlapping routes form a subproblem that is rigorously re-optimized using a targeted local solver (e.g., ALNS or HGS). If the local optimization yields a cost reduction, the global giant tour is seamlessly updated.

- **Kernel Search (KS):** A restricted mathematical search framework (Angelelli et al., 2010) that decomposes the problem at the binary variable level. The algorithm first solves the continuous LP relaxation of the global MILP. Variables with strictly positive fractional values are extracted to form the initial "Kernel" ($\mathcal{K}_0$). The remaining zero-valued variables are sorted by their reduced costs and divided into discrete Buckets ($\mathcal{B}_1, \dots, \mathcal{B}_m$). At each iteration $i$, a restricted MILP is solved over the variable space $\mathcal{K}_{i-1} \cup \mathcal{B}_i$, strictly enforcing $x_j = 0 \; \forall j \notin (\mathcal{K}_{i-1} \cup \mathcal{B}_i)$. Any variable from $\mathcal{B}_i$ utilized in the optimal integer solution is permanently added to the Kernel for the next iteration:  $$\mathcal{K}_i = \mathcal{K}_{i-1} \cup \left\{ j \in \mathcal{B}_i \;\middle|\; x_j^* > 0 \right\}$$
- **Adaptive Kernel Search (AKS):** An advanced extension (Guastaroba et al., 2017) that dynamically adapts the kernel trajectory based on computational difficulty. AKS evaluates the runtime $t_{\text{solve}}$ required to solve the restricted MILP at the root node and classifies the specific instance bucket as EASY, NORMAL, or HARD using a defined threshold $t_{\text{easy}}$.
  To enforce strict bounds on computational time for HARD instances, AKS aggressively fixes highly promising variables derived from the LP relaxation $\tilde{x}$. Specifically, utilizing a tolerance parameter $\epsilon$ (e.g., 0.1), it forces variables near their upper bounds into the final integer solution:  $$x_j = 1 \quad \forall j \in \mathcal{K} \text{ where } \tilde{x}_j \ge 1 - \epsilon$$
  Conversely, for EASY instances, the mathematical fixing is bypassed, allowing the solver to iteratively expand the search kernel to chase higher-quality global bounds.

- **Local Branching (LB):** An exact matheuristic architecture (Fischetti & Lodi, 2003) that restricts the search space by placing strict mathematical bounds on the structural Hamming distance from a known feasible incumbent solution. Let $\bar{x}$ be the binary vector of the incumbent solution, $B_1 = \{j \mid \bar{x}_j = 1\}$, and $B_0 = \{j \mid \bar{x}_j = 0\}$. The algorithm introduces an asymmetric linear constraint to force the exact MILP solver to evaluate only solutions within a neighborhood $k$:
  $$\sum_{j \in B_0} x_j + \sum_{j \in B_1} (1 - x_j) \le k$$
  By restricting $k$ (e.g., $k=10$), the solver prunes millions of nodes instantly. If the sub-MIP reaches the time limit without finding an improvement, the algorithm dynamically diversifies the search space by relaxing the neighborhood (e.g., $k \leftarrow k + 2$) or adding a hard branch ($\ge k+1$) to permanently exclude the exhausted region.

- **Local Branching with Variable Neighborhood Search (LB-VNS):** A sophisticated hybrid framework (Hansen et al., 2006) that embeds the exact Local Branching solver within a Variable Neighborhood Search metaheuristic. It systematically cycles through three phases:
  1. **Shaking:** To escape deep local optima, it violently perturbs the incumbent by forcing the solver to find a starting point strictly at the boundary of neighborhood $N_k$. This is achieved by changing the LB inequality to a strict equality constraint:
     $$\sum_{j \in B_0} x_j + \sum_{j \in B_1} (1 - x_j) = k$$
  2. **Local Search:** It executes a tightly constrained standard Local Branching solve ($k_{LS} = 4$) originating from this new shaken point to seek local optimality.
  3. **Neighborhood Change:** If the resulting solution strictly dominates the global incumbent, $k$ is reset to $k_{\min}$ (intensification). Otherwise, the shaking radius is expanded $k \leftarrow k + k_{\text{step}}$ (diversification).

- **Relaxation Enforced Neighborhood Search (RENS):** A primal Large Neighborhood Search variant (Berthold, 2009) that functions strictly as a start heuristic, requiring no initial feasible incumbent. RENS solves the continuous LP relaxation of the VRP model to obtain a fractional optimal point $\tilde{x}$. It then constructs an aggressive sub-MIP to explore all possible feasible roundings of this point. For purely binary routing models, it mathematically fixes all naturally integer variables to their LP bounds:
  $$x_j = \tilde{x}_j \quad \forall j \text{ s.t. } \tilde{x}_j \in \{0, 1\}$$
  The exact solver is then executed strictly on the remaining fractional variables (forcing them to $\lfloor \tilde{x}_j \rfloor$ or $\lceil \tilde{x}_j \rceil$), rapidly establishing a high-quality global upper bound.

- **Cluster-First Route-Second (CF-RS):** A foundational two-stage routing paradigm (Fisher & Jaikumar, 1981) adapted for highly scalable, profit-aware logistics.
  1. **Assignment (Clustering):** The geographic space is partitioned into angular sectors, and a "seed" node is selected per sector (either by maximum demand or maximum radial distance). Unassigned nodes $i$ are then allocated to seeds $k$. This implementation supports both a high-speed greedy approximation based on insertion costs ($C_{ik} = d(0,i) + d(i,k) - d(0,k)$) and an exact Generalized Assignment Problem (GAP) formulation via MILP. The exact GAP maximizes network profit while respecting vehicle capacity $Q$:
     $$\max \sum_{i \in V} \sum_{k \in K} \left( R \cdot w_i - C \cdot C_{ik} \right) x_{ik}$$
     $$\text{s.t.} \quad \sum_{k \in K} x_{ik} \le 1 \quad \forall i, \quad \sum_{i \in V} w_i x_{ik} \le Q \quad \forall k$$
  2. **Routing (TSP):** Once clusters are mathematically locked, the intra-cluster routing is reduced to independent Traveling Salesperson Problems (TSP). These are solved using a continuous-domain metaheuristic, specifically Particle Swarm Optimization (PSO) with Enhanced Edge Recombination (EER) crossover and 2-opt mutation, which efficiently extracts the optimal intra-cluster sequence.

- **Relaxation Induced Neighborhood Search (RINS):** Similar to Local Branching, RINS turns the exact MILP solver into a neighborhood explorer. Instead of utilizing Hamming distances, it fixes all variables where the continuous LP relaxation perfectly matches the integer incumbent, executing the exact solver strictly on the remaining unfixed variables. *(NOT IMPLEMENTED)*

- **Restricted Master Heuristics (Heuristic Column Generation):** A framework that addresses the exponential column generation bottleneck in exact Branch-and-Price architectures. It replaces the exact subproblem (pricing) solver with high-speed metaheuristics (such as Tabu Search or ILS) to rapidly generate structurally valid columns with negative reduced costs. The exact MILP solver is then invoked exclusively on the Restricted Master Problem to optimally select combinations from this heuristically generated pool. *(NOT IMPLEMENTED)*

- **Learning-to-Branch / Learning-Augmented Matheuristics:** The modern frontier of routing matheuristics that integrates Machine Learning (e.g., Graph Neural Networks) directly into the Branch-and-Price architecture. Neural networks rapidly score and predict the outcome of strong branching tests, filtering out non-promising branches without the computationally exorbitant need to execute the LP solver. This ensures the algorithm traverses the search tree exponentially faster while retaining mathematical rigor. *(NOT IMPLEMENTED)*

---
## Meta-Heuristics

These are high-level frameworks used to find or select a heuristic that may provide a sufficiently good solution to an optimization problem.

- **Adaptive Large Neighborhood Search (ALNS)**: A domain-specific meta-heuristic utilizing ruin and recreate logic.

- **Artificial Bee Colony (ABC)**: A relabeled swarm-based meta-heuristic.

- **Evolution Strategy Mu Plus Lambda (ES-MPL)**: A canonical evolutionary meta-heuristic.

- **Evolution Strategy Mu, Kappa, Lambda (ES-MKL)**: A foundational evolutionary meta-heuristic introducing lifespan bounds.

- **Evolution Strategy Mu, Lambda (ES-MCL)**: A foundational evolutionary meta-heuristic focused on offspring selection.

- **Differential Evolution (DE)**: A canonical population-based meta-heuristic for continuous optimization.

- **Fast Iterative Localized Optimization (FILO)**: A domain-specific routing heuristic based on spatial granularity.

- **Firefly Algorithm (FA)**: A relabeled swarm meta-heuristic.

- **Genetic Algorithm (GA)**: A canonical evolutionary meta-heuristic.

- **GENIUS**: A deterministic local search meta-heuristic utilizing generalized insertion (GENI) and unstringing operators to evaluate non-adjacent vertex insertions via segment reversals.

- **Guided Local Search (GLS)**: A penalty-based canonical meta-heuristic.

- **Harmony Search (HS)**: A relabeled meta-heuristic isomorphic to Evolution Strategies.

- **Hybrid Genetic Search (HGS)**: A memetic algorithm combining GA with local search.

- **Hybrid Genetic Search with Ruin and Recreate (HGS-RR)**: A memetic algorithm intersecting population paradigms with trajectory operators.

- **Hybrid Volleyball Premier League (HVPL)**: A memetic algorithm combining swarm and trajectory principles.

- **Iterated Local Search (ILS)**: A canonical trajectory-based meta-heuristic.

- **K-Sparse Ant Colony Optimization (KS-ACO)**: An algorithmic regularization of the canonical ACO swarm meta-heuristic.

- **Knowledge Guided Local Search (KGLS)**: A domain-specific execution of Guided Local Search.

- **League Championship Algorithm (LCA)**: A relabeled evolutionary meta-heuristic.

- **Memetic Algorithm (MA)**: A general paradigm combining population search with local refinement.

- **Particle Swarm Optimization (PSO)**: A foundational continuous swarm archetype.

- **Particle Swarm Optimization Memetic Algorithm (PSOMA)**: A hybrid meta-heuristic combining swarm and local trajectory search.

- **Quantum Differential Evolution (QDE)**: A population-based meta-heuristic using chaotic distributions.

- **Reactive Tabu Search (RTS)**: A canonical memory-based meta-heuristic.

- **Simulated Annealing (SA)**: A canonical meta-heuristic based on statistical mechanics.

- **Simulated Annealing Neighborhood Search (SANS)**: A hybrid meta-heuristic combining SA with multiple neighborhood structures.

- **Sine Cosine Algorithm (SCA)**: A relabeled swarm meta-heuristic.

- **Slack Induction by String Removal (SISR)**: A domain-specific spatial meta-heuristic.

- **Soccer League Competition (SLC)**: A relabeled evolutionary meta-heuristic.

- **Tabu Search (TS):** A foundational trajectory-based meta-heuristic employing short-term memory to escape local optima.

- **Variable Neighborhood Search (VNS)**: A canonical meta-heuristic for trajectory diversification.

- **Volleyball Premier League (VPL)**: A relabeled evolutionary meta-heuristic.

---
## Hyper-Heuristics

Hyper-heuristics operate at a higher level of abstraction than classical meta-heuristics. Rather than searching the solution space directly, they search the space of heuristics, intelligently managing a portfolio of Low-Level Heuristics (LLHs) to adapt to the problem's topological landscape during execution.

* **Hidden Markov Model + Great Deluge Hyper-Heuristic (HMM-GD-HH):** A state-of-the-art online-learning selection hyper-heuristic modeled as an Input-Output Hidden Markov Model (IOHMM). It treats the underlying search space topology as a set of hidden states $S$ (e.g., improving, stagnating, escaping) that expand dynamically at a rate of $\mathcal{O}(\sqrt{\log t})$.
  The model takes LLHs as inputs $u_t$ and maps multi-objective solution changes ($\Delta$ Profit, $\Delta$ Cost) to a discrete observation alphabet $o_t$.
  1. **Action Selection:** At iteration $t$, the controller selects the LLH $u^*$ that maximizes a dynamic balance between expected normalized profit $\bar{P}(u)$ and the transition entropy $H$ (to enforce exploration), annealed by $\alpha_t$:     $$u^* = \arg\max_u \left[ \bar{P}(u) - \alpha_t \sum_{s'} P(s' | u) \ln(P(s' | u) + \epsilon) \right]$$
  2. **Belief Update:** The hidden state belief vector is updated using a Stochastic Online Expectation-Maximization (EM) approximation of the Forward algorithm:     $$b_t(j) \propto B_{u_t, j, o_t} \sum_i b_{t-1}(i) A_{u_t, i, j}$$
  3. **Acceptance:** Independent of the HMM's learning phase, candidate solutions are strictly filtered through a Great Deluge criterion adapted for maximization, where the water level $W_t$ monotonically rises:     $$f(s') \ge W_t \quad \text{where} \quad W_{t+1} = W_t + \lambda |f(s^*)|$$
* **Genetic Programming Hyper-Heuristic (GPHH):** A generative hyper-heuristic that evolves a mathematical scoring function for constructive routing (heuristic generation).
  * Represented as a GP expression tree, the function evaluates candidate insertions using continuous local tactical features such as `node_profit`, `distance_to_route`, `insertion_cost`, and `remaining_capacity`.
  * To resolve the exponential computational bottleneck of full constructive evaluation, it employs a K-Nearest Neighbors (K-NN) candidate list to restrict evaluations to the spatial endpoints of existing routes, mathematically reducing the time complexity from $\mathcal{O}(N^2 R)$ to $\mathcal{O}(N K R)$.
  * Fitness is rigorously evaluated via true generalization—calculating the average normalized profit across structurally distinct spatial training environments, heavily penalized by a parsimony coefficient to mitigate tree bloat.
  * The evolutionary process employs mutually exclusive deep genetic operators (80% subtree crossover, 20% point mutation or Ephemeral Random Constant perturbation) governed by strict Koza-style depth limits.

* **Guided Indicators Hyper-Heuristic (GIHH):** An adaptive selection hyper-heuristic utilizing Episodic Weight Updates via Roulette Wheel selection to manage a Multi-Objective Pareto Archive (ARCH).
  * Operator selection weights are dynamically guided by two primary performance indicators: **ScoreA** (Quality Reward), which tracks the frequency an operator's offspring successfully enters the non-dominated Pareto front, and **ScoreB** (Directional Reward), which tracks objective-space bias (e.g., favoring revenue maximization versus cost minimization).
  * The algorithm incorporates a rigorous theoretical correction for multi-objective balancing, resolving the "Equation 25 Paradox" from Chen et al. (2018).
  * It triggers directional weight updates strictly when an operator's bias is opposite to the current segment's dominant deviation, aggressively enforcing equilibrium across the Pareto frontier.
  * Additionally, structurally identical clones are explicitly rejected during archive evaluation to prevent archive explosion vulnerabilities.

* **Hidden Markov Model + Great Deluge Hyper-Heuristic (HMM-GD-HH):** An online-learning selection hyper-heuristic that mathematically treats the sequence of applied Low-Level Heuristics (LLHs) as a Markov chain.
  * Rather than assigning a single deterministic state, the system maintains a probability belief distribution over three hidden states (improving, stagnating, escaping) using the Forward Algorithm.
  * At each search iteration, the belief vector $P(S_t | O_{1:t})$ is updated via the observation likelihood of the normalized profit change $\Delta_{\text{norm}}$.
  * The LLH selection probabilities are then dynamically computed as the belief-weighted mixture of the per-state emission matrices.
  * Independent of the HMM's learning phase, candidate solutions are strictly filtered through a Great Deluge acceptance criterion, which deterministically accepts candidate solutions whose profit exceeds a linearly rising water level, eliminating the need for temperature parameter tuning.

* **Reinforcement Learning - Great Deluge Hyper-Heuristic (RL-GD-HH):** An adaptive parameter controller that integrates a utility-based Reinforcement Learning mechanism with a Great Deluge (GD) acceptance criterion, strictly mapping to the architecture of Ozcan et al. (2010).
  * The algorithm maintains a scalar utility $u_i$ bounded by an upper limit (e.g., $U_{\max} = 40$) for each heuristic, consistently selecting the operator with the highest utility while employing random tie-breaking.
  * It applies an additive reward ($u_i \leftarrow \min(U_{\max}, u_i + r)$) strictly upon strict objective improvement.
  * Neutral or worsening moves are explicitly punished using configurable adaptation variants: subtractive (RL1: $u \leftarrow \max(lb, u - p)$), divisional (RL2: $u \leftarrow \lfloor u/2 \rfloor$), or root (RL3: $u \leftarrow \lfloor\sqrt{u}\rfloor$).
  * Worsening candidate solutions are gated by the Great Deluge water level, which linearly updates from the initial solution quality $f_0$ toward a predefined quality lower-bound over the exact search budget.

* **Sequence-based Selection Hyper-Heuristic (SS-HH):** An online-learning hyper-heuristic based on Markov chain principles that constructs and evaluates variable-length *sequences* of Low-Level Heuristics (LLHs) prior to application. Rigorously modeled after Kheiri (2014), the selection engine is driven by two dynamically updated matrices: a Transition Matrix (`TMatrix`) encoding the success probability of executing heuristic $j$ immediately after heuristic $i$, and an Acceptance-Strategy Matrix (`ASMatrix`) dictating whether to extend the current heuristic sequence ($AS=0$) or terminate and apply it to the solution ($AS=1$). Sequential transitions are sampled via roulette-wheel selection. Upon yielding a new global best solution, all involved matrix entries receive a continuous proportional reward scaled by the normalized objective improvement ($\Delta_{\text{norm}}$). Finally, candidate sequences are evaluated through a time-decaying threshold acceptance criterion, allowing temporary objective deterioration within mathematically bound limits.

* **Hyper-Heuristic Ant Colony Optimization (HH-ACO):** A swarm-based meta-algorithmic controller operating strictly within the heuristic space. Instead of traversing spatial customer graphs, artificial ants construct fixed-length sequences of local search and ruin-recreate operators (e.g., 2-opt, relocate, shaw removal). The transition probability from operator $i$ to operator $j$ is governed by the Ant Colony System (ACS) pseudo-random proportional rule, mathematically balancing aggressive exploitation (via a $q_0$ probability threshold selecting $\arg\max [\tau_{ij}^\alpha \eta_{ij}^\beta]$) and proportional exploration (via roulette-wheel sampling). The underlying pheromone matrix $\tau_{ij}$ is bounded by MAX-MIN Ant System (MMAS) constraints ($\tau_{\min}$, $\tau_{\max}$) and incorporates an evaporation rate $\rho$ to continuously reinforce highly synergistic sequences of algorithmic operators.

* **Hyper-heuristic with Unstringing/Stringing and Local search K-opt (HULK):** An advanced, domain-specific operator management architecture tailored for dynamic routing landscapes, mapping directly to the framework proposed by Müller and Bonilha (2022). The controller governs a tripartite pool of operators: Unstringing (generalized destruction), Stringing (generalized greedy/regret reconstruction), and Local Search (intra/inter-route $k$-opt and swaps). Operator selection is handled by an $\epsilon$-greedy Adaptive Operator Selector that tracks the running historical success (average objective score and application frequency) of each operator within a sliding memory window. The selection weights are continuously recalibrated using a dual-decay learning rate ($\text{weight} \leftarrow (1 - \text{lr}) \cdot \text{weight} \cdot \text{decay} + \text{lr} \cdot \text{avg\_score}$) to ensure the algorithm rapidly exploits high-performing operators without permanent stagnation. Acceptance of candidate moves is strictly governed by a Simulated Annealing thermodynamic criterion, which computes the transition probability $P = \exp(\Delta f / T)$ as the artificial temperature $T$ cools over successive epochs.

---
