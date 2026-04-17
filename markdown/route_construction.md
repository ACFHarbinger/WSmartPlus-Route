# Route Construction Algorithms
## Exact Stochastic and Decomposition Solvers

While meta-heuristics and neural models offer computational efficiency, exact solvers are the gold standard for establishing optimal baselines and providing theoretical guarantees. This work utilizes two primary exact frameworks for benchmarking and solving the state-space challenges of multi-period routing.

### Integer L-shaped Benders Decomposition (ILS-BD)
An exact algorithmic framework extending classical Benders decomposition, engineered for Stochastic Mixed-Integer Linear Programs (SMILP) where second-stage (recourse) decisions involve discrete variables. Because standard continuous LP duality cannot formulate Benders cuts for integer subproblems, this architecture dynamically generates integer optimality cuts (Laporte and Louveaux, 1993) at integer-feasible nodes of the master tree. Let $S^k$ be the index set of binary master variables $x_i$ taking value 1 at iteration $k$, and $L$ be a valid lower bound on the expected recourse cost $Q(x)$. The global optimality cut isolates scenario-dependent routing:
  $$\theta \ge (Q(x^k) - L) \left( \sum_{i \in S^k} x_i - \sum_{i \notin S^k} x_i - |S^k| + 1 \right) + L$$

### Logic-Based Benders Decomposition (LBBD)
A generalization of the Benders method that removes the requirement for the subproblem to be a continuous linear program. By modeling routing subproblems via alternative paradigms—such as Constraint Programming (CP)—LBBD derives exact bounding cuts through logical inference rather than extreme rays of a dual polyhedron. Let $f(x, y)$ be the combinatorial subproblem cost. The master problem utilizes a bounding function $B_k(x)$ inferred from the CP solver's proof of optimality or infeasibility, adding cuts of the form:
  $$\theta \ge B_k(x) \quad \text{where} \quad B_k(x^k) = \min_{y \in Y(x^k)} f(x^k, y)$$

### Exact Stochastic Dynamic Programming (ESDP)
A rigorous mathematical framework for solving multi-stage stochastic routing challenges by evaluating Bellman's principle of optimality backward through the temporal horizon. By exhaustively evaluating the exact state-space transitions, ESDP computes the optimal policy for every system state $S_t$. The value function $V_t(S_t)$ is solved recursively:
  $$V_t(S_t) = \min_{x_t \in \mathcal{X}(S_t)} \left\{ C(S_t, x_t) + \gamma \mathbb{E}_{\omega_t} \left[ V_{t+1}\big(\mathcal{T}(S_t, x_t, \omega_t)\big) \right] \right\}$$
  While providing a flawless theoretical baseline, its practical execution is bottlenecked by the curse of dimensionality.

### Progressive Hedging (PH)
A scenario-based decomposition strategy that tackles complexity by temporarily relaxing non-anticipativity constraints. It optimizes each probabilistic scenario $s \in \mathcal{S}$ independently and applies an augmented Lagrangian penalty to penalize deviations from a consensus policy $z$. At iteration $k$, the scenario subproblem minimizes:
  $$\min_{x_s} \left\{ p_s f_s(x_s) + w_s^{(k)T} x_s + \frac{\rho}{2} \| x_s - z^{(k-1)} \|^2 \right\}$$
  The consensus $z^{(k)} = \sum_{s \in \mathcal{S}} p_s x_s^{(k)}$ and multipliers $w_s^{(k+1)} = w_s^{(k)} + \rho(x_s^{(k)} - z^{(k)})$ are iteratively updated until convergence.

### Scenario Tree Extensive Form (ST-EF)
The foundational baseline approach for solving finite stochastic routing models. It constructs the full probabilistic scenario tree and formulates the Deterministic Equivalent Problem (DEP) as a single, monolithic MILP:
  $$\min_{x, y_s} \sum_{s \in \mathcal{S}} p_s \left( c^T x + q_s^T y_s \right)$$
  $$\text{s.t.} \quad A x \le b, \quad T_s x + W_s y_s \le h_s \quad \forall s \in \mathcal{S}$$
  While circumventing algorithmic decomposition, its memory footprint scales exponentially.

### Constraint Programming with Boolean Satisfiability (CP-SAT)
An inference architecture integrating finite-domain CP with modern Boolean Satisfiability (SAT) engines. Through lazy clause generation, complex spatial constraints and vehicle capacities are translated into SAT clauses dynamically. For instance, if routing node $i$ to $j$ under capacity violations causes conflict, a nogood clause $\neg(x_i = v_1 \land x_j = v_2)$ is injected into the SAT solver, frequently outperforming classical continuous linear relaxations in highly constrained spaces.

### Branch-and-Bound (BB)
The foundational search algorithm for exact MILP optimization. BB systematically partitions the feasible decision space into smaller subproblems (branching) and computes continuous LP relaxation lower bounds $\text{LB} \le Z^*$ at each node. If a node's lower bound strictly exceeds the cost of the best-known integer incumbent solution ($\text{LB} \ge \text{UB}$), or if the subproblem is mathematically infeasible, the subtree is safely pruned. This implicitly exhausts the search space, guaranteeing global optimality without explicit enumeration.

### Branch-and-Cut (BC)
An exact algorithm that enhances the classical Branch-and-Bound framework by embedding cutting plane methods directly into the search tree. At each node, the solver evaluates the fractional LP solution $\tilde{x}$. If $\tilde{x} \notin \text{conv}(\mathcal{X})$, the algorithm dynamically generates valid inequalities (cutting planes) $\alpha^T x \le \beta$ that strictly separate $\tilde{x}$ from the true integer polyhedron. This aggressively tightens the continuous relaxations and significantly reduces the exponential size of the search tree.

### Branch-and-Price (BP)
A specialized exact algorithm integrating Branch-and-Bound with column generation, primarily deployed for Set Partitioning formulations featuring an exponentially large variable space. Instead of enumerating all possible routing variables, the Restricted Master Problem (RMP) is solved using a limited subset of columns. A pricing subproblem—typically modeled as an Elementary Shortest Path Problem with Resource Constraints (ESPPRC)—is then solved to dynamically generate new route columns $j$ exhibiting negative reduced costs:
  $$\bar{c}_j = c_j - \sum_{i \in V} \pi_i a_{ij} - \mu < 0$$
  where $\pi_i$ represent the dual prices of the node-covering constraints.

### Branch-and-Price-and-Cut (BPC)
The state-of-the-art exact framework for integer multicommodity flow and routing problems. BPC unifies column generation and cutting planes within a branch-and-bound tree. To combat severe problem symmetry—where standard fractional branching yields negligible bound improvements—BPC dynamically separates advanced valid inequalities, such as Subset-Row Cuts (SRCs) or lifted cover inequalities derived from 0-1 knapsack constraints. The pricing subproblem is systematically modified to incorporate the dual variables of these generated cuts, ensuring the linear programming relaxation remains mathematically tight.

### Smart Waste Collection - Two-Commodity Flow (SWC-TCF)
An MILP formulation adapted from the exact two-commodity network flow model (Baldacci et al., 2004). It utilizes continuous flow variables $y_{ij}$ (waste load) and $y'_{ij}$ (empty vehicle capacity) to replace exponential subtour elimination constraints. For vehicle capacity $Q$ and binary routing variables $x_{ij}$, the flow conservation guarantees valid routes:
  $$y_{ij} + y'_{ij} = Q x_{ij} \quad \forall (i,j) \in A$$
  $$\sum_{j \in V} y_{ji} - \sum_{j \in V} y_{ij} = q_i \quad \forall i \in V$$
  This dictates the optimal subset of bins and sequences to maximize $\sum_i (R \cdot q_i) - C \sum_{i,j} d_{ij} x_{ij}$, subject to multi-period SLA thresholding.

---

## Matheuristics

Matheuristics represent a powerful hybrid class of algorithms that combine the speed of trajectory-based local search with the mathematical rigor of exact optimization models.

### ILS-RVND-SP (Iterated Local Search + Randomized VND + Set Partitioning)
A "gold-standard" hybridization proposed by Subramanian et al. (2013). It operates in two tightly coupled phases:
  1. **Route Generation:** The algorithm utilizes ILS to escape local optima, combined with a Randomized Variable Neighborhood Descent (RVND) local search. Unlike standard VND, RVND explores a set of neighborhood operators $\mathcal{N} = \{N_1, N_2, \dots, N_k\}$ in a completely random sequence. If an operator yields a strict objective improvement ($f(s') > f(s)$), the sequence is immediately reshuffled and reset, continuing until no operator in $\mathcal{N}$ can find an improvement.
  2. **Exact Selection:** The diverse pool of high-quality feasible routes generated by the ILS-RVND phase is collected into a universe $\Omega$. An exact Set Partitioning (SP) model is then solved via Integer Linear Programming (ILP) to extract the absolute optimal combination of routes. For a maximization objective (like VRPP), it evaluates binary route variables $x_r$ with route profit $P_r$:
  $$\max \sum_{r \in \Omega} P_r x_r \quad \text{s.t.} \quad \sum_{r \in \Omega} a_{ir} x_r \le 1 \quad \forall i \in V, \quad x_r \in \{0,1\}$$
  where $a_{ir} = 1$ if route $r$ visits node $i$.

### Partial Optimization Metaheuristic Under Special Intensification Conditions (POPMUSIC)
A state-of-the-art decomposition architecture (Taillard & Voss, 2002) that dynamically partitions a massive routing instance into highly localized, manageable subproblems. Rather than relying on exact BCP solvers, this implementation leverages a KD-Tree spatial index to construct proximity networks. By querying a seed node's position, it identifies the $K$ closest route centroids in $\mathcal{O}(K \log N)$ time, bypassing $\mathcal{O}(N^2)$ brute-force distance computations. The selected overlapping routes form a subproblem that is rigorously re-optimized using a targeted local solver (e.g., ALNS or HGS). If the local optimization yields a cost reduction, the global giant tour is seamlessly updated.

### Kernel Search (KS)
A restricted mathematical search framework (Angelelli et al., 2010) that decomposes the problem at the binary variable level. The algorithm first solves the continuous LP relaxation of the global MILP. Variables with strictly positive fractional values are extracted to form the initial "Kernel" ($\mathcal{K}_0$). The remaining zero-valued variables are sorted by their reduced costs and divided into discrete Buckets ($\mathcal{B}_1, \dots, \mathcal{B}_m$). At each iteration $i$, a restricted MILP is solved over the variable space $\mathcal{K}_{i-1} \cup \mathcal{B}_i$, strictly enforcing $x_j = 0 \; \forall j \notin (\mathcal{K}_{i-1} \cup \mathcal{B}_i)$. Any variable from $\mathcal{B}_i$ utilized in the optimal integer solution is permanently added to the Kernel for the next iteration:  $$\mathcal{K}_i = \mathcal{K}_{i-1} \cup \left\{ j \in \mathcal{B}_i \;\middle|\; x_j^* > 0 \right\}$$

### Adaptive Kernel Search (AKS)
An advanced extension (Guastaroba et al., 2017) that dynamically adapts the kernel trajectory based on computational difficulty. AKS evaluates the runtime $t_{\text{solve}}$ required to solve the restricted MILP at the root node and classifies the specific instance bucket as EASY, NORMAL, or HARD using a defined threshold $t_{\text{easy}}$.
  To enforce strict bounds on computational time for HARD instances, AKS aggressively fixes highly promising variables derived from the LP relaxation $\tilde{x}$. Specifically, utilizing a tolerance parameter $\epsilon$ (e.g., 0.1), it forces variables near their upper bounds into the final integer solution:  $$x_j = 1 \quad \forall j \in \mathcal{K} \text{ where } \tilde{x}_j \ge 1 - \epsilon$$
  Conversely, for EASY instances, the mathematical fixing is bypassed, allowing the solver to iteratively expand the search kernel to chase higher-quality global bounds.

### Local Branching (LB)
An exact matheuristic architecture (Fischetti & Lodi, 2003) that restricts the search space by placing strict mathematical bounds on the structural Hamming distance from a known feasible incumbent solution. Let $\bar{x}$ be the binary vector of the incumbent solution, $B_1 = \{j \mid \bar{x}_j = 1\}$, and $B_0 = \{j \mid \bar{x}_j = 0\}$. The algorithm introduces an asymmetric linear constraint to force the exact MILP solver to evaluate only solutions within a neighborhood $k$:
  $$\sum_{j \in B_0} x_j + \sum_{j \in B_1} (1 - x_j) \le k$$
  By restricting $k$ (e.g., $k=10$), the solver prunes millions of nodes instantly. If the sub-MIP reaches the time limit without finding an improvement, the algorithm dynamically diversifies the search space by relaxing the neighborhood (e.g., $k \leftarrow k + 2$) or adding a hard branch ($\ge k+1$) to permanently exclude the exhausted region.

### Local Branching with Variable Neighborhood Search (LB-VNS)
A sophisticated hybrid framework (Hansen et al., 2006) that embeds the exact Local Branching solver within a Variable Neighborhood Search metaheuristic. It systematically cycles through three phases:
  1. **Shaking:** To escape deep local optima, it violently perturbs the incumbent by forcing the solver to find a starting point strictly at the boundary of neighborhood $N_k$. This is achieved by changing the LB inequality to a strict equality constraint:
     $$\sum_{j \in B_0} x_j + \sum_{j \in B_1} (1 - x_j) = k$$
  2. **Local Search:** It executes a tightly constrained standard Local Branching solve ($k_{LS} = 4$) originating from this new shaken point to seek local optimality.
  3. **Neighborhood Change:** If the resulting solution strictly dominates the global incumbent, $k$ is reset to $k_{\min}$ (intensification). Otherwise, the shaking radius is expanded $k \leftarrow k + k_{\text{step}}$ (diversification).

### Relaxation Enforced Neighborhood Search (RENS)
A primal Large Neighborhood Search variant (Berthold, 2009) that functions strictly as a start heuristic, requiring no initial feasible incumbent. RENS solves the continuous LP relaxation of the VRP model to obtain a fractional optimal point $\tilde{x}$. It then constructs an aggressive sub-MIP to explore all possible feasible roundings of this point. For purely binary routing models, it mathematically fixes all naturally integer variables to their LP bounds:
  $$x_j = \tilde{x}_j \quad \forall j \text{ s.t. } \tilde{x}_j \in \{0, 1\}$$
  The exact solver is then executed strictly on the remaining fractional variables (forcing them to $\lfloor \tilde{x}_j \rfloor$ or $\lceil \tilde{x}_j \rceil$), rapidly establishing a high-quality global upper bound.

### Cluster-First Route-Second (CF-RS)
A foundational two-stage routing paradigm (Fisher & Jaikumar, 1981) adapted for highly scalable, profit-aware logistics.
  1. **Assignment (Clustering):** The geographic space is partitioned into angular sectors, and a "seed" node is selected per sector (either by maximum demand or maximum radial distance). Unassigned nodes $i$ are then allocated to seeds $k$. This implementation supports both a high-speed greedy approximation based on insertion costs ($C_{ik} = d(0,i) + d(i,k) - d(0,k)$) and an exact Generalized Assignment Problem (GAP) formulation via MILP. The exact GAP maximizes network profit while respecting vehicle capacity $Q$:
     $$\max \sum_{i \in V} \sum_{k \in K} \left( R \cdot w_i - C \cdot C_{ik} \right) x_{ik}$$
     $$\text{s.t.} \quad \sum_{k \in K} x_{ik} \le 1 \quad \forall i, \quad \sum_{i \in V} w_i x_{ik} \le Q \quad \forall k$$
  2. **Routing (TSP):** Once clusters are mathematically locked, the intra-cluster routing is reduced to independent Traveling Salesperson Problems (TSP). These are solved using a continuous-domain metaheuristic, specifically Particle Swarm Optimization (PSO) with Enhanced Edge Recombination (EER) crossover and 2-opt mutation, which efficiently extracts the optimal intra-cluster sequence.

### Relaxation Induced Neighborhood Search (RINS)
Similar to Local Branching, RINS turns the exact MILP solver into a neighborhood explorer. Instead of utilizing Hamming distances, it fixes all variables where the continuous LP relaxation perfectly matches the integer incumbent, executing the exact solver strictly on the remaining unfixed variables. *(NOT IMPLEMENTED)*

### Restricted Master Heuristics (Heuristic Column Generation)
A framework that addresses the exponential column generation bottleneck in exact Branch-and-Price architectures. It replaces the exact subproblem (pricing) solver with high-speed metaheuristics (such as Tabu Search or ILS) to rapidly generate structurally valid columns with negative reduced costs. The exact MILP solver is then invoked exclusively on the Restricted Master Problem to optimally select combinations from this heuristically generated pool. *(NOT IMPLEMENTED)*

### Learning-to-Branch / Learning-Augmented Matheuristics
The modern frontier of routing matheuristics that integrates Machine Learning (e.g., Graph Neural Networks) directly into the Branch-and-Price architecture. Neural networks rapidly score and predict the outcome of strong branching tests, filtering out non-promising branches without the computationally exorbitant need to execute the LP solver. This ensures the algorithm traverses the search tree exponentially faster while retaining mathematical rigor. *(NOT IMPLEMENTED)*

---

## Meta-Heuristics

These are high-level frameworks used to guide the search trajectory, systematically navigating the combinatorial solution space to find high-quality solutions. They are explicitly designed to escape local optima by employing stochastic, memory-based, or population-driven mechanics. Below is a comprehensive mathematical and structural formalization of the utilized meta-heuristics.

### Adaptive Large Neighborhood Search (ALNS)
A domain-specific meta-heuristic utilizing a portfolio of ruin and recreate operators governed by a learning automaton. Let $\Omega_R$ and $\Omega_I$ be the sets of removal (destroy) and insertion (repair) operators, respectively. At each iteration, operators are selected via a roulette-wheel mechanism based on historical performance weights $w_j$. The probability of selecting operator $j$ is:
  $$p_j = \frac{w_j}{\sum_{k \in \Omega} w_k}$$
After application, the weight of the applied operator is updated using an exponential smoothing factor $\rho \in [0, 1]$ and a reward score $c$ (where $c_1 > c_2 > c_3$ for finding a new global best, a new improving solution, or an accepted deteriorating solution, respectively):
  $$w_{j, t+1} = (1 - \rho) w_{j, t} + \rho c$$
Acceptance of the newly reconstructed solution is typically governed by a Simulated Annealing criterion.

### Artificial Bee Colony (ABC)
A swarm-based meta-heuristic mimicking the foraging behavior of honey bees. The population is divided into employed bees, onlooker bees, and scouts. Let $x_{ij}$ be the $j$-th dimension of the $i$-th food source. Employed bees explore the neighborhood via the perturbation equation:
  $$v_{ij} = x_{ij} + \phi_{ij}(x_{ij} - x_{kj})$$
where $k$ is a randomly selected distinct neighbor and $\phi_{ij} \in [-1, 1]$ is a uniform random variable. Onlooker bees select food sources based on their fitness $F_i$ mapped to a probability:
  $$P_i = \frac{F_i}{\sum_{n=1}^{SN} F_n}$$
If a food source does not improve after a predefined `limit` of cycles, it is abandoned, and the employed bee becomes a scout, re-initializing the solution randomly within the domain bounds.

### Evolution Strategy Mu Plus Lambda (ES-MPL)
A canonical evolutionary meta-heuristic denoted as $(\mu + \lambda)$-ES. The algorithm maintains a parent population of size $\mu$. At each generation, it generates $\lambda$ offspring exclusively through mutation (e.g., Gaussian perturbation). The next generation's $\mu$ parents are deterministically selected from the combined pool of $\mu$ parents and $\lambda$ offspring, ensuring strict elitism:
  $$P_{t+1} \leftarrow \text{SelectBests}\left(\mu, P_t \cup \text{Offspring}_t\right)$$

### Evolution Strategy Mu, Kappa, Lambda (ES-MKL)
An evolutionary meta-heuristic introducing lifespan bounds, denoted as $(\mu, \kappa, \lambda)$-ES. It mitigates the risk of immortal local optima inherent in the "+" strategy. Each individual is assigned a maximum lifespan of $\kappa$ generations. During selection, any individual exceeding its lifespan $\kappa$ is strictly discarded, forcing the search to continuously adopt novel genetic material even if it implies a temporary deterioration in objective quality.

### Evolution Strategy Mu, Lambda (ES-MCL)
A foundational evolutionary meta-heuristic denoted as $(\mu, \lambda)$-ES. Unlike the "+" strategy, the selection phase for the next generation strictly discards the previous parents and selects the top $\mu$ individuals exclusively from the $\lambda$ offspring generated (requiring $\lambda \ge \mu$). This non-elitist approach naturally prevents the population from stagnating at a local optimum:
  $$P_{t+1} \leftarrow \text{SelectBests}\left(\mu, \text{Offspring}_t\right)$$

### Differential Evolution (DE)
A canonical population-based meta-heuristic for continuous and discrete optimization via real-key mapping. DE iteratively improves a population via difference vectors. For a target vector $x_{i,t}$, a mutant vector $v_{i,t}$ is generated by adding the scaled difference of two random population vectors to a third base vector:
  $$v_{i,t} = x_{r1,t} + F \cdot (x_{r2,t} - x_{r3,t})$$
where $F \in [0, 2]$ is the differential weight. A trial vector $u_{i,t}$ is then formed via binomial crossover with probability $CR$:
  $$u_{i,j,t} = \begin{cases} v_{i,j,t} & \text{if } \text{rand}_{i,j}(0,1) \le CR \text{ or } j = j_{rand} \\ x_{i,j,t} & \text{otherwise} \end{cases}$$
The trial vector replaces the target strictly if $f(u_{i,t}) \le f(x_{i,t})$.

### Fast Iterative Localized Optimization (FILO)
A domain-specific routing heuristic based on spatial granularity. Rather than evaluating the full objective function for every sequence perturbation, FILO operates on a spatially restricted $k$-nearest neighbor subgraph. It bounds the search trajectory by restricting the evaluation of edge exchanges strictly to geographically adjacent clusters, mathematically reducing neighborhood evaluation complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N \log K)$.

### Firefly Algorithm (FA)
A swarm meta-heuristic based on bioluminescent attraction. The attractiveness $\beta$ of a firefly diminishes with Cartesian distance $r_{ij} = \|x_i - x_j\|$. Given an absorption coefficient $\gamma$, the attractiveness is:
  $$\beta(r) = \beta_0 e^{-\gamma r_{ij}^2}$$
The movement of firefly $i$ attracted to a brighter firefly $j$ is formulated as:
  $$x_i^{t+1} = x_i^t + \beta_0 e^{-\gamma r_{ij}^2}(x_j^t - x_i^t) + \alpha(\text{rand} - 0.5)$$
where $\alpha$ is a randomization parameter bridging deterministic attraction with exploratory random walks.

### Genetic Algorithm (GA)
A canonical evolutionary meta-heuristic simulating natural selection. A population of routing solutions (chromosomes) is evaluated via a fitness function $f(x)$. Parent selection is executed via binary tournament or roulette wheel. Specialized recombination operators (e.g., Order Crossover, Edge Recombination) ensure topological validity. Mutation introduces spontaneous path swaps with probability $p_m$.

### GENIUS
A deterministic local search meta-heuristic utilizing generalized insertion (GENI) and unstringing operators. Instead of standard node insertion, GENIUS evaluates non-adjacent vertex insertions. To insert a vertex $v$ between unconnected nodes $i$ and $j$, it mathematically executes localized segment reversals, evaluating the net spatial cost using a generalized metric:
  $$\Delta C = d(i, v) + d(v, j) - d(i, k) - d(l, j) + d(k, l)$$
where $k$ and $l$ are proximal vertices in the existing tour, bypassing standard topological adjacency constraints.

### Guided Local Search (GLS)
A penalty-based trajectory meta-heuristic utilizing an augmented objective function to escape local minima. Let $f(s)$ be the baseline routing cost. GLS introduces a penalty term dynamically modified during the search. The augmented cost $h(s)$ evaluated by the local search is:
  $$h(s) = f(s) + \lambda \sum_{i=1}^{M} p_i I_i(s)$$
where $I_i(s) \in \{0, 1\}$ is an indicator function for the presence of solution feature $i$ (e.g., an expensive edge), $p_i$ is the penalty counter, and $\lambda$ is a scaling factor. Upon reaching a local optimum, the algorithm penalizes the feature $i$ exhibiting the maximum utility:
  $$\text{util}(i) = \frac{c_i}{1 + p_i}$$

### Harmony Search (HS)
A meta-heuristic isomorphic to Evolution Strategies, inspired by musical improvisation. It maintains a Harmony Memory (HM). A new solution vector $x'$ is generated component-by-component. With Harmony Memory Considering Rate (HMCR), a component is sampled from the HM; otherwise, it is generated randomly. If selected from the HM, it is perturbed by a Pitch Adjusting Rate (PAR):
  $$x_j' \leftarrow x_j' \pm bw \cdot \epsilon \quad \text{if } U(0,1) \le PAR$$
where $bw$ is the bandwidth and $\epsilon$ is a random scalar.

### Hybrid Genetic Search (HGS)
A state-of-the-art memetic algorithm combining a GA framework with aggressive local search intensification. To preserve population diversity in heavily constrained routing spaces, HGS employs a bi-criteria evaluation. A solution $p$'s Biased Fitness $BF(p)$ balances objective quality $fit(p)$ with its contribution to population diversity $\Delta(p)$ (measured via normalized Hamming distance to neighbors):
  $$BF(p) = fit(p) + \left(1 - \frac{nc_{pop}}{N}\right) \Delta(p)$$
where $nc_{pop}$ is the current number of clones in the population. Every offspring generated via crossover immediately undergoes a localized steepest-descent TSP solver before re-entering the population.

### Hybrid Genetic Search with Ruin and Recreate (HGS-RR)
An advanced memetic framework intersecting population paradigms with trajectory operators. It replaces the standard static mutation operators in HGS with dynamic ALNS-style destroy-and-repair mechanisms. This ensures that large-scale topological defects generated by crossover are violently disrupted and repaired using regret-insertion heuristics, rather than relying on weak, localized edge swaps.

### Iterated Local Search (ILS)
A canonical trajectory-based meta-heuristic defined by a four-step Markov chain structure: `GenerateInitialSolution`, `LocalSearch`, `Perturbation`, and `AcceptanceCriterion`. The trajectory continuously extracts the search from local optima $s^*$ using a violent perturbation function (e.g., random multiple edge swaps or segment relocation) to yield $s'$, followed by immediate steepest-descent local search:
  $$s'' \leftarrow \text{LocalSearch}(\text{Perturbation}(s^*))$$
Acceptance of $s''$ as the new incumbent $s^*$ is governed by thermodynamic thresholds or threshold-accepting mechanisms.

### K-Sparse Ant Colony Optimization (KS-ACO)
An algorithmic regularization of the canonical ACO swarm meta-heuristic for dense graphs. To mitigate the $\mathcal{O}(V^2)$ bottleneck of transition probability calculations, the spatial graph is pre-processed into a sparse $k$-NN graph. The probability of an ant $k$ transitioning from node $i$ to $j$ relies strictly on pheromone $\tau$ and heuristic visibility $\eta$:
  $$P_{ij}^k = \frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_{l \in N_i^k} [\tau_{il}]^\alpha [\eta_{il}]^\beta}$$
where $N_i^k$ is restricted exclusively to the $K$ nearest geometric neighbors, effectively forcing the swarm to evaluate only mathematically viable clusters.

### Particle Swarm Optimization (PSO)
A foundational continuous swarm archetype operating in a real-numbered vector space. Particles navigate by tracking their personal best position ($P_{best}$) and the global swarm best position ($G_{best}$). The velocity $v_i$ and position $x_i$ of particle $i$ at iteration $t$ are updated as:
  $$v_i^{t+1} = w v_i^t + c_1 r_1 (P_{best,i}^t - x_i^t) + c_2 r_2 (G_{best}^t - x_i^t)$$
  $$x_i^{t+1} = x_i^t + v_i^{t+1}$$
where $w$ is the inertia weight, $c_1, c_2$ are cognitive and social coefficients, and $r_1, r_2 \sim U[0,1]$. For discrete routing, positions are mapped back to permutations via Random-Key encoding.

### Particle Swarm Optimization Memetic Algorithm (PSOMA)
A hybrid meta-heuristic combining swarm mechanics with local trajectory search. Following the velocity and position updates of the standard PSO, particles mapping to discrete combinatorial tours undergo periodic Local Search refinement (e.g., 2-opt, Swap). The locally optimized vector is then inverse-mapped to update the particle's spatial coordinates, effectively transforming the PSO into an intelligent macro-level diversification controller.

### Reactive Tabu Search (RTS)
A canonical memory-based meta-heuristic extending classical Tabu Search. Instead of utilizing a static, manually tuned tabu tenure $T$, RTS tracks the search trajectory using a hash map to detect cyclic repetitions of the objective state. If a configuration $s$ is revisited, the tabu tenure is aggressively expanded ($T \leftarrow T \times \gamma$) to force structural diversification. If no cycles are detected over a defined epoch, the tenure is gradually relaxed to permit localized intensification.

### Simulated Annealing (SA)
A canonical meta-heuristic based on statistical mechanics. It permits objective deterioration to escape local minima via the Metropolis criterion. The transition probability from state $s$ to $s'$ where $\Delta E = f(s') - f(s)$ is:
  $$P(s \to s') = \begin{cases} 1 & \text{if } \Delta E \le 0 \\ \exp\left(-\frac{\Delta E}{T_k}\right) & \text{if } \Delta E > 0 \end{cases}$$
The artificial temperature $T_k$ is monotonically reduced via a cooling schedule (e.g., geometric $T_{k+1} = \alpha T_k$ with $\alpha \in [0.8, 0.99]$) until it reaches a freezing threshold, collapsing the algorithm into a pure greedy descent.

### Sine Cosine Algorithm (SCA)
A relabeled swarm meta-heuristic operating heavily on trigonometric perturbation vectors. The position of agent $i$ updating toward a destination $P_i$ utilizes switching sine and cosine functions:
  $$x_i^{t+1} = \begin{cases} x_i^t + r_1 \cdot \sin(r_2) \cdot |r_3 P_i^t - x_i^t| & \text{if } r_4 < 0.5 \\ x_i^t + r_1 \cdot \cos(r_2) \cdot |r_3 P_i^t - x_i^t| & \text{if } r_4 \ge 0.5 \end{cases}$$
where $r_1$ dictates the search radius (decaying over time), and $r_2, r_3, r_4$ dictate directional weights.

### Slack Induction by String Removal (SISR)
A domain-specific spatial meta-heuristic designed heavily for heavily constrained routing models (like Time Windows and Capacity). SISR removes sequences (strings) of vertices that are physically proximate. Let $\bar{c}$ be the average number of vertices removed per route. It selects a seed vertex and evaluates the spatial relatedness to adjacent route strings, probabilistically forcing "slack" into the network capacities to enable tight recompilation of the destroyed edges.

### Tabu Search (TS)
A foundational trajectory-based meta-heuristic employing short-term memory to escape local optima. TS maintains a Tabu List $T_L$ of recent inverse moves or solution attributes (e.g., "Edge $i \to j$ cannot be added"). At each step, it explores the neighborhood $\mathcal{N}(s)$ and selects the absolute best candidate $s^* \in \mathcal{N}(s) \setminus T_L$, even if $f(s^*) > f(s)$. An Aspiration Criterion allows the algorithm to override the tabu status if a candidate yields a historically unprecedented global best score.

### Variable Neighborhood Search (VNS)
A canonical meta-heuristic formalizing trajectory diversification through dynamically expanding neighborhood structures. It defines a set of nested neighborhoods $\mathcal{N}_k, k = 1, \dots, k_{\max}$. The algorithm executes a stochastic "Shaking" phase $s' \leftarrow \text{Random}(N_k(s))$ followed by a deterministic "Local Search" phase $s'' \leftarrow \text{LocalSearch}(s')$. If the local minimum $s''$ is strictly superior to the incumbent $s$, the search centers on $s''$ and resets to $k=1$. If not, the neighborhood radius is expanded ($k \leftarrow k+1$), enforcing wider trajectory jumps to break out of large attractor basins.

---

## Hyper-Heuristics

Hyper-heuristics operate at a higher level of abstraction than classical meta-heuristics. Rather than searching the solution space directly, they search the space of heuristics, intelligently managing a portfolio of Low-Level Heuristics (LLHs) to adapt to the problem's topological landscape during execution.

### Hidden Markov Model + Great Deluge Hyper-Heuristic (HMM-GD-HH)
A state-of-the-art online-learning selection hyper-heuristic modeled as an Input-Output Hidden Markov Model (IOHMM). It treats the underlying search space topology as a set of hidden states $S$ (e.g., improving, stagnating, escaping) that expand dynamically at a rate of $\mathcal{O}(\sqrt{\log t})$.
  The model takes LLHs as inputs $u_t$ and maps multi-objective solution changes ($\Delta$ Profit, $\Delta$ Cost) to a discrete observation alphabet $o_t$.
  1. **Action Selection:** At iteration $t$, the controller selects the LLH $u^*$ that maximizes a dynamic balance between expected normalized profit $\bar{P}(u)$ and the transition entropy $H$ (to enforce exploration), annealed by $\alpha_t$:     $$u^* = \arg\max_u \left[ \bar{P}(u) - \alpha_t \sum_{s'} P(s' | u) \ln(P(s' | u) + \epsilon) \right]$$
  2. **Belief Update:** The hidden state belief vector is updated using a Stochastic Online Expectation-Maximization (EM) approximation of the Forward algorithm:     $$b_t(j) \propto B_{u_t, j, o_t} \sum_i b_{t-1}(i) A_{u_t, i, j}$$
  3. **Acceptance:** Independent of the HMM's learning phase, candidate solutions are strictly filtered through a Great Deluge criterion adapted for maximization, where the water level $W_t$ monotonically rises:     $$f(s') \ge W_t \quad \text{where} \quad W_{t+1} = W_t + \lambda |f(s^*)|$$

### Genetic Programming Hyper-Heuristic (GPHH)
A generative hyper-heuristic that evolves a mathematical scoring function for constructive routing (heuristic generation).
  * Represented as a GP expression tree, the function evaluates candidate insertions using continuous local tactical features such as `node_profit`, `distance_to_route`, `insertion_cost`, and `remaining_capacity`.
  * To resolve the exponential computational bottleneck of full constructive evaluation, it employs a K-Nearest Neighbors (K-NN) candidate list to restrict evaluations to the spatial endpoints of existing routes, mathematically reducing the time complexity from $\mathcal{O}(N^2 R)$ to $\mathcal{O}(N K R)$.
  * Fitness is rigorously evaluated via true generalization—calculating the average normalized profit across structurally distinct spatial training environments, heavily penalized by a parsimony coefficient to mitigate tree bloat.
  * The evolutionary process employs mutually exclusive deep genetic operators (80% subtree crossover, 20% point mutation or Ephemeral Random Constant perturbation) governed by strict Koza-style depth limits.

### Guided Indicators Hyper-Heuristic (GIHH)
An adaptive selection hyper-heuristic utilizing Episodic Weight Updates via Roulette Wheel selection to manage a Multi-Objective Pareto Archive (ARCH).
  * Operator selection weights are dynamically guided by two primary performance indicators: **ScoreA** (Quality Reward), which tracks the frequency an operator's offspring successfully enters the non-dominated Pareto front, and **ScoreB** (Directional Reward), which tracks objective-space bias (e.g., favoring revenue maximization versus cost minimization).
  * The algorithm incorporates a rigorous theoretical correction for multi-objective balancing, resolving the "Equation 25 Paradox" from Chen et al. (2018).
  * It triggers directional weight updates strictly when an operator's bias is opposite to the current segment's dominant deviation, aggressively enforcing equilibrium across the Pareto frontier.
  * Additionally, structurally identical clones are explicitly rejected during archive evaluation to prevent archive explosion vulnerabilities.

### Hidden Markov Model + Great Deluge Hyper-Heuristic (HMM-GD-HH)
An online-learning selection hyper-heuristic that mathematically treats the sequence of applied Low-Level Heuristics (LLHs) as a Markov chain.
  * Rather than assigning a single deterministic state, the system maintains a probability belief distribution over three hidden states (improving, stagnating, escaping) using the Forward Algorithm.
  * At each search iteration, the belief vector $P(S_t | O_{1:t})$ is updated via the observation likelihood of the normalized profit change $\Delta_{\text{norm}}$.
  * The LLH selection probabilities are then dynamically computed as the belief-weighted mixture of the per-state emission matrices.
  * Independent of the HMM's learning phase, candidate solutions are strictly filtered through a Great Deluge acceptance criterion, which deterministically accepts candidate solutions whose profit exceeds a linearly rising water level, eliminating the need for temperature parameter tuning.

### Reinforcement Learning - Great Deluge Hyper-Heuristic (RL-GD-HH)
An adaptive parameter controller that integrates a utility-based Reinforcement Learning mechanism with a Great Deluge (GD) acceptance criterion, strictly mapping to the architecture of Ozcan et al. (2010).
  * The algorithm maintains a scalar utility $u_i$ bounded by an upper limit (e.g., $U_{\max} = 40$) for each heuristic, consistently selecting the operator with the highest utility while employing random tie-breaking.
  * It applies an additive reward ($u_i \leftarrow \min(U_{\max}, u_i + r)$) strictly upon strict objective improvement.
  * Neutral or worsening moves are explicitly punished using configurable adaptation variants: subtractive (RL1: $u \leftarrow \max(lb, u - p)$), divisional (RL2: $u \leftarrow \lfloor u/2 \rfloor$), or root (RL3: $u \leftarrow \lfloor\sqrt{u}\rfloor$).
  * Worsening candidate solutions are gated by the Great Deluge water level, which linearly updates from the initial solution quality $f_0$ toward a predefined quality lower-bound over the exact search budget.

### Sequence-based Selection Hyper-Heuristic (SS-HH)
An online-learning hyper-heuristic based on Markov chain principles that constructs and evaluates variable-length *sequences* of Low-Level Heuristics (LLHs) prior to application. Rigorously modeled after Kheiri (2014), the selection engine is driven by two dynamically updated matrices: a Transition Matrix (`TMatrix`) encoding the success probability of executing heuristic $j$ immediately after heuristic $i$, and an Acceptance-Strategy Matrix (`ASMatrix`) dictating whether to extend the current heuristic sequence ($AS=0$) or terminate and apply it to the solution ($AS=1$). Sequential transitions are sampled via roulette-wheel selection. Upon yielding a new global best solution, all involved matrix entries receive a continuous proportional reward scaled by the normalized objective improvement ($\Delta_{\text{norm}}$). Finally, candidate sequences are evaluated through a time-decaying threshold acceptance criterion, allowing temporary objective deterioration within mathematically bound limits.

### Hyper-Heuristic Ant Colony Optimization (HH-ACO)
A swarm-based meta-algorithmic controller operating strictly within the heuristic space. Instead of traversing spatial customer graphs, artificial ants construct fixed-length sequences of local search and ruin-recreate operators (e.g., 2-opt, relocate, shaw removal). The transition probability from operator $i$ to operator $j$ is governed by the Ant Colony System (ACS) pseudo-random proportional rule, mathematically balancing aggressive exploitation (via a $q_0$ probability threshold selecting $\arg\max [\tau_{ij}^\alpha \eta_{ij}^\beta]$) and proportional exploration (via roulette-wheel sampling). The underlying pheromone matrix $\tau_{ij}$ is bounded by MAX-MIN Ant System (MMAS) constraints ($\tau_{\min}$, $\tau_{\max}$) and incorporates an evaporation rate $\rho$ to continuously reinforce highly synergistic sequences of algorithmic operators.

### Hyper-heuristic with Unstringing/Stringing and Local search K-opt (HULK)
An advanced, domain-specific operator management architecture tailored for dynamic routing landscapes, mapping directly to the framework proposed by Müller and Bonilha (2022). The controller governs a tripartite pool of operators: Unstringing (generalized destruction), Stringing (generalized greedy/regret reconstruction), and Local Search (intra/inter-route $k$-opt and swaps). Operator selection is handled by an $\epsilon$-greedy Adaptive Operator Selector that tracks the running historical success (average objective score and application frequency) of each operator within a sliding memory window. The selection weights are continuously recalibrated using a dual-decay learning rate ($\text{weight} \leftarrow (1 - \text{lr}) \cdot \text{weight} \cdot \text{decay} + \text{lr} \cdot \text{avg\_score}$) to ensure the algorithm rapidly exploits high-performing operators without permanent stagnation. Acceptance of candidate moves is strictly governed by a Simulated Annealing thermodynamic criterion, which computes the transition probability $P = \exp(\Delta f / T)$ as the artificial temperature $T$ cools over successive epochs.

---
