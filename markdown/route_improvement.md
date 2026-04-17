# Route Improvement and Refinement Strategies

Route improvement algorithms operate on pre-existing, structurally feasible routing configurations to systematically reduce operational costs or maximize net profit. This taxonomy categorizes the implemented algorithms ranging from fast localized topological operators to exact mathematical decompositions and machine-learning-augmented orchestrators.

## Constructive & Augmentation Heuristics
These algorithms dynamically insert unassigned or highly profitable nodes into an active routing structure, modifying vehicle capacities and route boundaries.

### Cheapest Insertion Augmentation
A greedy constructive heuristic. For every unassigned node $k$ and valid adjacent edge $(i, j)$ in active routes, it computes the marginal insertion penalty $\Delta C = d_{i,k} + d_{k,j} - d_{i,j}$. For profit-aware environments, it evaluates net marginal profit (Revenue - $\Delta C$). The node yielding the absolute maximum economic or spatial gain is iteratively committed.

### Regret-$k$ Insertion Augmentation
An advanced constructive heuristic mitigating spatial shortsightedness. For each unassigned node $i$, it computes insertion costs across its $k$ best viable positions. The "regret" penalty for failing to insert the node into its optimal position is $\text{Regret}_i = \sum_{j=2}^k (\Delta C_{i,j} - \Delta C_{i,1})$. Nodes with maximum regret are prioritized for insertion.

### Profitable Detour Augmentation
A spatially-constrained heuristic for the Vehicle Routing Problem with Profits (VRPP). For active edge $(u, v)$ and unassigned bin $b$, if the spatial detour $\Delta d = d_{u,b} + d_{b,v} - d_{u,v}$ satisfies a proportional threshold ($\Delta d \le \epsilon \cdot d_{u,v}$), the net economic gain $\Delta \text{Profit} = (R \cdot w_b) - (C \cdot \Delta d)$ is calculated. Candidates with positive marginal utility are greedily assimilated.

### Path-Induced Synergistic Pickup
An opportunistic filter exploiting fleet physical trajectories. If an unassigned bin physically resides along the exact geometric shortest-path sequence between consecutive scheduled stops $(u, v)$, and slack capacity exists, the bin is seamlessly assimilated at zero marginal routing distance.

---

## Intra-Route Intensification (TSP Optimizers)
These exact or highly-optimized heuristic solvers treat individual vehicle routes as isolated Traveling Salesperson Problems (TSP), strictly optimizing the sequence of visits without altering route assignments.

### Steepest 2-opt Refinement
A deterministic intensification operator. It exhaustively maps the 2-opt neighborhood (edge crossing removals) for every individual route independently. Executing a steepest-descent trajectory, it commits to the permutation yielding the absolute maximum cost reduction until a strict intra-route local minimum is reached.

### Dynamic Programming (DP) Route Reoptimization
An exact intra-route solver utilizing the foundational Held-Karp dynamic programming algorithm. It guarantees the absolute optimal sequence for any subset of nodes assigned to a vehicle. Given its $\mathcal{O}(|V|^2 2^{|V|})$ complexity, it is strictly gated by a hard cardinality limit (e.g., $|V| \le 20$).

### Fast TSP Refinement
A high-speed, scalable local search heuristic. It systematically isolates individual routes, strips depot connections, and delegates the isolated clusters to a highly optimized C++ TSP backend, ensuring each vehicle sequence is locally optimal before evaluating complex inter-route exchanges.

### Lin-Kernighan-Helsgaun (LKH) Refinement
The premier state-of-the-art heuristic for the TSP. It dynamically constructs highly complex, variable $k$-opt sequential exchanges. To constrain the exponential search space, the Helsgaun variant restricts evaluations to the algorithmic $\alpha$-nearness 1-tree, heavily prioritizing highly probable edges.

---

## Inter-Route & Topological Local Search
These algorithms operate across multiple vehicle routes simultaneously, performing complex structural recombinations to balance load distributions and optimize spatial boundaries.

### Classical Local Search
A deterministic trajectory-based refinement wrapping a suite of fundamental operators (e.g., 3-opt, 4-opt, swap-star). It systematically evaluates localized edge permutations and drives the multi-route solution to a strict local minimum via iterative or steepest-descent improvement.

### Steepest Node Exchange
An exhaustive steepest-descent operator computing the exact objective delta for all possible pairwise spatial swaps ($i \leftrightarrow j$) across both intra- and inter-route neighborhoods, irrevocably committing to the single optimal permutation per iteration.

### Or-opt Refinement (Steepest & Iterative)
An aggressive generalization of localized routing. It models the extraction and optimal re-insertion of contiguous sequence chains (lengths $L \in \{1, 2, 3\}$). The steepest variant rigorously maps the entire neighborhood before execution, while the iterative variant utilizes a first-improvement trajectory for rapid convergence.

### Cross-Exchange Local Search
An advanced inter-route trajectory operator generalizing swaps and 2-opt* moves. It exchanges a contiguous segment (bounded by length $L$) from route $r_1$ with a segment from $r_2$, effectively breaking and reconnecting four distinct edges simultaneously to facilitate massive structural recombinations.

---

## Trajectory Meta-Heuristics
These frameworks introduce stochasticity or memory structures to accept objective deterioration, deliberately navigating the search trajectory out of deep local optima basins.

### Simulated Annealing (SA) Refinement
A stochastic trajectory meta-heuristic utilizing topological perturbations. Move acceptance is strictly governed by the Boltzmann-Metropolis thermodynamic criterion ($P = \exp(\Delta f / T)$), permitting controlled objective deterioration as an artificial temperature parameter cools over successive epochs.

### Guided Local Search (GLS)
A penalty-based meta-heuristic. It monitors the sequence of edges utilized in incumbent solutions and penalizes edges that are frequently traversed but globally suboptimal. Operators evaluate moves based on an augmented utility function: $U(i, j) = c_{ij} / (1 + p_{ij})$, forcing diversification into unexplored topological regions.

### Randomized Local Search
A stochastic search bypassing exhaustive steepest descent. It utilizes a predefined probability distribution to randomly sample and apply an operator (e.g., Or-opt, cross-exchange) at each iteration, introducing chaotic permutations to escape shallow local optima.

---

## Large Neighborhood Search (Destroy & Repair)
Algorithms that aggressively dismantle substantial portions of the routing topology and intelligently reconstruct them to bridge disparate regions of the solution space.

### Ruin and Recreate (LNS)
A foundational destroy-and-repair meta-heuristic. It disrupts the incumbent by applying spatial or randomized removal operators, subsequently rebuilding via greedy or regret insertion. Acceptance of candidate states is controlled via pluggable thermodynamic criteria to decouple search trajectories from strict monotonic improvement.

### Adaptive Large Neighborhood Search (ALNS)
A highly sophisticated meta-heuristic where operator selection is governed dynamically by a Multi-Armed Bandit utilizing Thompson Sampling. The bandit maintains a Bayesian posterior of success for each ruin (e.g., Shaw, worst) and recreate operator, biasing selection toward pairs historically yielding the highest objective improvements.

---

## Exact & Matheuristic Solvers
Hybrid architectures combining the speed of local search with the rigorous mathematical bounding capabilities of Mixed-Integer Linear Programming (MILP).

### Set-Partitioning (Pool-Restricted Exact)
A matheuristic that automatically constructs a massive pool of candidate routes through diverse generation strategies (e.g., LNS perturbations, Held-Karp sequence optimization, mandatory singletons). Following canonical deduplication, a restricted Set Partitioning MILP model is solved to global optimality over the pool.

### Set-Partitioning Polish
An exact mathematical refinement that acts strictly on a pre-supplied pool of high-quality routes, utilizing a commercial MILP solver (e.g., Gurobi) to extract the optimal combination of routes while enforcing capacity and visit constraints.

### Branch-and-Price (B&P) Refinement
An exact column-generation matheuristic applied as a localized improver. It formulates the routing state as a Set Partitioning Master Problem. To iteratively populate the Restricted Master Problem with improving routes, it solves an Exact Resource-Constrained Shortest Path Problem (RCSPP) utilizing $ng$-route relaxation and Lagrangian bounding, guaranteeing integrality via Ryan-Foster branching.

### Fix-and-Optimize Matheuristic
A decomposition-based exact strategy. It permanently "fixes" (locks) a subset of high-performing routes, while nodes belonging to the remaining "free" routes are passed to an MILP solver. The solver exactly re-optimizes this restricted sub-MIP, circumventing the intractability of solving the global graph simultaneously.

---

## Neural & Meta-Algorithmic Orchestrators
Advanced algorithmic management systems that sequence operators or utilize deep learning to map the combinatorial search space.

### Learned Route Improver (Neural Operator)
A machine-learning-augmented operator utilizing a pre-trained Graph Neural Network (GNN) to bypass the $\mathcal{O}(n^k)$ computational bottleneck of exhaustive $k$-opt evaluations. Using multi-layer perceptron (MLP) node/edge encoders, a neural "move head" directly predicts expected objective improvements of topological permutations, executing the highest-scoring moves sequentially.

### Multi-Phase Composition
A meta-algorithmic architecture pipelining multiple distinct refinement strategies into a sequential execution graph. It constructs complex life-cycles—e.g., executing a greedy augmentation phase, followed by inter-route SA local search, concluding with an exact LKH polish—while tracking granular metrics across discrete transitions.

---
