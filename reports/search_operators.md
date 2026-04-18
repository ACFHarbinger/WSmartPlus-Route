# Combinatorial Optimization Operator Taxonomy

At the core of modern combinatorial optimization — particularly in complex vehicle routing and scheduling landscapes — lies a highly orchestrated ecosystem of algorithmic operators. Rather than relying on a single monolithic solver, state-of-the-art frameworks (LNS, ALNS, memetic algorithms, hyper-heuristics) construct solutions by dynamically chaining together specialized functional units.

These operators serve five distinct strategic purposes:

1. **Construction and Repair** — Building feasible topological states from scratch (Initialization) or intelligently restoring feasibility after massive structural disruptions (Recreate).
2. **Exploitation (Descent and Intensification)** — Aggressively optimizing a localized region of the search space to reach a strict mathematical local minimum (Improvement).
3. **Exploration (Diversification and Perturbation)** — Dismantling or altering configurations to force the trajectory out of deep local optima basins (Destroy, Shaking, Mutation).
4. **Synthesis (Recombination)** — Merging the most successful structural features of multiple parent solutions into superior offspring (Crossover).
5. **Meta-Control (Orchestration)** — Governing *which* operator to apply, *when*, and in what *sequence*, often using machine learning or performance memory (Search Heuristics, Sequence Merging).

Let the routing topology be defined on graph $G = (V, E)$ with capacity bounds $Q$.

---

## Crossover (Recombination)

Crossover operators synthesize a new offspring solution by inheriting structural features from two or more parent solutions. Let parents be denoted $P_1$ and $P_2$ and the offspring as $O$.

### Sequence-Based Crossovers

#### Ordered Crossover (OX) ✅
Selects a contiguous subsequence from $P_1$ and copies it directly to $O$. The remaining nodes $V \setminus O$ are inserted in the exact order they appear in $P_2$, omitting duplicates. Perfectly preserves local relative ordering and guarantees Hamiltonian feasibility without generating sub-tours. Complexity: $\mathcal{O}(|V|)$.

#### Partially Mapped Crossover (PMX) ❌
Selects two random cut points defining a segment in $P_1$ which is copied directly to $O$. The remaining positions are filled from $P_2$, using a position-mapping defined by the overlapping segment to resolve duplicates. PMX preserves the absolute positions of nodes in the crossover segment while maintaining a globally valid permutation, making it particularly effective in VRP variants where position along a route is semantically meaningful (e.g., time window satisfaction).

#### Cycle Crossover (CX) ❌
Identifies cycles in the permutation mapping between $P_1$ and $P_2$. Nodes in odd cycles are inherited from $P_1$; nodes in even cycles are inherited from $P_2$. CX guarantees that every node occupies a position held by the same node in at least one parent, strictly preserving absolute positional structure. This is appropriate when the identity of the node at a specific route position matters more than the relative order of nodes.

#### Random Node Inheritance ✅
A uniform crossover where each node $v_i$'s route assignment is inherited from either $P_1$ or $P_2$ via an independent Bernoulli draw $X \sim \text{Bernoulli}(0.5)$. Capacity violations are frequent and mandate a secondary $\mathcal{O}(|V|\log|V|)$ feasibility repair phase to unassign and heuristically reinsert violating nodes.

#### Sequential Constructive Crossover (SCX) ❌
A routing-specific crossover that constructs the offspring by greedily selecting the next node from either $P_1$ or $P_2$ at each step. Starting from the first node, the algorithm looks ahead in both parents from the current position and chooses the edge leading to the unvisited node with the lower insertion cost. SCX tends to produce offspring with shorter total route cost than OX or PMX because the construction is guided by the objective function, not purely by inherited structure.

### Structural and Partition Crossovers

#### Position-Independent Crossover (PIX) ✅
Inherits elements based on their absolute assignment to specific vehicle routes rather than sequence position. For each route $R_k \in P_1$, a corresponding route $R_j \in P_2$ is identified; the offspring inherits the intersection $R^O_k = R_k \cap R_j$. Highly effective for cluster-first, route-second architectures where route membership is more stable across generations than within-route ordering.

#### Selective Route Exchange Crossover (SREX) ✅
Operates at the macroscopic route level. A subset of structurally intact, high-quality routes in $P_1$ (evaluated by density or cost-per-demand) is identified and swapped with geographically intersecting routes in $P_2$. Duplicate nodes are purged from the less optimal routes to restore bipartite graph validity.

#### Generalized Partition Crossover (GPX) ✅
A highly advanced operator that identifies independent sub-graphs (partitions) where the two parent tours differ. Let $G_\Delta$ be the symmetric difference of edges $E(P_1) \oplus E(P_2)$. The algorithm identifies mutually exclusive connected components within $G_\Delta$, exhaustively evaluates each partition, and swaps the optimal sub-graph assignments between parents. The offspring is mathematically guaranteed to satisfy $f(O) \le \min(f(P_1), f(P_2))$.

#### Route Profit — Generalized Partition Crossover ✅
A variant of GPX formulated for orienteering problems. Partitions are evaluated by net revenue: $\Delta\text{Profit} = \sum_{i \in S} p_i - \lambda\sum_{(i,j) \in S} d_{ij}$. Sub-graphs are swapped only if the net profit of the offspring strictly increases.

### Edge-Based Crossovers

#### Edge Recombination Crossover (ERX) ✅
Focuses on adjacency preservation. An edge-map $M(v)$ of all neighbors for each node $v$ across $E(P_1) \cup E(P_2)$ is constructed. The offspring $O$ is built by iteratively transitioning to the available neighbor in $M(v)$ that possesses the fewest unvisited neighbors, strongly preserving inherited geometric edge structures.

#### Capacity-Aware Edge Recombination Crossover (CA-ERX) ✅
Extends ERX by enforcing dynamic vehicle capacity limits $Q$ during construction. Any edge transition $(u,v)$ that would cause route load $\sum_{i \in R} q_i + q_v > Q$ is explicitly pruned from the adjacency map, preventing infeasible offspring generation without a post-hoc repair phase.

### Multi-Period Crossovers

#### Pattern and Itinerary Crossover ✅
Designed for Periodic VRPs (PVRP). The offspring inherits visit-day frequency matrices $A_{it} \in \{0,1\}$ (temporal schedule assignments) from $P_1$, while inheriting the spatial routing sequences (geometric itinerary) for those specific days from $P_2$. This decoupling of temporal and spatial inheritance allows independent optimization of the visit schedule and the routing topology.

---

## Destroy (Ruin)

Destroy operators map a feasible solution $s$ to a partial solution $s'$ by ejecting a subset of nodes $U$ ($|U| = k$) into an unassigned pool. All operators feature a standard distance/cost variant and a `_profit` variant where applicable.

### Stochastic and Cardinality-Based Removals

#### Random Removal ✅
The baseline exploration operator. Samples $k$ nodes from $V \setminus \{0\}$ via a uniform PMF $P(X=v) = 1/n$. Guarantees asymptotic traversal of the full search space and prevents deterministic stalling. Complexity: $\mathcal{O}(k)$.

### Cost and Proximity-Based Removals

#### Worst Removal ✅
Evaluates the marginal objective contribution $\Delta f_i = f(s) - f(s_{-i})$ of every active node. In VRPs, this is the localized detour cost: $\Delta C_i = d_{prev_i,i} + d_{i,next_i} - d_{prev_i,next_i}$. Nodes are sorted descending by $\Delta C_i$ and removed deterministically or via a randomized rank-selection parameter $p \ge 1$ to prevent cycling.

#### Shaw Removal ✅
Removes a cluster of related nodes to enable dense spatial reconstruction. Relatedness $R(i,j)$ is a normalized weighted sum:

$$R(i,j) = \phi \cdot \frac{d_{ij}}{\max d} + \chi \cdot \frac{|t_i - t_j|}{\max \Delta t} + \psi \cdot \frac{|q_i - q_j|}{\max \Delta q}$$

An initial seed node is selected; nodes with minimum $R(\cdot, j)$ relative to the active removed set $U$ are iteratively ejected.

#### Neighbor Removal ✅
Selects seed node $v_{\text{seed}}$ and removes its $k$ nearest spatial neighbors regardless of current route assignments, punching a geographic hole in the solution topology.

#### Time-Window-Based Removal ❌
A constraint-aware removal for VRPTW instances. Groups nodes by time window overlap: a node $j$ is related to seed $i$ with score $R_{tw}(i,j) = 1 / (|e_i - e_j| + |l_i - l_j| + 1)$. Removing temporally clustered nodes creates large, coherent time-window gaps that the repair operator can fill more efficiently than random or spatial removal alone, because re-inserted nodes compete for the same narrow time slots.

#### Historical Removal ✅
Leverages a long-term search memory matrix $M_{ij}$ tracking the frequency with which edge $(i,j)$ appears in global-best solutions. Nodes connected by low-historical-success edges are prioritized for ejection, guiding the search away from empirically poor structures.

#### Penalized Removal ❌
Uses an augmented objective $f'(s) = f(s) + \lambda \sum p_{ij}$ where $p_{ij}$ counts consecutive iterations an edge has remained static in a local optimum. Targets and breaks the highest-penalty edges, forcing trajectory escape.

### Macro-Structural and Geometric Removals

#### Cluster Removal ✅
Bypasses individual node evaluation. Executes a rapid partitioning (K-means or DBSCAN), selects centroid $\mu_k$, and ejects all nodes assigned to that cluster, forcing macroscopic topological reconstruction.

#### Route Removal ✅
Ejects all nodes from a uniformly selected route $R_k$, forcing the repair phase to pack them into the remaining $K-1$ fleet. Acts as a proxy for fleet-size minimization.

#### String Removal ✅
Removes a contiguous topological sequence $\{v_i, v_{i+1}, \dots, v_{i+L}\}$ within a single route $R_k$ with length $L \sim U(1, L_{\max})$, preserving the overarching route framework while creating a localized, flexible gap.

#### Sector Removal ✅
Ejects all nodes within a geometric wedge from the depot. Given a seed angle $\theta_{\text{seed}}$ and sweep radius $\Delta\theta$, node $i$ is ejected if $\min(|\theta_i - \theta_{\text{seed}}|, 2\pi - |\theta_i - \theta_{\text{seed}}|) \le \Delta\theta$, cleanly severing intersecting radial routes.

#### Zone Removal ❌
A concentric-ring variant of Sector Removal. Rather than an angular wedge, all nodes within an Euclidean annulus $[r_{\min}, r_{\max}]$ centered on a randomly selected node (not necessarily the depot) are ejected. This creates a circular gap in the solution topology — particularly useful for instances with ring-structured routes or depot-peripheral customer clusters.

### Multi-Period Removals

#### Random Horizon Removal ✅
Flattens the $T$-day schedule into $(node_i, day_t)$ tuples and uniformly samples $k$ specific visits, decoupling spatial proximity from temporal disruption.

#### Worst Profit Horizon Removal ✅
Evaluates temporal marginal profit $P_{i,t} = p_i - C(d_{prev,i} + d_{i,next} - d_{prev,next})$ for every scheduled visit. The visits with the lowest net profit are ejected, freeing temporal capacity for higher-yielding alternatives.

#### Shaw Horizon Removal ✅
Cross-temporal extension of Shaw. After a seed visit $(i, t)$ is removed, subsequent removals $(j, t')$ are sampled based on spatial proximity $d_{ij}$ and temporal proximity $|t - t'|$, stripping spatiotemporal clusters.

#### Urgency-Aware Removal ✅
Identifies nodes near critical inventory threshold $\tau$ and removes their currently scheduled visits, forcing the repair phase to reschedule them earlier to avert stockouts.

#### Shift Visit Removal ✅
Ejects a visit on day $t$ with a hard re-insertion constraint: the node must be re-inserted only into day $t-1$ or $t+1$, sliding the schedule along the temporal axis without altering visit frequency.

#### Pattern Removal ✅
Erases the full visit-frequency pattern vector $A_i = [a_{i1}, \dots, a_{iT}]$ for a target node $i$, dumping all its scheduled instances into $U$ for completely unconstrained temporal and spatial rescheduling.

---

## Evolutionary Mutation

Mutation operators introduce localized, stochastic variations into offspring post-crossover to maintain population diversity and prevent premature genetic convergence.

### Sequence and Position Mutations

#### Swap Mutation ✅
Selects two nodes uniformly at random and swaps their positions: $\mathcal{O}(1)$ localized perturbation.

#### Inversion ✅
Selects two split points $p_1, p_2$ and reverses the substring between them. Topologically equivalent to a randomized 2-opt edge exchange.

#### Scramble ✅
Selects a substring $[p_1, p_2]$ and applies a random internal permutation, destroying local sequence while preserving macro-assignments.

#### Random 2-Opt ✅
Identifies two non-adjacent edges $(u,v)$ and $(x,y)$ uniformly at random, deletes them, and reconnects as $(u,x)$ and $(v,y)$. Feasibility regarding time windows is checked post-mutation.

### Continuous-to-Discrete Mutations

#### Random Differential Evolution ✅
Maps the canonical DE mutation to combinatorial integer spaces. A mutant vector $v_i$ is generated by:

$$v_i = x_{r1} \oplus F \otimes (x_{r2} \ominus x_{r3})$$

where $\oplus, \ominus, \otimes$ are mapped to heuristic edge-exchange probabilities.

#### Best Differential Evolution ✅
Identical to Random DE, but the base vector $x_{r1}$ is forced to the global best individual $x_{\text{best}}$, aggressively biasing mutation toward exploitation.

---

## Generalized Insertion and Deletion (GENI)

The GENI architecture bypasses the requirement that insertions and deletions occur only between currently adjacent nodes. By evaluating multiple non-adjacent reconnections simultaneously, GENI accesses sub-neighborhoods invisible to standard 2-opt or Or-opt moves.

### Unstringing (Generalized Deletion)

#### Type I Unstringing ✅
Removes node $V_i$ between adjacent $V_{i-1}$ and $V_{i+1}$. Breaks four arcs and inserts three new arcs, explicitly reversing two distinct internal sub-tours:

$$\Delta C = (d_{i-1,j} + d_{i+1,k} + d_{j+1,k+1}) - (d_{i-1,i} + d_{i,i+1} + d_{j,j+1} + d_{k,k+1})$$

#### Type II Unstringing ✅
Involves three non-adjacent reference nodes, breaking five arcs and creating three reversed sub-tours. Complexity: $\mathcal{O}(|V|^3)$ per targeted node.

#### Type III Unstringing ✅
Removes $V_i$ and reconnects using two non-adjacent reference points, reversing two sub-tours while maintaining a distinct topological orientation from Type I.

#### Type IV Unstringing ✅
The most computationally intensive generalized deletion ($\mathcal{O}(|V|^4)$). Uses four reference nodes, extracting two forward and two reversed segments. Restricted to heavily constrained subproblems.

### Stringing (Generalized Insertion)

#### Type I Stringing ✅
Inserts candidate $V_x$ using reference nodes $V_i, V_j, V_k$. Severs three arcs and establishes four new links, reversing sub-tours $(V_{i+1} \dots V_j)$ and $(V_{j+1} \dots V_k)$:

$$\Delta C = (d_{i,x} + d_{x,j} + d_{i+1,k} + d_{j+1,k+1}) - (d_{i,i+1} + d_{j,j+1} + d_{k,k+1})$$

#### Type II Stringing ✅
Uses two additional distant reference points ($V_k, V_l$), forcing non-contiguous sub-tour reversals to test deeper insertion cavities.

#### Type III Stringing ✅
Explores macro-level disruption: inserts $V_x$ while reversing nearly the entire sequence outside the immediate insertion neighborhood.

#### Type IV Stringing ✅
Performs a massive simultaneous insertion involving four non-adjacent reference nodes. $\mathcal{O}(|V|^4)$ complexity restricts use to highly constrained subproblems.

---

## Improvement (Descent)

Improvement operators monotonically optimize the objective, acting as local search sinks by rejecting worsening moves ($\Delta f \ge 0$).

### Edge-Exchange Descent

#### Steepest 2-Opt ✅
Exhaustively evaluates the full $\mathcal{O}(|N|^2)$ neighborhood of pairwise edge-cross removals. Caches all valid improving moves and commits strictly to the move with the maximum cost reduction $\max|\Delta C|$.

#### 3-Opt ✅
Removes three distinct edges, fracturing the route into three sub-tours. Evaluates all 7 possible reconnecting configurations. 3-opt breaks the subset of local minima that 2-opt cannot (non-reversing configurations), at $\mathcal{O}(|N|^3)$ cost.

#### 4-Opt
Removes four edges and evaluates all valid reconnection configurations. The number of topologically distinct reconnections grows combinatorially, making exhaustive 4-opt search impractical for large routes. In practice, 4-opt is applied selectively — either via the Double-Bridge move (a specific non-sequential 4-opt configuration used in perturbation) or restricted to a small candidate set identified by the 3-opt neighborhood.

#### K-Opt (Variable Depth) ✅
A generalized $k$-edge exchange. As $k$ increases, valid reconnection count grows exponentially; bounding heuristics (such as $\alpha$-nearness candidate lists) are required to maintain computational viability.

### Node and Sequence Descent

#### Steepest Or-Opt ✅
Iterates over all continuous chains of lengths $L \in \{1, 2, 3\}$. Completely extracts each chain and evaluates re-insertion at every valid position $\mathcal{O}(|N|^2)$. Commits to the maximal improvement, repeating until the neighborhood confirms a local minimum.

#### Steepest Node Exchange ✅
Systematically evaluates the objective delta for all pairwise node swaps across the sequence. Iterates until $\Delta f \ge 0$ for all pairs, forcing the solution strictly downhill.

---

## Intensification (Fixing)

Operators that mathematically lock specific solution components in place, allowing exact sub-solvers to intensify within the restricted space.

### Exact Mathematical Formulations

#### Exact Dynamic Programming ✅
Extracts isolated vehicle routes and formulates them as independent TSPs, solved exactly via Held-Karp DP. Viable only for $|V_{R_k}| \le 20$ due to $\mathcal{O}(|V|^2 2^{|V|})$ complexity.

#### Fix and Optimize ✅
A decomposition matheuristic. A large subset of binary routing variables is locked ($x_{ij} = \bar{x}_{ij}$), and the remaining "free" variables are passed to an exact MILP solver (Gurobi, CPLEX) to find the provably optimal sub-MIP solution.

#### Set Partitioning Polish ✅
Aggregates all heuristically generated feasible routes into a memory universe $\Omega$ and solves an exact Set Partitioning formulation:

$$\min \sum_{r \in \Omega} c_r y_r \quad \text{s.t.} \quad \sum_{r \in \Omega} a_{ir} y_r = 1 \;\forall i \in V, \quad y_r \in \{0,1\}$$

---

## Inter-Route Local Search

Operators altering macroscopic graph topology by exchanging nodes or edges between different routes, explicitly balancing vehicle capacities.

### Segment and Suffix Exchanges

#### Cross Exchange ✅
Swaps a contiguous segment $S_A$ from $R_A$ with a segment $S_B$ from $R_B$, preserving internal sequence. Capacity feasibility: $L(R_A) - \sum_{i \in S_A} q_i + \sum_{j \in S_B} q_j \le Q$.

#### Lambda Interchange ✅
Exhaustively explores all valid subset exchanges between route pairs up to cardinality $\lambda_{\max}$ (Osman, 1993). Complexity: $\mathcal{O}(|R_A|^\lambda \cdot |R_B|^\lambda)$.

#### Improved Cross Exchange (I-CROSS) ✅
Extends Cross Exchange by evaluating four topological configurations for each segment pair: standard, $S_A$ reversed, $S_B$ reversed, and both reversed. Steepest descent selects the best configuration.

#### Or-opt* (Inter-Route Or-opt) ✅
The inter-route generalization of standard Or-opt. Chains of length $L \in \{1, 2, 3\}$ are extracted from route $R_A$ and evaluated for re-insertion at every position in every other route $R_B \ne R_A$, subject to capacity feasibility. This is the most frequently improving inter-route operator in CVRP local search after Node Relocation, because it allows efficient transfer of coherent node chains between routes without the geometric disruption of full segment swaps.

$$\Delta C = (d_{p_u, n_u'} - d_{p_u, u_1} - d_{u_L, n_u}) + (d_{v, u_1} + d_{u_L, n_v} - d_{v, n_v})$$

where $u_1, \dots, u_L$ is the extracted chain and $v, n_v$ is the insertion position.

### Subramanian Neighborhoods

#### Shift(2,0) ✅
Extracts a contiguous block $B_u = (u_1, u_2)$ from $R_A$ and relocates it after node $v$ in $R_B$. Objective delta:

$$\Delta C = (d_{p_u, n_u} + d_{v, u_1} + d_{u_2, n_v}) - (d_{p_u, u_1} + d_{u_2, n_u} + d_{v, n_v})$$

#### Swap(2,1) ✅
Asymmetric exchange: block $(u_1, u_2)$ from $R_A$ swapped with single node $v$ from $R_B$. Validates capacities on both routes: $L(R_A) - q_{u_1} - q_{u_2} + q_v \le Q$ and $L(R_B) - q_v + q_{u_1} + q_{u_2} \le Q$.

#### Swap(2,2) ✅
Symmetrical macroscopic exchange of blocks $(u_1, u_2)$ from $R_A$ and $(v_1, v_2)$ from $R_B$. Evaluates in $\mathcal{O}(|R_A| \cdot |R_B|)$; effective for exchanging geometrically overlapping neighborhood pairs.

#### Cross ✅
Exchanges the exact suffixes of two routes starting after split positions $p_u \in R_A$ and $p_v \in R_B$. Severs edges $(u, u+1)$ and $(v, v+1)$, establishing crossover links. Capacity validation requires computing aggregate demand of the respective swapped tails: $\Delta Q_A = \sum_{i > u} q_i$, $\Delta Q_B = \sum_{j > v} q_j$.

### Multi-Route and Cascading Displacements

#### Cyclic Transfer ($p$-exchange) ✅
Coordinates a simultaneous exchange of single nodes across $p \ge 3$ distinct routes, bypassing multi-route capacity deadlocks that stall simple 2-route operators. For a Forward Shift:

$$\Delta C = \sum_{i=0}^{p-1} \left[\Delta\text{Insert}(v_{(i-1) \bmod p}, R_i) - \Delta\text{Remove}(v_i, R_i)\right]$$

The sequence is committed only if strict capacity feasibility $L_i - q_{v_i} + q_{v_{i-1}} \le Q$ is maintained across all $p$ routes simultaneously.

#### Ejection Chain ✅
A recursive intensification mechanism for strict fleet-size minimization. Targets a `source_route` $R_s$ and attempts complete unassignment, cascading displacements through the fleet: if inserting node $u$ into route $R_k$ violates capacity, an existing node $v \in R_k$ is displaced recursively, up to `max_depth`. Failure triggers a full localized rollback.

#### Exchange Chains ✅
Applies explicit cardinality bounds to sequential shifts: ejects exactly $k$ nodes from one route while reciprocally receiving exactly $h$ nodes, bounding the computational depth of standard ejection chains.

---

## Intra-Route Local Search

Operators bounded strictly to mutating the topological sequence *within a single isolated route* $R_k$.

### Edge-Exchange Neighborhoods

#### 2-Opt ✅
Standard internal edge uncrossing. Removes two non-adjacent arcs and establishes the single valid Hamiltonian reconnection:

$$\Delta C = d_{i,j} + d_{i+1,j+1} - d_{i,i+1} - d_{j,j+1}$$

#### 3-Opt ✅
Removes three distinct internal edges, fracturing the route into three sub-tours. Exhaustively evaluates all 7 reconnecting configurations, producing strictly better local minima than 2-opt at $\mathcal{O}(|N|^3)$ cost.

#### K-Opt ✅
A generalized $k$-edge exchange. The explosive reconnection count as $k$ increases is controlled via $\alpha$-nearness candidate lists.

### Node-Displacement Neighborhoods

#### K-Permutation ✅
Selects $k$ nodes within the route and evaluates all $k!$ orderings of those nodes, holding the surrounding sequence infrastructure fixed.

#### Relocate ✅
Removes single node $u$ and inserts it immediately after $v$:

$$\Delta C = (d_{p_u, n_u} - d_{p_u, u} - d_{u, n_u}) + (d_{v, u} + d_{u, n_v} - d_{v, n_v})$$

#### Relocate Chain (L3) ✅
Extracts a contiguous sequence of $k$ nodes and shifts the entire block to a new valid insertion point, checking downstream time-window violations.

#### Or-Opt ✅
An exhaustive first-improvement wrapper using Relocate Chain with $k \in \{1, 2, 3\}$. Repeats until $\Delta f \ge 0$.

#### Swap ✅
Swaps internal positions of nodes $u$ and $v$. If adjacent, the objective evaluation accounts for the shared edge to prevent double-counting.

---

## Perturbation (Shaking)

Operators designed explicitly for diversification. Their goal is to push the search trajectory completely out of entrenched local optima basins, frequently accepting massive objective degradation ($\Delta f \gg 0$).

### Heuristic Sequence Disruptions

#### Double Bridge ✅
A canonical 4-opt move that severely disrupts sequence continuity. Breaks a tour into four quarters and reconnects in a non-sequential order (A-D-C-B). Critically, no standard 2-opt or 3-opt move can reverse this reconnection — it is *not* contained in the 2-opt or 3-opt neighborhoods — guaranteeing genuine trajectory escape rather than immediate regression to the previous local optimum.

#### Kick ✅
Dismantles a configurable fraction (e.g., 30%) of the active solution via uniform node ejection, followed by immediate greedy repair. In profit variants, destruction is biased toward nodes with poor marginal utility: $P_u = p_u - \lambda(d_{prev,u} + d_{u,next} - d_{prev,next})$.

#### Perturb ✅
Executes $k$ random node swaps. A parameter `prob_unvisited` dictates the probability of swapping a routed node with a completely unassigned node $u \in U$, structurally forcing the network into entirely new geographic sectors.

### Exact and Evolutionary Shaking

#### Branch and Bound Perturbation ✅
Directs an exact MILP solver to find a feasible solution lying strictly outside the current heuristic neighborhood, injecting mathematically validated but structurally novel configurations.

#### Evolutionary Perturbation ✅
A micro-GA triggered on a restricted spatial cluster of active routes. Flattens selected routes into a Giant Tour, applies Swap mutations, and for orienteering models, executes a "harvesting" pre-step: forcibly swapping highly profitable unvisited nodes into the sequence before re-optimization.

#### Genetic Transformation (GT) ✅
Compares the current state against an archived global-elite solution. Intersecting edges $E_{\text{common}} = E_{\text{current}} \cap E_{\text{elite}}$ are locked. All other nodes are ejected and stochastically reinserted, protecting historically proven macro-structures while randomizing the micro-topology.

### Multi-Period Disruptions

#### Cross-Day Shuffling ✅
Perturbs multi-period schedules by shifting nodes or entire route structures across the $T$-day horizon, forcing evaluation of vastly different inventory and visit-frequency combinations.

---

## Recreate (Repair)

The structural reconstruction phase. These operators process the unassigned pool $U$ and execute topological re-insertions into the partial state $s'$ to restore feasibility $s''$.

### Myopic and Cost-Based Insertions

#### Greedy Insertion ✅
Iterates over $U$ and all valid insertion positions, greedily committing to the minimum localized cost increase $\arg\min \Delta C$. Complexity per iteration: $\mathcal{O}(|U| \cdot |V_{\text{active}}|)$.

#### Deep Insertion ✅
An aggressive load-balancing heuristic (Archetti et al.) penalizing fragmented vehicle space. Candidate evaluations use a composite score:

$$\text{Score} = \Delta C - \alpha\left(\frac{Q - L(r) - q_v}{Q}\right)$$

Mathematically coerces tight packing, implicitly minimizing fleet size $K$.

#### Noise-Perturbed Insertion ❌
Adds a small stochastic noise term $\epsilon_{ij} \sim U(-\delta, \delta)$ to the insertion cost $\Delta C_{ij} + \epsilon_{ij}$ before committing. The noise breaks systematic tie-preferences between geometrically similar insertion positions, diversifying the repair output across LNS iterations without sacrificing the greedy selection structure. Particularly effective when repeated ALNS iterations produce identical repair sequences.

### Look-Ahead and Regret Insertions

#### Regret Insertion ✅
Calculates the opportunity cost of deferred insertion. Let $\Delta C_{i,j}$ be the cost of inserting node $i$ into its $j$-th best route. Nodes with maximum regret are prioritized:

$$\text{Regret}_i = \sum_{j=2}^k (\Delta C_{i,j} - \Delta C_{i,1})$$

#### Forward-Looking Insertion ✅
Expands evaluation beyond localized $\Delta C$ by incorporating a temporal penalty for downstream slack consumption. Checks the aggregate temporal slack $w_i$ of all nodes downstream of the insertion point, penalizing insertions that restrict future flexibility.

### Spatial and Geometric Insertions

#### Farthest Insertion ✅
Targets the unassigned node furthest from any currently active route sequence. Establishes extreme structural "skeletons" first, preventing tight overlapping central clusters.

#### Nearest Insertion ✅
Inserts the unassigned node $u \in U$ minimizing Euclidean distance to the active network $V_{\text{active}}$. Biases topology toward dense, short-distance loops.

### Advanced and Exact Insertions

#### GENI Insertion ✅
Rebuilds tours using GENI Type I and II stringing logic, bypassing geometric adjacency requirements and evaluating deeper sub-tour reversals to weave nodes into the graph.

#### Greedy Blink ✅
Stochastic modification to strict greedy logic. The objectively best insertion position is deliberately ignored ("blinked") with probability $p$, forcing stochastic variance into the reconstruction sequence and preventing deterministic LNS cycling.

#### Branch and Bound Insertion ✅
Delegates reconstruction of $U$ to an exact sub-solver, guaranteeing the absolute mathematically optimal partial insertion relative to the current route architectures.

---

## Search Heuristics

Macro-level frameworks controlling the sequence and orchestration of low-level functional units.

### Trajectory and Ejection Frameworks

#### Guided Ejection Search ✅
A high-level controller managing cascaded ejection chains, dynamically adjusting displacement depth limits based on the severity of capacity constraints.

#### Large Neighborhood Search ✅
The primary Destruction → Recreation orchestration loop, accepting newly synthesized states based on a pluggable Acceptance Criterion (SA, TA, GD).

### Advanced Edge-Exchange Frameworks

#### Lin-Kernighan ✅
A variable-depth heuristic dynamically evaluating sequential $k$-opt exchanges based on continuous distance evaluations, structurally unbounding $k$ until local optimality is proven.

#### Lin-Kernighan-Helsgaun (LKH) ✅
The state-of-the-art exact/heuristic hybrid. Bounds the explosive LK search space using mathematically rigorous $\alpha$-nearness 1-trees, enabling extreme 5-opt and partition moves in viable computational time.

---

## Sequence Merging

Hyper-heuristic mechanisms that mutate, construct, and optimize sequences of *other algorithmic operators*, moving the optimization layer from the physical routing graph to the algorithmic logic itself.

### Probabilistic and Learning-Based Merging

#### Ant Colony Optimization Sequence ✅
Applies ACO logic to operator selection. The pheromone matrix $\tau_{ij}$ represents the historically validated probability of improving the objective by executing Operator $j$ immediately following Operator $i$.

#### Markov Chain Sequence ✅
Maintains a row-stochastic transition matrix $T[i,j] = P(\text{Op}_j \mid \text{Op}_i)$, sampling the next operation based on the current algorithmic state.

#### Sequential Selection ✅
A hyper-heuristic controller tracking real-time LLH success rates and dictating the execution pipeline via RL frameworks ($\epsilon$-greedy or Softmax).

### Evolutionary Merging

#### Sequence Recombination ✅
Treats the entire executed operator sequence as a genetic chromosome. Evolves the controller logic by applying OX crossover and point mutations to the operator chains.

---

## Solution Initialization

Foundational constructive algorithms building the initial feasible state $s_0$ from an empty $G = (V, E)$.

### Greedy and Geometric Constructors

#### Greedy ✅
Constructs the sequence via lowest-cost ($\min d_{ij}$) or highest-profit ($\max p_i$) sequential insertion until fleet limits are met.

#### Nearest Neighbor ✅
Instantiates at the depot and iteratively traverses to the unvisited node minimizing $d_{ij}$. Computationally trivial ($\mathcal{O}(|V|^2)$) but frequently yields globally suboptimal, deeply overlapping macro-clusters.

### Merge and Look-Ahead Constructors

#### Savings (Clarke-Wright) ✅
Begins with every node assigned to an isolated back-and-forth route $(0, i, 0)$. Evaluates spatial savings $s_{ij} = d_{0,i} + d_{0,j} - d_{i,j}$ and iteratively commits to the highest savings merge valid under capacity bounds $Q$.

#### Regret ✅
Applies Regret-$k$ logic to the fully unassigned universe, prioritizing geographically isolated nodes that would incur massive penalties if deferred, building $s_0$ with a globally balanced initial topology.

#### GRASP (Greedy Randomized Adaptive Search Procedure) ✅
A hybrid constructive algorithm bridging deterministic logic with stochastic exploration. Computes greedy costs for all candidate moves, constructs a Restricted Candidate List (RCL) containing only elements within threshold $\alpha$ of the optimal, then samples the next move uniformly from the RCL. GRASP provides controllable randomization: $\alpha = 0$ recovers pure greedy; $\alpha = 1$ recovers pure random construction. GRASP solutions typically serve as high-quality starting points for subsequent local search, making it the standard initialization method in GRASP + Local Search meta-heuristic frameworks.

---
