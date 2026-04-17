# Combinatorial Optimization Operator Taxonomy

At the core of modern combinatorial optimization—particularly in complex vehicle routing and scheduling landscapes—lies a highly orchestrated ecosystem of algorithmic operators. Rather than relying on a single monolithic solver, state-of-the-art frameworks (such as LNS, ALNS, memetic algorithms, and hyper-heuristics) construct solutions by dynamically chaining together specialized functional units.

Broadly, these operators serve distinct strategic purposes in navigating the search space:
1. **Construction & Repair:** Building feasible topological states from scratch (Initialization) or intelligently restoring feasibility after massive structural disruptions (Recreate).
2. **Exploitation (Descent & Intensification):** Aggressively optimizing a localized region of the search space to reach a strict mathematical local minimum (Local Search, Improvement).
3. **Exploration (Diversification & Perturbation):** Violently dismantling or altering configurations to force the trajectory out of deep local optima basins (Destroy, Shaking, Mutation).
4. **Synthesis (Recombination):** Merging the most successful structural features (such as edge clusters or sub-routes) of multiple parent solutions into superior offspring (Crossover).
5. **Meta-Control (Orchestration):** The overarching "brain" that governs *which* operator to apply, *when*, and in what *sequence*, frequently utilizing machine learning or historical performance memory (Search Heuristics, Sequence Merging).

Understanding this taxonomy is critical for designing resilient solvers that successfully balance the fundamental optimization trade-off: the speed of local convergence versus the breadth of global exploration. Let the routing topology be defined on a graph $G = (V, E)$, with capacity bounds $Q$.

---

## Crossover (Recombination)

Crossover operators are evolutionary mechanisms that synthesize a new offspring solution by inheriting structural features from two or more parent solutions. Let parents be denoted as $P_1$ and $P_2$, and the offspring as $O$.

### Sequence-Based Crossovers

#### Ordered Crossover
Selects a contiguous subsequence of nodes from $P_1$ and copies it directly to $O$. The remaining nodes $V \setminus O$ are inserted in the exact order they appear in $P_2$, omitting duplicates. This perfectly preserves local relative ordering and guarantees Hamiltonian feasibility without generating sub-tours. Complexity: $\mathcal{O}(|V|)$.

#### Random Node Inheritance
A uniform crossover approach where each node $v_i$'s route assignment is inherited from either $P_1$ or $P_2$ based on a random indicator variable $X \sim \text{Bernoulli}(0.5)$. Because this frequently violates capacity constraints $\sum_{i \in R} q_i \le Q$, it mandates a secondary $\mathcal{O}(|V| \log |V|)$ feasibility repair phase to unassign and heuristically reinsert violating nodes.

### Structural & Partition Crossovers

#### Position Independent Crossover
Inherits elements based on their absolute assignment to specific vehicle routes rather than geometric sequence. For each route $R_k \in P_1$, a corresponding route $R_j \in P_2$ is identified. The offspring inherits the intersection of assignments $R^{O}_k = R_k \cap R_j$, which is highly effective for clustering-first, routing-second architectures.

#### Selective Route Exchange Crossover
Operates at the macroscopic route level. It identifies a subset of structurally intact, high-quality routes in $P_1$ (evaluated via density or cost-per-demand) and explicitly swaps them with geographically intersecting routes in $P_2$. Duplicate nodes are purged from the less optimal routes to restore bipartite graph validity.

#### Generalized Partition Crossover
A highly advanced operator that identifies independent sub-graphs (partitions) where the two parent tours differ. Let $G_{\Delta}$ be the symmetric difference of edges $E(P_1) \oplus E(P_2)$. The algorithm identifies mutually exclusive connected components within $G_{\Delta}$. It exhaustively evaluates and swaps the optimal partitions between parents, generating an offspring mathematically guaranteed to satisfy $f(O) \le \min(f(P_1), f(P_2))$.

#### Route Profit Generalized Partition Crossover
A variant of Generalized Partition Crossover strictly formulated for orienteering problems (OP). Instead of pure distance minimization, partitions are evaluated by their net revenue: $\Delta \text{Profit} = \sum_{i \in S} p_i - \lambda \sum_{(i,j) \in S} d_{ij}$. Sub-graphs are swapped only if the net profit scalar of the offspring strictly increases.

### Edge-Based Crossovers

#### Edge Recombination Crossover
Focuses strictly on adjacency. It builds an edge-map $M(v)$ of all neighbors for each node $v$ across both $E(P_1)$ and $E(P_2)$. It constructs $O$ by iteratively transitioning to the available neighbor in $M(v)$ that possesses the fewest unvisited neighbors itself, strongly preserving inherited geometric structures and minimizing foreign edge injections.

#### Capacity-Aware Edge Recombination Crossover
An extension of Edge Recombination that incorporates dynamic vehicle capacity limits $Q$. During the iterative neighbor selection from $M(v)$, any edge transition $(u, v)$ that would violate the route load constraint $\sum_{i \in R} q_i + q_v > Q$ is explicitly pruned from the adjacency map, structurally preventing infeasible offspring generation.

### Multi-Period Crossovers

#### Pattern and Itinerary Crossover
Designed for Periodic VRPs (PVRP). It decouples the temporal domain from the spatial domain. The offspring $O$ inherits the visit-day frequency matrices $A_{it} \in \{0,1\}$ (e.g., Monday-Thursday assignments) from $P_1$, while inheriting the spatial routing sequences (the geometric itinerary) on those specific days from $P_2$.

---

## Destroy (Ruin)

Destroy operators map a feasible solution $s$ to a partial solution $s'$ by ejecting a subset of nodes $U$ ($|U| = k$) into an unassigned pool. All implemented operators feature a standard distance/cost variant and a `_profit` variant.

### Stochastic & Cardinality-Based Removals

#### Random Removal
The baseline exploration operator. It samples $k$ nodes from $V \setminus \{0\}$ using a uniform probability mass function $P(X=v) = 1/n$. This guarantees asymptotic traversal of the entire search space and strictly prevents deterministic stalling. Complexity: $\mathcal{O}(k)$.

### Cost & Proximity-Based Removals

#### Worst Removal
Evaluates the marginal objective contribution of every active node $i$. The removal delta is $\Delta f_i = f(s) - f(s_{-i})$. In standard VRPs, this is the localized detour cost: $\Delta C_i = d_{prev_i, i} + d_{i, next_i} - d_{prev_i, next_i}$. Nodes are sorted descending by $\Delta C_i$ and removed deterministically or via a randomized rank-selection parameter $p \ge 1$ to prevent cycling.

#### Shaw Removal
Removes a cluster of mathematically "related" nodes to allow dense spatial reconstruction. The relatedness $R(i, j)$ between nodes $i$ and $j$ is calculated as a normalized weighted sum:
$$R(i,j) = \phi \cdot \frac{d_{ij}}{\max d} + \chi \cdot \frac{|t_i - t_j|}{\max \Delta t} + \psi \cdot \frac{|q_i - q_j|}{\max \Delta q}$$
An initial seed node is selected, and nodes with the lowest $R(i,j)$ relative to the active removed set $U$ are iteratively ejected.

#### Neighbor Removal
Selects a seed node $v_{seed}$ and iteratively calculates the Euclidean distance $d(v_{seed}, j)$ to all other active nodes. It strictly removes the $k$ nearest spatial neighbors, ignoring current route assignments, which effectively punches a geographic hole in the solution topology.

#### Historical Removal
Leverages a long-term search memory matrix $M_{ij}$ that tracks the frequency at which edge $(i,j)$ appears in the global best solutions. Nodes connected by edges with low historical success frequencies are heavily penalized and prioritized for ejection, explicitly guiding the search away from empirically poor structures.

#### Penalized Removal
Utilizes an augmented objective function $f'(s) = f(s) + \lambda \sum p_{ij}$ where $p_{ij}$ counts the consecutive iterations an edge has remained static in a local optimum. It deterministically targets and breaks the edges with the highest penalty weights, forcing trajectory escape.

### Macro-Structural & Geometric Removals

#### Cluster Removal
Bypasses individual node evaluation to remove entire contiguous clusters. It executes a rapid partitioning algorithm (e.g., K-means or DBSCAN) on the geographic coordinates of $V$, selects a centroid $\mu_k$, and violently ejects all nodes assigned to that cluster, forcing macroscopic topological reconstruction.

#### Route Removal
Selects a uniformly random route $R_k$ and entirely obliterates it, ejecting all constituent nodes $v \in R_k$ into $U$. This forces the algorithmic recreate phase to attempt to pack these nodes into the remaining $K-1$ fleet, acting as a strict proxy for fleet-size minimization.

#### String Removal
Removes a contiguous topological sequence (string) of vertices $\{v_i, v_{i+1}, \dots, v_{i+L}\}$ within a single route $R_k$. The length $L$ is sampled from $L \sim U(1, L_{max})$. This preserves the overarching route framework and anchor nodes while creating a localized, highly flexible gap.

#### Sector Removal
Ejects all nodes falling within a specific geometric wedge originating from the depot. Given a randomly selected seed angle $\theta_{seed}$ and a sweep radius $\Delta \theta$, any node $i$ whose polar angle $\theta_i$ satisfies $\min(|\theta_i - \theta_{seed}|, 2\pi - |\theta_i - \theta_{seed}|) \le \Delta \theta$ is ejected, cleanly severing intersecting radial routes.

### Multi-Period Removals
Specialized destructors operating in the multi-day horizon $T$, manipulating the binary visit assignment matrix $x_{ijt}$.

#### Random Horizon Removal
Flattens the entire $T$-day schedule into a one-dimensional array of $(node_i, day_t)$ tuples and uniformly samples $k$ specific visits for removal, decoupling spatial proximity from temporal disruption.

#### Worst Profit Horizon Removal
Evaluates the temporal marginal profit for every scheduled visit across $T$. The score is $P_{i,t} = p_i - C \cdot \left( d_{prev, i} + d_{i, next} - d_{prev, next} \right)$. The visits yielding the lowest net profit are ejected, freeing temporal capacity for higher-yielding alternatives.

#### Shaw Horizon Removal
A cross-temporal extension of the Shaw operator. Once a seed visit $(i, t)$ is removed, subsequent removals $(j, t')$ are sampled based on spatial proximity $d_{ij}$ and strict temporal proximity $|t - t'|$, stripping highly correlated spatiotemporal clusters from the schedule.

#### Urgency-Aware Removal
Evaluates the underlying inventory levels $I_{it}$ for each node. It identifies nodes that are strictly near their critical stock-out threshold $\tau$. The operator explicitly removes their currently scheduled visits, artificially forcing the recreate phase to schedule them earlier in the horizon to avert inventory violations.

#### Shift Visit Removal
Ejects a visit on day $t$ but mathematically restricts its unassigned state pool. It applies a hard constraint that the node must be re-inserted exclusively into day $t-1$ or $t+1$, physically sliding the schedule along the temporal axis without altering the total visit frequency.

#### Pattern Removal
Targets a specific node $i$ and entirely erases its assigned visit-frequency pattern vector $A_i = [a_{i1}, \dots, a_{iT}]$. All scheduled instances of $i$ across the entire horizon are dumped into $U$, allowing a completely unconstrained topological and temporal reschedule.

---

## Evolutionary Mutation

Mutation operators introduce localized, stochastic variations into offspring post-crossover to maintain population diversity and prevent premature genetic convergence.

### Sequence & Position Mutations

#### Swap
Selects two nodes uniformly at random from the active chromosome $s$ and strictly swaps their integer positions, creating an $\mathcal{O}(1)$ localized perturbation.

#### Inversion
Selects two random split points $p_1$ and $p_2$. The continuous substring of nodes between these indices is completely reversed. Topologically, this is equivalent to a randomized 2-opt edge exchange.

#### Scramble
Selects a continuous substring bounded by $p_1$ and $p_2$ and applies a random permutation to the internal nodes, completely destroying the localized sequence while preserving the macro-assignments.

#### Random 2-Opt
Identifies two non-adjacent edges $(u,v)$ and $(x,y)$ uniformly at random. It deletes them and establishes the single valid Hamiltonian reconnection $(u,x)$ and $(v,y)$. Feasibility regarding time windows $l_i$ is checked strictly post-mutation.

### Continuous-to-Discrete Mutations

#### Random Differential Evolution
A canonical continuous operator mapped to combinatorial integer spaces. It generates a mutant vector $v_i$ by adding the scaled difference of two random population vectors ($x_{r2}, x_{r3}$) to a base vector $x_{r1}$:
$$v_i = x_{r1} \oplus F \otimes (x_{r2} \ominus x_{r3})$$
The operators $\oplus, \ominus, \otimes$ are mapped to heuristic edge-exchange probabilities.

#### Best Differential Evolution
Structurally identical to `Random Differential Evolution`, however, the base vector $x_{r1}$ is deterministically forced to be the global best individual $x_{best}$ in the current population. This aggressively biases the mutation toward localized exploitation rather than pure stochastic exploration.

---

## Generalized Insertion and Deletion (GENI)

The GENI architecture bypasses the classical requirement that insertions and deletions must occur strictly between currently adjacent nodes. By simultaneously evaluating multiple non-adjacent topological reconnections, GENI grants access to complex sub-neighborhoods invisible to standard 2-Opt or Or-Opt moves.

### Unstringing (Generalized Deletion)
Unstringing operators systematically remove a target node $V_i$ and reconstruct the fractured route by evaluating non-adjacent inter-route connections, forcing localized sub-tour reversals.

#### Type I Unstringing
Removes node $V_i$ located between adjacent nodes $V_{i-1}$ and $V_{i+1}$. It breaks four total arcs: $(V_{i-1}, V_i)$, $(V_i, V_{i+1})$, and two non-adjacent edges $(V_j, V_{j+1})$ and $(V_k, V_{k+1})$. Feasibility is restored by inserting three new arcs: $(V_{i-1}, V_j)$, $(V_{i+1}, V_k)$, and $(V_{j+1}, V_{k+1})$. This explicitly reverses two distinct internal sub-tours: $(V_{i+1} \dots V_j) \to (V_j \dots V_{i+1})$ and $(V_{j+1} \dots V_k) \to (V_k \dots V_{j+1})$.
The objective delta is evaluated as:
$$\Delta C = \left( d_{i-1, j} + d_{i+1, k} + d_{j+1, k+1} \right) - \left( d_{i-1, i} + d_{i, i+1} + d_{j, j+1} + d_{k, k+1} \right)$$

#### Type II Unstringing
Expands the disruption magnitude by involving three non-adjacent reference nodes ($V_j, V_k, V_l$). The route is reconstructed by breaking five arcs and inserting four new inter-connections. Topologically, this creates three discrete reversed sub-tours within the host route. The complexity bounds to $\mathcal{O}(|V|^3)$ evaluations per targeted node.

#### Type III Unstringing
A direct topological inversion of Type I stringing. It removes $V_i$ and reconnects the remaining nodes utilizing two non-adjacent reference points $V_j$ and $V_k$. It establishes new links that explicitly reverse the sub-tours $(V_{i+1} \dots V_j)$ and $(V_{j+1} \dots V_k)$.

#### Type IV Unstringing
The most computationally intensive generalized deletion ($\mathcal{O}(|V|^4)$ evaluations). It isolates $V_i$ by breaking five structural arcs and utilizes four distinct reference nodes to extract two forward-oriented segments and two reversed segments, weaving them back into a single Hamiltonian cycle.

### Stringing (Generalized Insertion)
Stringing operators insert an unassigned candidate node $V_x$ into a route by executing localized segment reversals between nodes that are not currently adjacent.

#### Type I Stringing
Inserts candidate $V_x$ using reference nodes $V_i, V_j, V_k$. The operator severs three existing arcs: $(V_i, V_{i+1})$, $(V_j, V_{j+1})$, and $(V_k, V_{k+1})$. It establishes four new links to weave $V_x$ into the tour: $(V_i, V_x)$, $(V_x, V_j)$, $(V_{i+1}, V_k)$, and $(V_{j+1}, V_{k+1})$. The internal sub-tours $(V_{i+1} \dots V_j)$ and $(V_{j+1} \dots V_k)$ are directionally reversed.
$$\Delta C = \left( d_{i, x} + d_{x, j} + d_{i+1, k} + d_{j+1, k+1} \right) - \left( d_{i, i+1} + d_{j, j+1} + d_{k, k+1} \right)$$

#### Type II Stringing
Inserts $V_x$ utilizing two additional distant reference points ($V_k, V_l$). It forces the reversal of non-contiguous sub-tours $(V_{i+1} \dots V_{l-1})$ and $(V_l \dots V_j)$ to maintain strict topological feasibility while testing deeper insertion cavities.

#### Type III Stringing
Explores severe macro-level disruption. It inserts $V_x$ while simultaneously reversing almost the entire sequence of the route lying *outside* of the immediate insertion neighborhood, functionally flipping the overarching traversal direction of the graph.

#### Type IV Stringing
Performs a massive, simultaneous structural insertion involving four non-adjacent reference nodes. Because it forces multiple concurrent segment reversals, it requires verifying time-window feasibility $l_n$ for almost the entire route. Its $\mathcal{O}(|V|^4)$ complexity generally restricts its use to heavily constrained sub-problems.

---

## Improvement (Descent)

Improvement operators are functional engines strictly designed to monotonically optimize the objective function, acting as local search sinks by rejecting worsening moves ($\Delta f \ge 0$).

### Edge-Exchange Descent

#### Steepest 2-Opt
Exhaustively evaluates the entire $\mathcal{O}(|N|^2)$ localized neighborhood of all possible pairwise edge-cross removals. It caches all valid improving moves and strictly commits only to the single permutation yielding the absolute steepest mathematical gradient (the maximum cost reduction $\max |\Delta C|$).

### Node & Sequence Descent

#### Steepest Or-Opt
Iterates over all possible continuous chains of nodes (lengths $L \in \{1, 2, 3\}$). It completely extracts each chain and evaluates its re-insertion at every valid position $\mathcal{O}(|N|^2)$. It commits strictly to the maximal objective improvement, repeating until the neighborhood confirms a local minimum.

#### Steepest Node Exchange
Systematically evaluates the objective delta for all pairwise node swaps across the sequence. By iterating until $\Delta f \ge 0$ for all pairs, it forces the current topological state strictly downhill to the floor of its local basin.

---

## Intensification (Fixing)

Operators that mathematically or logically lock specific components of a solution in place, allowing exact sub-solvers to heavily exploit the restricted space.

### Exact Mathematical Formulations

#### Exact Dynamic Programming
Extracts isolated vehicle routes $R_k$ and formulates them as independent Traveling Salesperson Problems (TSPs). It exactly optimizes the internal sequence utilizing the Held-Karp dynamic programming formulation. Complexity scales strictly as $\mathcal{O}(|V_{R_k}|^2 2^{|V_{R_k}|})$, making it viable only for routes where $|V| \le 20$.

#### Fix and Optimize
A decomposition matheuristic. It permanently locks a vast subset of binary routing variables (e.g., $x_{ij} = \bar{x}_{ij}$ for structurally intact routes) and passes the remaining small pool of "free" variables to an exact MILP solver (e.g., Gurobi or CPLEX). This identifies the provable local optimum of the heavily restricted sub-MIP.

#### Set Partitioning Polish
During heuristic search, a memory universe $\Omega$ of all generated feasible routes is aggregated. The operator pauses the search and solves an exact Set Partitioning formulation to extract the optimal combination of historical routes:
$$\min \sum_{r \in \Omega} c_r y_r \quad \text{s.t.} \quad \sum_{r \in \Omega} a_{ir} y_r = 1 \quad \forall i \in V, \quad y_r \in \{0,1\}$$
This guarantees the mathematically best macro-structure achievable given the explored topology.

---

## Inter-Route Local Search

Operators that fundamentally alter macroscopic graph topology by exchanging nodes or edges between two *different* routes, explicitly balancing dynamic vehicle capacities.

### Segment & Suffix Exchanges

#### Cross Exchange
Swaps a contiguous segment of nodes $S_A$ of arbitrary length from route $R_A$ with a segment $S_B$ from route $R_B$. The internal topological sequence of both segments is strictly preserved. Feasibility requires strict adherence to capacity boundaries: $L(R_A) - \sum_{i \in S_A} q_i + \sum_{j \in S_B} q_j \le Q$.

#### Lambda Interchange
A systematic neighborhood search wrapper formulated by Osman (1993). It exhaustively explores all valid subset exchanges between pairs of routes, up to a specified maximum cardinality bounded by $\lambda_{max}$ (typically 2). Complexity per evaluation scales as $\mathcal{O}(|R_A|^\lambda \cdot |R_B|^\lambda)$.

#### Improved Cross Exchange (I-CROSS)
An advanced extension of the standard Cross Exchange that breaks the rigidness of strict sequence preservation. For every candidate segment pair $(S_A, S_B)$, I-CROSS mathematically evaluates four distinct topological configurations: 1) Standard sequence, 2) $S_A$ reversed, 3) $S_B$ reversed, and 4) Both reversed. It evaluates the steepest descent.

### Subramanian Neighborhoods

#### Shift(2,0)
Extracts a contiguous block of exactly two nodes, $B_u = (u_1, u_2)$, from source route $R_A$ and relocates it immediately after node $v$ in destination route $R_B$. Internal ordering of $B_u$ is preserved. Feasibility strictly requires $L(R_B) + q_{u_1} + q_{u_2} \le Q$. The objective cost delta assesses inter-route edge updates:
$$\Delta C = \left( d_{prev_u, next_u} + d_{v, u_1} + d_{u_2, next_v} \right) - \left( d_{prev_u, u_1} + d_{u_2, next_u} + d_{v, next_v} \right)$$

#### Swap(2,1)
A structurally asymmetrical inter-route exchange. It removes a contiguous block $B_u = (u_1, u_2)$ from $R_A$ and strictly swaps it with a single node $v$ from $R_B$. Topological validation requires verifying intermediate capacities for both routes concurrently: $L(R_A) - q_{u_1} - q_{u_2} + q_v \le Q$ and $L(R_B) - q_v + q_{u_1} + q_{u_2} \le Q$.

#### Swap(2,2)
A symmetrical macroscopic exchange. It extracts block $B_u = (u_1, u_2)$ from $R_A$ and swaps it directly with block $B_v = (v_1, v_2)$ from $R_B$. This evaluates in $\mathcal{O}(|R_A| \cdot |R_B|)$ time and is highly effective at exchanging geometrically overlapping neighborhood pairs assigned to competing vehicles.

#### Cross
Exchanges the exact suffixes of two routes starting after respective split positions $p_u \in R_A$ and $p_v \in R_B$. It severs edges $(u, u+1)$ and $(v, v+1)$, establishing crossover links $(u, v+1)$ and $(v, u+1)$. Capacity validation requires calculating the aggregate demand of the respective swapped tails:
$\Delta Q_A = \sum_{i > u} q_i$ and $\Delta Q_B = \sum_{j > v} q_j$, requiring $L(R_A) - \Delta Q_A + \Delta Q_B \le Q$.

### Multi-Route & Cascading Displacements

#### Cyclic Transfer ($p$-exchange)
Generalizes standard pairwise swaps by orchestrating a simultaneous, coordinated exchange of single nodes across $p$ distinct routes ($p \ge 3$). This effectively bypasses the multi-route capacity deadlocks that inherently stall simple 2-route operators.
Given a set of nodes $\{v_0, \dots, v_{p-1}\}$ situated in routes $\{R_0, \dots, R_{p-1}\}$, it evaluates a cyclic permutation. For a Forward Shift (where $R_i$ donates $v_i$ to $R_{(i+1) \bmod p}$):
$$\Delta C = \sum_{i=0}^{p-1} \left[ \Delta \text{Insert}(v_{(i-1) \bmod p}, R_i) - \Delta \text{Remove}(v_i, R_i) \right]$$
The multi-variable sequence is only committed if strict capacity feasibility $L_i - q_{v_i} + q_{v_{i-1}} \le Q$ is maintained across all $p$ routes concurrently.

#### Ejection Chain
An aggressive, recursive intensification mechanism mathematically designed for strict fleet-size minimization (minimizing $K$). It isolates a target `source_route` $R_s$ and attempts to completely unassign it, forcing all nodes $u \in R_s$ into the remaining active fleet.
If hard capacity bounds $Q$ prevent the direct insertion of $u$ into route $R_k$, the operator recursively displaces an existing node $v \in R_k$ to make room, provided $L_k - q_v + q_u \le Q$ holds. Node $v$ becomes an active "orphan" and triggers the chain again, searching for insertion or displacement up to a strict `max_depth`. Failure triggers a total localized state rollback to preserve global feasibility.

#### Exchange Chains
Applies explicit cardinality bounds to sequential shifts, defining exact limits for ejecting $k$ nodes into a foreign route while reciprocally receiving exactly $h$ nodes back, balancing the computational depth of standard Ejection Chains.

### Recombination & Optimal Insertion

#### K-opt Star
Recombines two distinct routes by fracturing both tours at specific split points and swapping their "tails". Topologically, the path sequence of Route A prior to the split is cleanly merged with the path sequence of Route B after the split, requiring two inter-route edge cuts.

#### Swap Star
Rather than blindly swapping node $u \in R_A$ and node $v \in R_B$ in-place (which frequently yields highly suboptimal objectives due to contextual geographic destruction), Swap Star evaluates the absolute optimal mathematical insertion position for $u$ within the entirely of $R_B$, and similarly for $v$ within $R_A$.

---

## Intra-Route Local Search

Operators bounded strictly to mutating the topological sequence *within a single isolated route* $R_k$, preventing any inter-route capacity violations.

### Edge-Exchange Neighborhoods

#### 2-Opt
Standard internal edge uncrossing. Removes two non-adjacent arcs and establishes the single valid Hamiltonian reconnection. Mathematical delta: $\Delta C = d_{i, j} + d_{i+1, j+1} - d_{i, i+1} - d_{j, j+1}$.

#### 3-Opt
Removes three distinct internal edges, fracturing the route into three sub-tours. It exhaustively evaluates all 7 possible reconnecting configurations (excluding the origin state). Because it bypasses the minima trap of 2-Opt (which cannot perform non-reversing swaps), it yields superior intensification at an $\mathcal{O}(|V|^3)$ cost.

#### K-Opt
A generalized $k$-edge exchange. As $k$ increases, the number of valid reconnections explodes exponentially. Generally limited via bounding heuristics (like $\alpha$-nearness) to maintain computational viability.

### Node-Displacement Neighborhoods

#### K-Permutation
Selects a discrete subset of $k$ nodes within the route and evaluates all $k!$ possible orderings of those specific nodes, locking the surrounding sequence infrastructure in place.

#### Relocate
Removes a single node $u$ and inserts it immediately after $v$. The local delta evaluates the subtraction of surrounding edges and the addition of insertion links:
$$\Delta C = (d_{prev_u, next_u} - d_{prev_u, u} - d_{u, next_u}) + (d_{v, u} + d_{u, next_v} - d_{v, next_v})$$

#### Relocate Chain (L3)
Extends base relocation logic to extract an unbroken contiguous sequence of $k$ nodes and structurally shift the entire block to a new valid insertion point, strictly checking for downstream time-window violations.

#### Or-Opt
An exhaustive first-improvement wrapper utilizing `Relocate Chain`. It extracts sequences of varying lengths (e.g., $k \in \{1, 2, 3\}$) and comprehensively queries the local neighborhood for improving internal permutations, repeating until $\Delta f \ge 0$.

#### Swap
Swaps the internal positions of node $u$ and node $v$. If $u$ and $v$ are adjacent, the objective evaluation must correctly account for the shared edge to prevent double-counting subtractions.

---

## Perturbation (Shaking)

Operators designed explicitly for violent diversification. Their singular mathematical goal is to push the search trajectory completely out of deep, entrenched local optima basins, frequently accepting massive objective degradation ($\Delta f \gg 0$).

### Heuristic Sequence Disruptions

#### Double Bridge
A canonical 4-opt move that severely disrupts sequence continuity. It breaks a tour into four distinct quarters and reconnects them in a jumbled order (A-D-C-B). This explicitly alters the graph topology in a way that standard 2-opt and 3-opt descent algorithms cannot mathematically reverse, ensuring genuine trajectory escape.

#### Kick
Dismantles a massive configurable fraction of the active solution (e.g., $30\%$) via completely uniform node ejection, instantly followed by a greedy repair. In the profit variant, destruction is biased strictly toward nodes with poor marginal utility: $P_u = p_u - \lambda (d_{prev, u} + d_{u, next} - d_{prev, next})$.

#### Perturb
Executes a rapid sequence of $k$ random node swaps. Crucially, it incorporates a parameter (`prob_unvisited`) that dictates the probability of swapping an active, currently routed node with a completely unassigned node $u \in U$. This structurally forces the network to map into entirely new geographic sectors.

### Exact & Evolutionary Shaking

#### Branch and Bound Perturbation
Forces an exact MILP solver to identify a mathematically feasible solution that strictly lies outside the boundaries of the current heuristic neighborhood, injecting mathematically validated but structurally novel configurations.

#### Evolutionary Perturbation
A micro-GA triggered specifically on a restricted spatial cluster of active routes. It flattens the selected routes into a single "Giant Tour" vector. It applies massive Swap mutations. For Orienteering models, it executes a "harvesting" pre-step: forcibly swapping heavily unvisited nodes with massive profit multipliers into the sequence before re-optimizing.

#### Genetic Transformation (GT)
Compares the current localized state against an archived global elite solution to extract intersecting structural edges: $E_{common} = E_{current} \cap E_{elite}$. Nodes within $E_{common}$ are mathematically "locked". All other nodes are violently ejected and stochastically reinserted, protecting historically proven macro-structures while randomizing the micro-topology.

### Multi-Period Disruptions

#### Cross-Day Shuffling
Perturbs multi-period schedules by violently shifting nodes or entire route structures across the continuous $T$-day time horizon, ignoring immediate temporal logic to force the algorithm to evaluate vastly different inventory/visit-frequency combinations.

---

## Recreate (Repair)

The structural reconstruction phase. These operators process the unassigned pool $U$ and execute topological re-insertions into the partial state $s'$ to restore feasibility $s''$.

### Myopic & Cost-Based Insertions

#### Greedy Insertion
A myopic formulation that iterates over $U$ and all valid insertion positions across all routes. It greedily commits to the insertion yielding the absolute minimum localized cost increase $\arg \min \Delta C$. Complexity per iteration: $\mathcal{O}(|U| \cdot |V_{active}|)$.

#### Deep Insertion
An aggressive load-balancing heuristic (Archetti et al.) that structurally penalizes greedy assignments that leave fragmented, empty vehicle space. Candidate evaluations use a heavily modified composite objective: $\text{Score} = \Delta C - \alpha \left( \frac{Q - L(r) - q_v}{Q} \right)$. This mathematically coerces tight packing, implicitly minimizing $K$.

### Look-Ahead & Regret Insertions

#### Regret Insertion
Calculates the mathematical opportunity cost (regret) of deferring an insertion. Let $\Delta C_{i, j}$ be the cost of inserting $i$ into its $j$-th best route. The regret explicitly prioritizes nodes that will become massively expensive if not placed optimally:
$$\text{Regret}_i = \sum_{j=2}^k (\Delta C_{i,j} - \Delta C_{i,1})$$

#### Forward-Looking Insertion
Expands evaluation beyond localized $\Delta C$ by incorporating a temporal penalty term. It checks the aggregate temporal slack $w_i$ of all downstream nodes post-insertion, penalizing moves that consume too much slack and restrict future algorithmic flexibility.

### Spatial & Geometric Insertions

#### Farthest Insertion
A spatial dispersion algorithm mapping the geographical boundaries. It strictly targets the unassigned node lying the furthest mathematical distance from any currently active route sequence. By establishing the extreme structural "skeletons" first, it prevents the algorithm from spiraling into tight, overlapping central clusters.

#### Nearest Insertion
The inverse of Farthest Insertion. It sequences the unassigned node $u \in U$ that explicitly minimizes the Euclidean distance $d_{ui}$ to the active network $V_{active}$, highly biasing the topology towards dense, localized, short-distance loops.

### Advanced & Exact Insertions

#### GENI Insertion
Rebuilds the tour utilizing GENI Type I and II stringing logic, bypassing the requirement for geometric adjacency and evaluating deeper sub-tour reversals to weave nodes into the graph.

#### Greedy Blink
A stochastic modification to strict greedy logic. During evaluation, the objectively best insertion position is deliberately ignored ("blinked") with an explicit probability $p$. This forces stochastic variance into the reconstruction sequence, preventing deterministic LNS cycling.

#### Branch and Bound Insertion
Delegates the sequence reconstruction of $U$ directly to an exact sub-solver, guaranteeing the absolute mathematically optimal partial insertion relative to the current route architectures.

---

## Search Heuristics

Macro-level algorithmic frameworks operating as the controlling "orchestrators" for the low-level functional units.

### Trajectory & Ejection Frameworks

#### Guided Ejection Search
A high-level controller specifically managing cascaded ejection chains, dynamically adjusting displacement depth limits based on the severity of capacity constraints.

#### Large Neighborhood Search
The primary orchestration loop driving Destruction $\to$ Recreation. It accepts the newly synthesized state only if it passes a rigorous Acceptance Criterion (typically Simulated Annealing or threshold bounds).

### Advanced Edge-Exchange Frameworks

#### Lin-Kernighan
A variable-depth heuristic that dynamically evaluates sequential $k$-opt exchanges based on continuous distance evaluations, structurally unbounding $k$ until local optimality is proven.

#### Lin-Kernighan-Helsgaun
The state-of-the-art exact/heuristic hybrid. It severely bounds the explosive Lin-Kernighan search space utilizing mathematically rigorous $\alpha$-nearness 1-trees, allowing the execution of extreme 5-opt and partition moves in viable computational time.

---

## Sequence Merging

Hyper-heuristic mechanisms that mutate, construct, and optimize sequences of *other algorithmic operators*, moving the optimization layer from the physical routing graph to the algorithmic logic itself.

### Probabilistic & Learning-Based Merging

#### Ant Colony Optimization Sequence
Applies ACO logic directly to operator selection. The pheromone matrix $\tau_{ij}$ represents the historically validated probability of successfully improving the objective by executing Operator $j$ immediately following Operator $i$.

#### Markov Chain Sequence
Maintains a row-stochastic transition probability matrix $T[i,j] = P(\text{Op}_j \mid \text{Op}_i)$. It mathematically samples the next heuristic operation based on the current state of the algorithmic trajectory.

#### Sequential Selection
A Hyper-Heuristic controller that dynamically tracks the real-time success rate of all Low-Level Heuristics (LLH). It utilizes reinforcement learning frameworks ($\epsilon$-greedy exploration or Softmax distribution) to dictate the execution pipeline.

### Evolutionary Merging

#### Sequence Recombination
Treats the entire sequence of executed heuristic operators as a genetic chromosome. It evolves the controller logic by applying standard sequence crossovers (OX1) and mutations to the operator chains.

---

## Solution Initialization

Foundational constructive algorithms used to build the strict initial feasible state $s_0$ from an empty $G=(V,E)$.

### Greedy & Geometric Constructors

#### Greedy
Constructs the initial sequence strictly via lowest-cost ($\min d_{ij}$) or highest-profit ($\max p_i$) sequential insertion until fleet limits are met.

#### Nearest Neighbor
Instantiates at the depot ($i=0$) and iteratively traverses strictly to the unvisited node minimizing $d_{ij}$. While computationally trivial ($\mathcal{O}(|V|^2)$), it frequently results in globally sub-optimal, deeply overlapped macro-clusters.

### Merge & Look-Ahead Constructors

#### Savings (Clarke-Wright)
A foundational merge heuristic. It begins with every node $i$ assigned strictly to an isolated back-and-forth route $(0, i, 0)$. It comprehensively evaluates the spatial savings of merging routes: $s_{ij} = d_{0,i} + d_{0,j} - d_{i,j}$. It iteratively commits to the highest savings valid under capacity bounds $Q$.

#### Regret
Applies a Regret-$k$ mathematical heuristic to the completely unassigned universe, building $s_0$ by prioritizing nodes positioned in geographically isolated areas that would incur massive penalties if deferred.

#### GRASP (Greedy Randomized Adaptive Search Procedure)
A hybrid constructive algorithm bridging deterministic logic with stochastic exploration. It strictly calculates the greedy costs for all candidate moves, constructs a Restricted Candidate List (RCL) containing only elements within a threshold $\alpha$ of the optimal, and samples the next move strictly uniformly from the RCL.
