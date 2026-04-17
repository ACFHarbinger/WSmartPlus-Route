# Routing Problem Topologies and Formulations

The domain of Combinatorial Optimization encompasses a vast hierarchy of routing paradigms. These problems dictate the spatial and temporal traversal of graphs to satisfy varying operational requirements, ranging from pure distance minimization to complex multi-period inventory synchronization, arc servicing, and facility location. Below is a comprehensive formalization of the primary routing problem classes and their mathematical variants.

## The Traveling Salesperson Problem (TSP) Family

The foundational baseline of all vehicle routing operations. It evaluates a single, uncapacitated entity navigating a fully connected graph to visit a set of nodes exactly once.

### Symmetric Traveling Salesperson Problem (STSP)
The canonical routing problem. Given a complete graph $G = (V, E)$ where the travel cost between any two nodes is symmetric ($c_{ij} = c_{ji}$), the objective is to find a Hamiltonian cycle of minimum total weight. Using binary decision variables $x_{ij} \in \{0,1\}$ indicating if edge $(i,j)$ is traversed, the Integer Linear Program (ILP) minimizes the objective $\sum_{i \neq j} c_{ij} x_{ij}$, subject to degree constraints ($\sum_{i} x_{ij} = 1$ and $\sum_{j} x_{ij} = 1$). To prevent disconnected cycles, Dantzig-Fulkerson-Johnson (DFJ) subtour elimination constraints are enforced for any proper subset $S \subset V$:
  $$\sum_{i,j \in S} x_{ij} \le |S| - 1$$

### Asymmetric Traveling Salesperson Problem (ATSP)
A directed generalization of the TSP where traversing the network is direction-dependent ($c_{ij} \neq c_{ji}$), typically reflecting real-world conditions like one-way streets or elevation changes. The mathematical formulation relies on a directed graph $G = (V, A)$, frequently requiring specialized bounding algorithms (such as the Assignment Problem relaxation) due to the inapplicability of undirected spanning tree bounds.

### Prize-Collecting TSP (PCTSP)
An economic relaxation of the mandatory visitation constraint. The solver is not required to visit all nodes. Instead, each node $i$ carries a profit $p_i$ and a penalty $\gamma_i$ if omitted. The vehicle must collect a minimum total prize subset $P_{\min}$. The objective function shifts to minimize travel costs plus the penalties of unvisited nodes:
  $$\min \sum_{(i,j) \in A} c_{ij} x_{ij} + \sum_{i \in V} \gamma_i (1 - y_i)$$
  $$\text{s.t.} \quad \sum_{i \in V} p_i y_i \ge P_{\min}$$
where $y_i \in \{0,1\}$ is the binary activation variable for node $i$.

### Traveling Salesperson Problem with Time Windows (TSPTW)
A temporal extension injecting strict service schedules. Each node $i$ must be serviced within a continuous time window $[e_i, l_i]$. Using a continuous variable $t_i$ to track the arrival time at node $i$, and service duration $s_i$, the spatial sequence is constrained by temporal logic. If edge $(i,j)$ is traversed ($x_{ij}=1$), the arrival time at $j$ is bound by the "Big-M" formulation:
  $$t_i + s_i + c_{ij} \le t_j + M(1 - x_{ij})$$
  $$e_i \le t_i \le l_i$$

---

## The Vehicle Routing Problem (VRP) Family

The VRP extends the TSP to a fleet of homogeneous or heterogeneous vehicles dispatching from a central depot to service a distributed set of customers.

### Capacitated Vehicle Routing Problem (CVRP)
The fundamental multi-vehicle architecture. A homogeneous fleet of $K$ vehicles, each with a maximum capacity $Q$, must service a set of customers $V_c$ with deterministic demands $q_i$. The objective minimizes the total fleet travel cost. The defining capacity constraint, expressed via fractional cut inequalities for any subset $S \subseteq V_c$, necessitates a minimum number of vehicles $k(S)$ to service the subset:
  $$\sum_{i \in S} \sum_{j \notin S} x_{ij} \ge 2 \left\lceil \frac{\sum_{i \in S} q_i}{Q} \right\rceil$$

### VRP with Time Windows (VRPTW)
Integrates temporal bounding into the CVRP. Like the TSPTW, nodes have distinct $[e_i, l_i]$ bounds. However, VRPTW exponentially increases state-space complexity by interleaving vehicle capacity tracking with temporal scheduling. It is predominantly solved using exact Branch-and-Price frameworks where the pricing subproblem is an Elementary Shortest Path Problem with Resource Constraints (ESPPRC), tracking time and capacity as consumed resources.

### VRP with Profits (VRPP) / Orienteering Problem (OP)
A selective routing paradigm where customer demands are replaced by economic profits, and fleet capacity is replaced by a strict distance or temporal budget $T_{\max}$. The objective pivots from cost minimization to profit maximization. For binary selection variables $y_i$, the Orienteering Problem maximizes the total collected score:
  $$\max \sum_{i \in V} p_i y_i \quad \text{s.t.} \quad \sum_{(i,j) \in A} c_{ij} x_{ij} \le T_{\max}$$

### Split Delivery VRP (SDVRP)
A relaxation of the classical CVRP where the demand $q_i$ of a single customer may exceed vehicle capacity $Q$, or it is simply economically optimal to fragment the delivery. Multiple vehicles are permitted to visit node $i$. This requires replacing the binary visitation variable with a continuous or integer flow variable $w_{ik}$, representing the quantity delivered to node $i$ by vehicle $k$:
  $$\sum_{k=1}^K w_{ik} = q_i \quad \forall i \in V_c$$

### Heterogeneous Fleet VRP (HFVRP)
Models operational reality by defining a fleet of vehicles with varying capacities $Q_k$, fixed activation costs $f_k$, and variable distance costs $c_{ij}^k$. The objective evaluates the trade-off between deploying fewer large, expensive vehicles versus numerous small, cheap vehicles.

### Periodic VRP (PVRP)
A multi-period generalization of the CVRP over a finite time horizon $H$ (e.g., a week). Each customer $i$ requires a specific service frequency $f_i$ (e.g., 2 visits per week) and provides a set of permissible visitation patterns $P_i$ (e.g., Monday-Thursday, or Tuesday-Friday). Let $y_{ip} \in \{0,1\}$ be the decision to assign customer $i$ to pattern $p$, and $a_{pt} = 1$ if pattern $p$ mandates a visit on day $t$. The assignment constraints guarantee exactly one pattern is chosen per customer:
  $$\sum_{p \in P_i} y_{ip} = 1 \quad \forall i \in V_c$$
  $$z_{it} = \sum_{p \in P_i} a_{pt} y_{ip} \quad \forall i \in V_c, \forall t \in H$$
where $z_{it}$ dictates if node $i$ must be routed on day $t$, coupling the pattern assignment strictly to the daily capacitated routing subproblems.

### Multi-Period VRP with Profits (MPVRPP)
An extension of the VRPP over a multi-period horizon $H$. Nodes accumulate profit dynamically over time or have time-dependent economic values $p_{it}$. The solver must selectively choose *which* nodes to visit and *when* to visit them to maximize cumulative profit, subject to daily routing budgets $T_{\max}$ and vehicle capacities. The objective is formalized as:
  $$\max \sum_{t \in H} \sum_{i \in V} p_{it} y_{it}$$
This paradigm heavily relies on mandatory selection heuristics and stochastic look-ahead to prevent nodes from overflowing or degrading in value while maximizing spatial synergy across operating days.

### Roll-on/Roll-off VRP (RRVRP)
A specialized domain variant prevalent in construction logistics and heavy waste management. Instead of consolidating multiple small demands, vehicles (tractors) transport massive single-unit containers (skips or roll-on/roll-off bins). A tractor can only carry one container at a time. The state space shifts from volumetric capacity tracking to discrete tractor states:
$$S \in \{\text{Empty}, \text{Carrying Empty Bin}, \text{Carrying Full Bin}\}$$

Routing involves complex sequences of dropping empty bins at customer sites, hauling full bins to disposal facilities, and returning. The node degree constraints are heavily relaxed to allow multiple visits to the same customer and disposal sites within a single tour.

---

## The Pick-up and Delivery Problem (PDP) Family

A spatial synchronization variant where freight is not delivered from a central depot, but transported between paired or mixed nodes.

### Standard Pick-up and Delivery Problem (PDP)
Every request consists of a paired pickup node $i^+$ and a delivery node $i^-$. The formulation requires strict precedence constraints ensuring the pickup occurs before delivery ($t_{i^+} \le t_{i^-}$), and coupling constraints guaranteeing that the exact same vehicle $k$ services both nodes:
  $$\sum_{j \in V} x_{i^+ j}^k = \sum_{j \in V} x_{i^- j}^k = 1 \quad \forall k$$
This necessitates dynamic, fluctuating load tracking $L_{ik}$, as the vehicle payload increases and decreases continuously along the trajectory:
  $$L_{jk} \ge L_{ik} + q_j - M(1 - x_{ij}^k)$$

### Dial-a-Ride Problem (DARP)
The human-transport variant of the PDP (e.g., paratransit, ride-sharing). In addition to paired precedence and time windows, DARP enforces strict Quality of Service (QoS) constraints for passengers. Let $L_i$ be the maximum ride time a passenger is willing to tolerate. The arrival times are mathematically bounded to prevent excessive detouring:
  $$t_{i^-} - t_{i^+} \le L_i$$
This severely restricts the feasible search space and forces the meta-heuristic to balance operational vehicle costs against passenger inconvenience penalties.

### VRP with Simultaneous Pickups and Deliveries (VRPSPD)
Unlike standard PDP, nodes are not necessarily paired. A single customer node $i$ demands a delivery quantity $d_i$ originating from the depot, and simultaneously supplies a pickup quantity $p_i$ destined for the depot. The vehicle capacity $Q$ must never be breached by the fluctuating net load at any point in the tour:
  $$\sum_{i \in S} d_i \le Q, \quad \sum_{i \in S} p_i \le Q \quad \forall S \subseteq V_c$$
The exact payload $w_{ij}$ on edge $(i,j)$ must satisfy $w_{ij} \le Q$ independently of the total tour demands.

---

## The Arc Routing Problem (ARP) Family

Problems where the operational demand is inherently continuous and located strictly along the network edges or arcs, rather than localized at discrete vertices (e.g., street sweeping, snow plowing, mail delivery).

### Chinese Postman Problem (CPP)
The foundational arc routing paradigm. Given an undirected graph $G = (V, E)$ or directed graph $G = (V, A)$, the objective is to find the minimum cost closed walk that traverses *every* edge at least once. If the graph is Eulerian (all nodes have even degree), the optimal solution equals the sum of all edge weights. If not, the problem requires finding a minimum-weight perfect matching among odd-degree nodes to determine which edges must be traversed multiple times (deadheading).

### Rural Postman Problem (RPP)
A generalization of the CPP where demand does not exist on every edge. Given a subset of required edges $E_R \subset E$, the vehicle must traverse all edges in $E_R$, but may optionally traverse non-required edges to maintain graph connectivity. This is strongly NP-hard. Let $x_{ij} \ge 0$ be the integer variable representing the number of deadheading traversals on edge $(i,j)$, and $c_{ij}$ be the cost. The objective is:
  $$\min \sum_{(i,j) \in E} c_{ij} x_{ij} + \sum_{(i,j) \in E_R} c_{ij}$$
  $$\text{s.t.} \quad \sum_{j} (x_{ij} - x_{ji}) = 0 \quad \forall i \in V \quad \text{(Flow Conservation)}$$

### Capacitated Arc Routing Problem (CARP)
Extends the RPP by introducing a homogeneous fleet of vehicles with capacity $Q$. Each required edge $e \in E_R$ possesses a demand $q_e > 0$. The total demand of edges serviced on a single vehicle's trip cannot exceed $Q$. Vehicles must start and end at a central depot. The mathematical representation utilizes binary variables $y_{ij}^k$ (indicating if vehicle $k$ services edge $(i,j)$) and integer variables $x_{ij}^k$ (indicating deadheading flow).

### Periodic CARP (PCARP)
A multi-period extension of the CARP, conceptually mirroring the PVRP. Required arcs possess service frequencies, and the fleet must be scheduled across a horizon $H$ to service the arcs while respecting daily capacity limits and allowable visitation patterns, maximizing spatial edge-synergy on active days.

---

## The General Routing Problem (GRP) Family

A unifying mathematical framework that absorbs both Node Routing (VRP/TSP) and Arc Routing (ARP/CARP).

### Standard General Routing Problem (GRP)
Given a graph $G = (V, E)$, a subset of required nodes $V_R \subseteq V$ must be visited, and a subset of required edges $E_R \subseteq E$ must be traversed. The objective minimizes the total walk distance.
* If $V_R = V$ and $E_R = \emptyset$, the GRP collapses exactly into the TSP.
* If $V_R = \emptyset$ and $E_R = E$, the GRP collapses exactly into the CPP.
* If $V_R = \emptyset$ and $E_R \subset E$, the GRP collapses exactly into the RPP.
Subtour elimination constraints must be dynamically formulated to ensure the walk forms a single connected component covering all elements in $V_R \cup E_R$.

### Capacitated General Routing Problem (CGRP)
Injects fleet capacities into the GRP. Both required nodes $v \in V_R$ and required edges $e \in E_R$ can exhibit distinct payload demands. Vehicles must route through the mixed topology without breaching capacity $Q$.

---

## The Location Routing Problem (LRP) Family

Strategic and operational decisions are intrinsically linked. LRP simultaneously determines the optimal geographic placement of facilities and the optimal VRP execution originating from those localized hubs.

### Standard Location-Routing Problem (LRP)
Given a set of candidate depot locations $D$ and customers $V_c$. Let $z_j \in \{0,1\}$ be the binary variable opening depot $j \in D$ incurring a massive fixed capital cost $F_j$. The algorithm must assign customers to open depots and sequence the routes. The objective harmonizes capital expenditure with daily routing attrition:
  $$\min \sum_{j \in D} F_j z_j + \sum_{k} \sum_{(i,j) \in A} c_{ij} x_{ij}^k$$
Subject to depot activation coupling constraints, ensuring routes only originate from open facilities:
  $$\sum_{i \in V_c} x_{ji}^k \le z_j \quad \forall j \in D, \forall k$$

### Capacitated Location-Routing Problem (CLRP)
A highly constrained variant where both the vehicles have a maximum capacity $Q$ AND the candidate depots have a maximum throughput capacity $W_j$. The total demand of all customers assigned to routes originating from depot $j$ must not exceed its structural limit. Let $y_{ij} \in \{0,1\}$ be the assignment of customer $i$ to depot $j$:
  $$\sum_{i \in V_c} q_i y_{ij} \le W_j z_j \quad \forall j \in D$$

### Multi-Echelon Location-Routing Problem (ME-LRP) / Two-Echelon VRP (2E-VRP)
An advanced logistics architecture predominantly used in urban freight distribution. It maps a multi-tier network. Freight is transported from large primary depots (1st Echelon) using massive, high-capacity vehicles to intermediate satellite facilities or cross-docks. From the satellites, a secondary fleet of smaller, agile vehicles (2nd Echelon) distributes the goods to the final customers. The model simultaneously optimizes facility locations, 1st-echelon synchronization, and 2nd-echelon routing, ensuring continuous flow conservation across the tiered graph.

---

## The Inventory Routing Problem (IRP) Family

The IRP transcends pure spatial routing by absorbing the tactical tier of Vendor-Managed Inventory (VMI). It optimizes the synchronization of vehicle routes, delivery quantities, and node-level inventory holding states over a multi-period temporal horizon $\mathcal{T}$.

### Standard Inventory Routing Problem (IRP)
A deterministic multi-period integration. For each node $i$ at time $t$, let $I_{it}$ be the inventory level, $h_i$ be the unit holding cost, $q_{it}$ be the delivered quantity, and $d_{it}$ be the deterministic consumption rate. The objective minimizes routing costs plus total network holding costs:
  $$\min \sum_{t \in \mathcal{T}} \left( \sum_{(i,j) \in A} c_{ij} x_{ijt} + \sum_{i \in V} h_i I_{it} \right)$$
The formulation is strictly governed by the inventory flow conservation constraint, ensuring the physical state evolves sequentially without breaching the maximum storage capacity $C_i$:
  $$I_{it} = I_{i, t-1} + q_{it} - d_{it} \quad \forall t \in \mathcal{T}$$
  $$0 \le I_{it} \le C_i$$

### Stochastic Inventory Routing Problem (SIRP)
A highly complex variant where the daily consumption $d_{it}$ is a random variable. To prevent stockouts under uncertainty, SIRP utilizes either robust optimization buffers or chance constraints. A chance-constrained formulation guarantees that the probability of a stockout remains below an acceptable risk threshold $\alpha$:
  $$\Pr[I_{i, t-1} + q_{it} - d_{it} < 0] \le \alpha$$
This transforms the deterministic capacity bounds into statistical confidence intervals governed by the variance of the underlying demand distribution.

### Multi-Commodity IRP (MCIRP)
An extension modeling the distribution of distinct product families $P$. It introduces a third dimension to the delivery variables ($q_{it}^p$), requiring the exact tracking of sub-compartments within the vehicle capacities and distinct depletion rates for each commodity at the client nodes.
