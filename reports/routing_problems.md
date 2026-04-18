# Routing Problem Topologies and Formulations

The domain of combinatorial optimization encompasses a vast hierarchy of routing paradigms. These problems govern the spatial and temporal traversal of graphs to satisfy varying operational requirements, ranging from pure distance minimization to complex multi-period inventory synchronization, arc servicing, and facility location. The problems below are organized by the structural feature that distinguishes each family — whether the decision unit is a node, an arc, or a mixed topology — and are progressively refined by layering real-world constraints (capacity, time windows, stochasticity, profits) onto the base formulation.

---

## The Traveling Salesperson Problem (TSP) Family

The foundational baseline for all vehicle routing. A single uncapacitated agent navigates a fully connected graph to visit a set of nodes subject to varying structural constraints.

### Symmetric TSP (STSP)
The canonical routing problem. Given a complete undirected graph $G = (V, E)$ where $c_{ij} = c_{ji}$, the objective is to find the minimum-cost Hamiltonian cycle. Using binary edge variables $x_{ij} \in \{0,1\}$, the ILP minimizes $\sum_{i \neq j} c_{ij} x_{ij}$ subject to degree constraints ($\sum_j x_{ij} = 2$ for all $i$). Subtour elimination is enforced via Dantzig-Fulkerson-Johnson (DFJ) constraints for every proper subset $S \subset V$:

$$\sum_{i \in S,\, j \notin S} x_{ij} \ge 2$$

### Asymmetric TSP (ATSP)
A directed generalization where $c_{ij} \neq c_{ji}$, modelling one-way streets, elevation asymmetry, or time-dependent travel. Defined on a directed graph $G = (V, A)$. The LP relaxation is tightened by the Assignment Problem bound rather than spanning tree bounds, and subtour elimination requires directed cut constraints. Frequently solved via transformation to STSP using a gadget that replaces each node with a pair.

### Generalized TSP (GTSP)
Partitions the node set into $m$ disjoint clusters $C_1, \dots, C_m$. The vehicle must visit exactly one node from each cluster, minimizing the total inter-cluster tour cost. GTSP generalizes TSP ($m = |V|$, singleton clusters) and provides a natural model for location selection combined with routing. It is typically solved by transformation to an equivalent ATSP.

### Prize-Collecting TSP (PCTSP)
A selective visitation model. Each node $i$ carries a profit $p_i$ and a penalty $\gamma_i$ if omitted. The vehicle must collect at least a minimum total prize $P_{\min}$. The objective minimizes travel cost plus penalties for unvisited nodes:

$$\min \sum_{(i,j) \in A} c_{ij} x_{ij} + \sum_{i \in V} \gamma_i (1 - y_i) \quad \text{s.t.} \quad \sum_{i \in V} p_i y_i \ge P_{\min}$$

where $y_i \in \{0,1\}$ activates node $i$.

### TSP with Time Windows (TSPTW)
Injects strict service schedules. Each node $i$ must be serviced within a hard time window $[e_i, l_i]$. Arrival time $t_i$ at node $i$ satisfies:

$$t_i + s_i + c_{ij} \le t_j + M(1 - x_{ij}), \quad e_i \le t_i \le l_i$$

where $s_i$ is the service duration. Soft time windows replace hard bounds with penalty terms, allowing violations at a cost.

### Clustered TSP (CTSP)
Requires the vehicle to complete a full visit to every node within each cluster before departing to the next cluster. Unlike GTSP, all nodes are mandatory. The CTSP decomposes into two sequential decisions: cluster visitation order (an inter-cluster TSP) and intra-cluster sequencing (an independent TSP per cluster).

---

## The Vehicle Routing Problem (VRP) Family

The VRP extends the TSP to a fleet of vehicles dispatching from one or more depots to service a distributed customer set. The family spans deterministic to stochastic, static to dynamic, and cost-minimizing to profit-maximizing formulations.

### Capacitated VRP (CVRP)
The fundamental multi-vehicle routing problem. A homogeneous fleet of $K$ vehicles, each with capacity $Q$, services customers $V_c$ with deterministic demands $q_i$. Capacity is enforced via fractional cut inequalities: for any subset $S \subseteq V_c$, the minimum number of vehicles $\lceil \sum_{i \in S} q_i / Q \rceil$ must cross the cut:

$$\sum_{i \in S,\, j \notin S} x_{ij} \ge 2 \left\lceil \frac{\sum_{i \in S} q_i}{Q} \right\rceil$$

### VRP with Time Windows (VRPTW)
Integrates customer-specific time windows $[e_i, l_i]$ into the CVRP. Each vehicle simultaneously tracks elapsed time and remaining capacity as consumed resources along its route, exponentially increasing state-space complexity. The standard exact approach is Branch-and-Price, where the pricing subproblem is an Elementary Shortest Path Problem with Resource Constraints (ESPPRC) tracking time and load jointly.

### Open VRP (OVRP)
A variant where vehicles are not required to return to the depot after completing their route. This models scenarios where drivers finish their shifts at the last customer site or vehicles are stationed regionally. Mathematically, depot-return arcs are removed from the feasible arc set, and the route cost accounts only for the outbound path. The OVRP is structurally equivalent to a CVRP in which the return arc costs are zeroed out, but optimal solutions differ substantially in topology.

### Multi-Depot VRP (MDVRP)
Extends the CVRP to $D$ geographically distinct depots, each with its own fleet and potentially distinct vehicle types. Each customer must be assigned to exactly one depot, and the routes from that depot must remain feasible under its fleet capacity. The assignment-routing coupling makes MDVRP significantly harder than CVRP: let $z_{id} \in \{0,1\}$ be the assignment of customer $i$ to depot $d$, then:

$$\sum_{d \in D} z_{id} = 1 \; \forall i \in V_c, \quad \sum_{i \in V_c} q_i z_{id} \le Q_d \; \forall d \in D$$

### VRP with Backhauls (VRPB)
Partitions customers into linehaul nodes $L$ (requiring outbound delivery from depot) and backhaul nodes $B$ (requiring inbound pickup to depot). A classical precedence constraint mandates that all linehaul customers on a route are serviced before any backhaul customer, reflecting vehicle loading constraints. The mixed variant (VRPMB) relaxes this ordering. Capacity tracking requires monitoring the net vehicle load as it decreases through linehaul stops and re-increases through backhaul stops.

### VRP with Profits (VRPP) / Orienteering Problem (OP)
A selective routing paradigm where nodes carry economic profits and visitation is optional. The fleet operates under a strict temporal or distance budget $T_{\max}$ per vehicle. The objective pivots from cost minimization to profit maximization:

$$\max \sum_{i \in V} p_i y_i \quad \text{s.t.} \quad \sum_{(i,j) \in A} c_{ij} x_{ij} \le T_{\max}, \quad y_i \le \sum_j x_{ji}$$

The single-vehicle variant is the Orienteering Problem; the multi-vehicle version is the Team Orienteering Problem (below).

### Team Orienteering Problem (TOP)
The multi-vehicle extension of the Orienteering Problem. A fleet of $K$ vehicles, each with individual budget $T_{\max}^k$, collectively maximizes total collected profit. Each node may be visited at most once across the entire fleet. The profit of a node is collected only once, even if multiple vehicles pass nearby:

$$\max \sum_{i \in V} p_i y_i \quad \text{s.t.} \quad \sum_{(i,j)} c_{ij} x_{ij}^k \le T_{\max}^k \;\forall k, \quad \sum_k y_{ik} \le 1 \;\forall i$$

TOP directly models multi-vehicle selective collection with shared profit budgets.

### Split Delivery VRP (SDVRP)
Relaxes the single-vehicle-per-customer constraint of classical CVRP. A customer demand $q_i > Q$ may be split across multiple vehicles, or splitting may be economically optimal even when $q_i \le Q$. Flow variables $w_{ik}$ represent quantities delivered by vehicle $k$:

$$\sum_{k=1}^K w_{ik} = q_i \; \forall i \in V_c, \quad \sum_{i \in R_k} w_{ik} \le Q \; \forall k$$

### Heterogeneous Fleet VRP (HFVRP)
Models a realistic fleet with varying vehicle types, each with distinct capacity $Q_k$, fixed activation cost $f_k$, and variable arc cost $c_{ij}^k$. The objective balances fixed fleet deployment costs against variable routing costs, creating a coupled vehicle assignment and routing problem.

### Periodic VRP (PVRP)
A multi-period generalization over horizon $H$. Each customer $i$ requires service frequency $f_i$ times per period and selects from a set of allowable visit-day patterns $P_i$. Let $y_{ip} \in \{0,1\}$ activate pattern $p$ for customer $i$:

$$\sum_{p \in P_i} y_{ip} = 1 \;\forall i, \quad z_{it} = \sum_{p \in P_i} a_{pt} y_{ip}$$

where $z_{it}$ drives the daily capacitated CVRP subproblem on day $t$.

### Multi-Period VRP with Profits (MPVRPP)
Combines the PVRP with profit-based selective visitation over horizon $H$. Nodes accumulate profit dynamically — either through time-varying values $p_{it}$ or physical fill-level accumulation. The solver jointly decides *which* nodes to visit and *when*, maximizing cumulative profit subject to daily fleet budgets and capacity:

$$\max \sum_{t \in H} \sum_{i \in V} p_{it} y_{it}$$

This paradigm is the direct routing counterpart of the Inventory Routing Problem and relies heavily on mandatory selection heuristics and stochastic look-ahead.

### Roll-on/Roll-off VRP (RRVRP)
A specialized variant for heavy waste logistics. Vehicles (tractors) transport single large containers (skips); a tractor carries exactly one container at a time. The state space shifts from volumetric tracking to discrete tractor states:

$$S \in \{\text{Empty}, \text{Carrying Empty Container}, \text{Carrying Full Container}\}$$

Routing consists of complex depot-customer-disposal sequences, and node degree constraints are relaxed to permit multiple visits to both customer sites and disposal facilities within a single tour.

### Green VRP / Electric VRP (GVRP / EVRP)
Extends the CVRP with energy consumption constraints. In the EVRP, each vehicle has a battery capacity $E$ that depletes along arcs proportionally to distance and load, and must be replenished at recharging stations $F \subset V$. Decision variables track both routing $x_{ij}^k$ and battery state $e_{ik}$ at each node. The non-linear energy depletion function — often depending on vehicle speed, gradient, and payload — is typically linearized or approximated. The GVRP uses fuel or emissions budgets in place of electric charge.

---

## Stochastic and Dynamic VRP Extensions

These variants inject uncertainty or real-time dynamism into the deterministic VRP framework. They are particularly important for multi-period operational routing where demand is observed only gradually.

### Stochastic VRP (SVRP)
Demands $q_i$ and/or customer presence $y_i$ are random variables revealed only upon vehicle arrival. The canonical model is the VRP with Stochastic Demands (VRPSD): if a customer's realized demand exceeds remaining vehicle capacity upon arrival, the vehicle must make a detour back to the depot (a *failure*). The objective minimizes expected total routing cost including the expected failure penalty:

$$\min \sum_{(i,j)} c_{ij} x_{ij} + \mathbb{E}\left[\text{Failure Cost}(x, \tilde{q})\right]$$

Solved via stochastic programming, sample average approximation (SAA), or dynamic programming over demand distributions.

### Dynamic VRP (DVRP)
Customer requests arrive online during the execution horizon, rather than being known in advance. The solver must respond in real time, re-routing the active fleet to incorporate new requests while honoring the service commitments already made. The degree of dynamism is quantified as the fraction of requests that arrive after dispatch. DVRP is typically modelled as a rolling-horizon MDP, with re-optimization triggered at fixed intervals or upon significant new request clusters.

### Robust VRP (RVRP)
A worst-case formulation that optimizes against an adversarial uncertainty set $\mathcal{U}$ for demands or travel times, rather than an assumed distribution. The minimax formulation guarantees feasibility for every realization within $\mathcal{U}$:

$$\min_x \max_{q \in \mathcal{U}} \left\{ \text{Cost}(x, q) \;\middle|\; \text{Constraints}(x, q) \right\}$$

Ellipsoidal and box uncertainty sets yield tractable second-order cone or robust linear programs; budget uncertainty sets (Bertsimas-Sim) yield linear reformulations directly.

---

## The Pickup and Delivery Problem (PDP) Family

A synchronization variant where freight originates and terminates at arbitrary customer nodes rather than at a central depot.

### Standard PDP
Each request is a paired pickup–delivery tuple $(i^+, i^-)$. Precedence constraints enforce $t_{i^+} \le t_{i^-}$ (pickup before delivery), and coupling constraints require the same vehicle $k$ to service both endpoints:

$$\sum_j x_{i^+j}^k = \sum_j x_{i^-j}^k \; \forall k$$

The vehicle load $L_{ik}$ fluctuates continuously as pickups add weight and deliveries remove it, requiring dynamic tracking along the route.

### Dial-a-Ride Problem (DARP)
The human-transport variant of the PDP (paratransit, on-demand ride-sharing). In addition to paired precedence and time window constraints, DARP enforces passenger ride-time limits $L_i$ bounding the maximum in-vehicle duration:

$$t_{i^-} - t_{i^+} \le L_i$$

This severely restricts the feasible solution space by coupling the spatial sequencing with individual passenger service quality, forcing meta-heuristics to balance fleet utilization against inconvenience penalties.

### VRP with Simultaneous Pickups and Deliveries (VRPSPD)
Unlike standard PDP, nodes are not paired. Each customer $i$ simultaneously requires an outbound delivery $d_i$ from the depot and supplies an inbound pickup $p_i$ destined for the depot. The vehicle capacity $Q$ must be respected at all points in the tour; the exact payload on edge $(i,j)$ must satisfy $w_{ij} \le Q$ accounting for the cumulative net load from deliveries and pickups up to that arc.

---

## The Arc Routing Problem (ARP) Family

Problems where operational demand is distributed along network edges rather than at discrete nodes — for example, street sweeping, snow plowing, and utility inspection.

### Chinese Postman Problem (CPP)
Given a graph $G = (V, E)$, find the minimum-cost closed walk traversing every edge at least once. If the graph is Eulerian (all nodes have even degree), the optimal tour cost equals the sum of all edge weights. If not, a minimum-weight perfect matching over odd-degree nodes determines which edges require deadheading (traversal without demand).

### Rural Postman Problem (RPP)
Generalizes the CPP to a required-edge subset $E_R \subset E$. The vehicle must traverse all $e \in E_R$ at least once and may traverse non-required edges to maintain connectivity. Let $x_{ij} \ge 0$ count deadheading traversals; the objective minimizes total traversal cost subject to flow conservation:

$$\min \sum_{(i,j) \in E} c_{ij} x_{ij} + \sum_{(i,j) \in E_R} c_{ij}, \quad \sum_j (x_{ij} - x_{ji}) = 0 \;\forall i$$

### Capacitated Arc Routing Problem (CARP)
Extends the RPP with a homogeneous fleet of capacity $Q$. Each required edge $e \in E_R$ carries a demand $q_e > 0$. The total demand of edges serviced by a single vehicle may not exceed $Q$. Vehicles start and end at a central depot. Binary variables $y_e^k$ activate service of edge $e$ by vehicle $k$; integer variables $x_{ij}^k$ track deadheading flow.

### Periodic CARP (PCARP)
A multi-period extension of the CARP mirroring the PVRP structure. Required arcs carry service frequencies, and the fleet is scheduled across horizon $H$ to satisfy all arc demands while respecting daily vehicle capacities and allowable service patterns.

---

## The General Routing Problem (GRP) Family

A unifying framework that absorbs both node routing (VRP/TSP) and arc routing (ARP/CARP) into a single formulation.

### Standard GRP
Given graph $G = (V, E)$, a required-node subset $V_R \subseteq V$ must be visited and a required-edge subset $E_R \subseteq E$ must be traversed in a single minimum-cost closed walk. The GRP is universal:

- $V_R = V,\; E_R = \emptyset$ $\Rightarrow$ TSP
- $V_R = \emptyset,\; E_R = E$ $\Rightarrow$ CPP
- $V_R = \emptyset,\; E_R \subset E$ $\Rightarrow$ RPP

Subtour elimination constraints must be dynamically formulated to ensure the walk covers all of $V_R \cup E_R$ as a single connected component.

### Capacitated GRP (CGRP)
Injects fleet capacity $Q$ into the GRP. Both required nodes $v \in V_R$ and required edges $e \in E_R$ may carry payload demands. Vehicles must route through the mixed topology without breaching capacity at any point.

---

## The Location Routing Problem (LRP) Family

Couples strategic facility-siting decisions with operational vehicle routing, optimizing both simultaneously rather than sequentially.

### Standard LRP
Given candidate depot locations $D$ and customers $V_c$. Binary variable $z_j \in \{0,1\}$ activates depot $j$ at fixed cost $F_j$. The objective harmonizes capital expenditure with routing cost:

$$\min \sum_{j \in D} F_j z_j + \sum_k \sum_{(i,j)} c_{ij} x_{ij}^k$$

Routes must originate only from open depots: $\sum_i x_{ji}^k \le z_j \;\forall j \in D,\, \forall k$.

### Capacitated LRP (CLRP)
Adds depot throughput limits $W_j$ to the LRP. The total demand assigned to depot $j$ may not exceed its capacity:

$$\sum_{i \in V_c} q_i y_{ij} \le W_j z_j \;\forall j \in D$$

where $y_{ij} \in \{0,1\}$ assigns customer $i$ to depot $j$.

### Multi-Echelon LRP / Two-Echelon VRP (ME-LRP / 2E-VRP)
Models a tiered urban distribution network. A primary fleet transports freight from large depots to intermediate satellite facilities (1st echelon). A secondary fleet of smaller vehicles distributes from satellites to end customers (2nd echelon). The formulation simultaneously optimizes satellite locations, 1st-echelon synchronization schedules, and 2nd-echelon customer routing, subject to flow conservation across both echelons and time-coupling constraints that ensure satellite inventory is replenished before downstream routes depart.

---

## The Inventory Routing Problem (IRP) Family

The IRP transcends pure spatial routing by integrating the tactical tier of Vendor-Managed Inventory (VMI). It optimizes vehicle routes, delivery quantities, and node-level inventory states jointly over a multi-period temporal horizon $\mathcal{T}$.

### Standard IRP
A deterministic multi-period integration. For node $i$ at period $t$: inventory $I_{it}$, unit holding cost $h_i$, delivered quantity $q_{it}$, deterministic consumption $d_{it}$. The objective minimizes routing plus inventory holding costs:

$$\min \sum_{t \in \mathcal{T}} \left( \sum_{(i,j) \in A} c_{ij} x_{ijt} + \sum_{i \in V} h_i I_{it} \right)$$

Governed by inventory flow conservation:

$$I_{it} = I_{i,t-1} + q_{it} - d_{it}, \quad 0 \le I_{it} \le C_i \;\forall t \in \mathcal{T}$$

### Stochastic IRP (SIRP)
Consumption $d_{it}$ is a random variable. To prevent stockouts, SIRP uses robust buffers or chance constraints bounding the stockout probability:

$$\Pr[I_{i,t-1} + q_{it} - d_{it} < 0] \le \alpha$$

This replaces deterministic inventory bounds with statistical confidence intervals derived from the demand distribution variance.

### Multi-Commodity IRP (MCIRP)
Models simultaneous distribution of distinct product families $P$. Delivery variables gain a commodity index $q_{it}^p$, requiring exact tracking of sub-compartments within vehicle capacity and distinct per-commodity depletion rates at each node.

### Robust IRP (RIRP)
A worst-case variant combining robust optimization with inventory routing. Demand $d_{it}$ is drawn from an uncertainty set $\mathcal{U}$, and the solution must be feasible for all realizations. The minimax formulation guarantees zero-stockout service under the worst-case demand trajectory within the budget set, typically yielding a tractable MILP reformulation via the Bertsimas-Sim approach.

### IRP with Lateral Transshipment (IRP-LT)
Extends the standard IRP by permitting direct inventory transfer between customer nodes (lateral transshipment), rather than routing all replenishment exclusively through the depot. This is modelled by introducing arc variables $f_{ijt}$ tracking the quantity shipped from node $i$ to node $j$ at period $t$, subject to balance constraints and vehicle capacity on the transshipment arcs. IRP-LT is particularly effective in settings where customers have complementary demand fluctuations, enabling one node to buffer another without depot intervention.

---
