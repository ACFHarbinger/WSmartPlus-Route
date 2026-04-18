# Neural Combinatorial Optimization for Vehicle Routing

The intersection of machine learning and operations research has produced **Neural Combinatorial Optimization (NCO)** — a family of methods that learn to extract spatial, temporal, and economic features from problem instances to predict near-optimal decisions, bypassing the need for hand-crafted heuristics or computationally expensive exact solvers.

Models in this domain are classified along two axes. By *construction strategy*: **Constructive** models build solutions from scratch (autoregressively or non-autoregressively); **Improvement** models learn to perturb and refine existing feasible solutions; **Hybrid** models parameterize classical meta-heuristics such as ALNS or ACO. By *generality*: **Instance-specific** models are trained and applied to a fixed problem class; **Multi-task** and **generalist** models handle heterogeneous variant families (CVRP, VRPTW, IRP, etc.) from a single set of weights.

---

## 1. Autoregressive Constructive Models

These models build a solution token by token using an encoder-decoder architecture. The encoder embeds the full instance graph once; the decoder attends over it at each step to select the next node.

### Pointer Network (Ptr-Net)
*Vinyals et al., 2015*

The foundational architecture that introduced attention as a trainable "pointer" over variable-length input sequences, resolving the fixed-output-vocabulary limitation of standard seq2seq models.

**Mechanism.** The encoder (LSTM) produces a hidden state $e_j$ for each node $j$. At decoding step $i$, the decoder hidden state $d_i$ is used to compute an unnormalized alignment score over all input positions, from which a categorical distribution over nodes is sampled or greedily decoded.

$$u_j^i = v^\top \tanh(W_1 e_j + W_2 d_i), \qquad p(C_i = j \mid C_{<i}, \mathcal{P}) = \operatorname{softmax}(u^i)_j$$

Ptr-Net was trained with supervised learning on exact solutions; later work replaced this with REINFORCE to remove the dependence on a solver oracle.

### Attention Model (AM)
*Kool et al., 2018*

The canonical Transformer-based routing model, which replaced the RNN encoder/decoder of Ptr-Net with Multi-Head Attention (MHA) and scaled to instances with hundreds of nodes.

**Mechanism.** Node embeddings are computed via a stack of MHA + feedforward layers. The decoder maintains a context vector $h_{(c)}$ — a concatenation of the graph embedding, the last visited node, and the first node (for CVRP, also the remaining capacity) — and produces a pointing distribution over unvisited nodes via a single-head attention with a clipping nonlinearity ($\tanh$ scaled by 10).

**Training.** REINFORCE with a rollout baseline: the baseline $b(s)$ is the cost of the greedy rollout of the best model from the previous epoch, providing a strong, low-variance signal.

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{p_\theta(\pi|s)}\!\left[\bigl(L(\pi) - b(s)\bigr) \nabla_\theta \log p_\theta(\pi \mid s)\right]$$

### POMO (Policy Optimization with Multiple Optima)
*Kwon et al., 2020*

One of the most impactful constructive models, achieving near-LKH3 quality on CVRP-100 at inference time. POMO exploits the fact that every routing problem has $N$ equivalent optimal starting nodes (the $N$-fold symmetry of the tour), using all of them simultaneously during both training and inference.

**Mechanism.** $N$ rollouts are launched in parallel from each of the $N$ starting nodes of an instance. During training, the multi-start reward mean is used as a self-supervised, instance-specific baseline, dramatically reducing gradient variance without requiring a separate baseline network. During inference, the best of the $N$ rollouts is returned.

$$\nabla_\theta \mathcal{J}(\theta) = \frac{1}{N} \sum_{n=1}^N \bigl(L(\pi_n) - \bar{L}\bigr) \nabla_\theta \log p_\theta(\pi_n), \quad \bar{L} = \frac{1}{N}\sum_{n=1}^N L(\pi_n)$$

POMO is often used as the backbone for improvement models (e.g., EAS, SGBS) because its multi-start structure maps naturally onto population-based search.

### SymNCO (Symmetry-Aware NCO)
*Kim et al., 2022*

Generalizes the symmetry exploitation of POMO to arbitrary problem-level invariances (rotation, reflection, node permutation) via a contrastive self-supervised auxiliary loss.

**Mechanism.** For each instance, $M$ semantically equivalent augmented views are generated (e.g., random 8-fold dihedral rotations of the coordinate plane). The policy is required to produce invariant embeddings across views, and the shared mean cost $\bar{L}$ across views serves as the REINFORCE baseline.

$$\nabla_\theta \mathcal{J}(\theta) = \frac{1}{M} \sum_{m=1}^M \bigl(L(\pi_m) - \bar{L}\bigr) \nabla_\theta \log p_\theta(\pi_m \mid s_m)$$

The augmentation-based baseline is strictly tighter than a separate critic network because it conditions on the exact instance topology.

### MatNet
*Kwon et al., 2021*

Designed for asymmetric combinatorial problems — the Asymmetric TSP, heterogeneous fleet VRP, or any setting where the transition cost $c_{ij} \neq c_{ji}$.

**Mechanism.** Rather than encoding nodes in isolation and querying pairwise costs implicitly, MatNet encodes the full cost matrix $C \in \mathbb{R}^{N \times N}$ via a bilinear cross-attention that reads the directed edge cost $c_{ij}$ directly into the attention logit.

$$a_{ij} = \frac{\exp\!\left(q_i^\top k_j + w \cdot c_{ij}\right)}{\sum_l \exp\!\left(q_i^\top k_l + w \cdot c_{il}\right)}$$

This makes MatNet the preferred architecture for Fleet-Mix VRP variants, where different vehicle types create an inherently asymmetric cost structure.

### LEHD (Light Encoder, Heavy Decoder)
*Luo et al., 2023*

Addresses the inference bottleneck of standard constructive models: while AM and POMO use symmetric encoder/decoder capacity, LEHD reconfigures the budget to a shallow encoder and a deep, autoregressive decoder that re-reads instance features at every decoding step.

**Mechanism.** Node embeddings are recomputed at each decoding step conditioned on the current partial solution state (visited set, remaining capacity, elapsed time). This replaces the fixed context vector with a dynamic, step-aware graph representation, substantially improving solution quality on VRPTW and CVRP at longer horizons without increasing parameter count.

### Multi-Decoder Attention Model (MDAM)
*Xin et al., 2021*

Addresses the tendency of greedy decoding to collapse to a single local optimum by learning multiple diverse routing policies simultaneously within one model.

**Mechanism.** $K$ independent decoder heads, each initialized with a different learned embedding bias, are trained jointly with a diversity regularization term that penalizes overlap between the tours they produce. At inference, all $K$ decoders run in parallel and the lowest-cost tour is returned, acting as a parallelized learned beam search without exponential branching.

---

## 2. Non-Autoregressive Constructive Models

Non-autoregressive (NAR) models predict the full solution structure in a single forward pass, treating routing as edge classification or heatmap generation over the instance graph. They are orders of magnitude faster at inference but typically require a post-processing search step to extract a feasible tour.

### NARGNN
*Fu et al., 2021*

Frames TSP/VRP as an edge-existence classification problem over the complete instance graph.

**Mechanism.** A message-passing GNN produces node embeddings $h_i$; these are projected pairwise to produce a probability $P_{ij}$ that edge $(i,j)$ belongs to the optimal tour.

$$P_{ij} = \operatorname{Sigmoid}\!\left(\operatorname{MLP}([h_i \parallel h_j \parallel e_{ij}])\right)$$

Training uses Binary Cross-Entropy against solutions from an exact solver (Concorde, LKH3). The resulting edge heatmap is decoded into a tour via Monte Carlo Tree Search or greedy edge assembly with feasibility repair.

### DPDP (Dynamic Programming with Deep Learning)
*Kool et al., 2022*

Combines the global optimality guarantees of dynamic programming with learned heuristics to prune the DP state space.

**Mechanism.** A GNN scores subsets of states in the DP table, and only the top-$k$ highest-scoring partial solutions are retained at each DP stage (beam-DP). The GNN is trained to assign high scores to states that lie on the path to an optimal solution, effectively learning which partial tours are worth expanding.

This is particularly well-suited to problems with hard temporal constraints (VRPTW, multi-period IRP) where the DP state already encodes time-window feasibility.

### BQ-NCO (Bi-directional Q-learning NCO)
*Drakulic et al., 2023*

Recasts solution construction as value-function estimation rather than policy learning, enabling principled exploration of the solution space without requiring rollouts.

**Mechanism.** A transformer encodes both the "left" partial tour (already assigned nodes) and the "right" partial tour (end-anchor constraints) simultaneously. A Q-value head scores each candidate next node based on the expected solution quality of completing the tour from that intermediate state. Bi-directionality anchors the search from both the start and end of the sequence, significantly reducing the compounding error of purely left-to-right autoregressive construction.

### PolyNet
A constructive model trained to learn multiple structurally distinct policies simultaneously, deliberately covering disjoint regions of the solution space. Rather than diversity-by-randomness, PolyNet conditions each decoder on a learned "policy token" that specifies which exploration mode to use — functioning as a neural ensemble that spans the solution landscape in a structured, learnable way.

---

## 3. Improvement and Local Search Models

Instead of building a solution from scratch, improvement models learn to iteratively perturb a feasible solution toward lower cost. They are often competitive with or superior to constructive models at large instance scales because they can exploit the structure of a good initial solution.

### L2I (Learning to Improve)
*Chen & Tian, 2019*

One of the first end-to-end neural improvement models for VRP. L2I frames the search as a Markov Decision Process where the state is the current routing solution and the action is a local operator (2-opt, Or-opt, node relocation).

**Mechanism.** A two-phase encoder computes both node-level embeddings (spatial features) and route-level embeddings (sequence features). A pointer-based selector chooses which operator to apply and to which portion of the solution, trained via REINFORCE. Crucially, L2I maintains a feasibility mask that prevents the agent from selecting infeasible exchanges, making it directly applicable to CVRP and VRPTW without post-hoc repair.

### N2S (Neural Neighborhood Search)
*Chen et al., 2022*

A graph-encoder-based improvement model that generalizes local search by learning a continuous scoring function over all candidate node-swap actions.

**Mechanism.** The current routing state is encoded as a dynamic graph. N2S outputs a score matrix over all $(i, j)$ node-pair exchanges, from which it selects the highest-scoring feasible swap. Unlike taboo search with a fixed neighborhood size, the learned scorer implicitly adapts the effective neighborhood radius to the local solution geometry.

### DACT (Dual-Aspect Collaborative Transformer)
*Ma et al., 2021*

Improvement model that explicitly decouples the two information streams that matter for routing refinement: what the nodes *are* (coordinates, demands) and where they *currently sit* in the solution sequence.

**Mechanism.** Two parallel transformer stacks — a Node-Aspect Transformer and a Positional-Aspect Transformer — process spatial and sequential features respectively. A cross-attention collaboration module fuses both streams before the action head selects a node-pair relocation. This prevents the common failure mode where improvement models forget the current tour structure when attending to spatial features.

### NeuOpt (Neural Optimizer for $k$-Opt)
*Luo et al., 2023*

A neural parameterization of $k$-opt that avoids the $\mathcal{O}(N^k)$ exhaustive search by directly predicting which $k$ edges to exchange.

**Mechanism.** A self-attention network autoregressively selects a sequence of edges to break and reconnect, conditioned on all previously selected edges. The joint probability factorizes over the exchange tuple:

$$p(E_\text{swap} \mid G, S_t) = \prod_{i=1}^k p(e_i \mid G, S_t, e_1, \dots, e_{i-1})$$

NeuOpt effectively acts as a learned Lin-Kernighan heuristic, capable of discovering long-range improving chains that standard 2-opt or 3-opt miss entirely.

### SGBS (Simulation-Guided Beam Search)
*Choo et al., 2022*

A search-time enhancement for any trained constructive policy (POMO, AM) that uses Monte Carlo simulation to evaluate beam candidates rather than relying on greedy completion.

**Mechanism.** At each beam expansion step, candidate partial solutions are evaluated not by their immediate log-probability but by the expected total cost under $T$ stochastic rollouts from that partial state. This is lookahead-in-beam-search, and it converts a standard constructive model into a vastly more powerful inference-time planner without any retraining.

---

## 4. Meta-Heuristic Augmented Models

These architectures embed neural networks inside classical operations research algorithms, replacing hand-crafted heuristic components with learned counterparts while preserving the algorithmic skeleton that provides feasibility guarantees.

### DeepACO (Deep Ant Colony Optimization)
*Liu et al., 2023*

Replaces the static heuristic desirability measure $\eta_{ij} = 1/d_{ij}$ of classical ACO with a learned graph neural network, enabling the ants to follow instance-adaptive, constraint-aware guidance.

**Mechanism.** A GNN processes the instance graph and outputs a dense heuristic matrix. The transition probability for ant $k$ at node $i$ selecting node $j$ becomes:

$$p_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\operatorname{MLP}_\theta(h_i, h_j, e_{ij})]^\beta}{\sum_{l \in \mathcal{N}_i^k} [\tau_{il}]^\alpha \cdot [\operatorname{MLP}_\theta(h_i, h_l, e_{il})]^\beta}$$

where $\tau_{ij}$ is the pheromone trail (updated classically) and $\theta$ is trained via REINFORCE on the ACO colony's collective reward. DeepACO significantly outperforms classical ACO on CVRP, OP, and PDP benchmarks.

### GFlowNet Ant Colony System
Combines the emerging paradigm of Generative Flow Networks with ACO to overcome the mode-collapse failure of standard RL-trained routing policies.

**Mechanism.** Classical policy gradient methods collapse onto the single best tour they find during training. GFlowNets instead learn to sample solutions *proportionally to their reward* — a much more diverse distribution. The pheromone matrix is interpreted as a learned flow distribution, and training minimizes the Trajectory Balance loss:

$$\mathcal{L}(\theta) = \left(\log \frac{Z_\theta \cdot P_F(\tau)}{R(x) \cdot P_B(\tau)}\right)^2$$

where $P_F$ is the forward construction policy, $P_B$ is a fixed backward policy, $R(x)$ is solution quality, and $Z_\theta$ is a learned partition function. The result is a colony that reliably generates a diverse portfolio of high-quality solutions rather than concentrating on a single local optimum.

### DR-ALNS (Deep RL-based ALNS)
Classical ALNS selects destroy/repair operator pairs via a roulette-wheel mechanism weighted by recent performance. DR-ALNS replaces this with a DQN or PPO agent that learns the complex, instance-dependent dynamics of which operator pairs are most effective at each stage of the search.

**Mechanism.** The state $s_t$ encodes the current optimality gap, search iteration count, recent operator performance history, and fleet utilization metrics. The action $a_t \in \Omega_\text{destroy} \times \Omega_\text{repair}$ selects a pair from the operator catalog. The neural network learns that, for example, *relatedness removal + greedy repair* works well early in search, while *worst removal + SISR repair* is more effective near convergence — dynamics that static roulette wheels cannot capture.

### GLOP (Global-Local Optimization Policy)
*Ye et al., 2024*

A hierarchical hybrid that orchestrates the handoff between a neural constructor and a classical local search solver, addressing the well-known weakness of neural models at very large scales ($N > 500$).

**Mechanism.** A high-level RL policy partitions the instance into subproblems (spatial clustering), assigns each subproblem to a neural constructive model for an initial solution, then hands the partial state to a fast classical solver (LKH3) for local refinement. The partition policy is trained end-to-end on the combined neural + classical objective, learning to create subproblems that are optimally sized for the downstream solver.

---

## 5. Predict-Once Solvers

Predict-once solvers decouple the neural model from the search entirely. The DNN runs a single forward pass over the instance and outputs supplementary information — edge probability heatmaps, candidate adjacency lists, cost adjustments, or branching priors — that is handed to an independent OR solver or search algorithm. The solver then operates in its own loop without any further neural inference. This clean separation means the OR solver retains its feasibility guarantees while benefiting from learned, instance-aware guidance.

The key distinction from non-autoregressive constructive models (Section 2) is that NAR models output a representation that *is* the solution (decoded directly into a tour), whereas predict-once models output guidance that an entirely separate solver consumes as an input signal.

### GCN Edge Heatmap + Beam Search
*Joshi et al., 2019*

The canonical predict-once architecture. A Graph Convolutional Network is trained (via supervised learning against exact solutions) to output, for every edge $(i,j)$ in the instance graph, the probability that the edge belongs to the optimal tour. This heatmap is computed once, then a classical beam search assembles a feasible tour by following the highest-probability edges subject to routing constraints.

**Mechanism.** The GCN produces node embeddings $h_i$ via $L$ message-passing layers. Edge probabilities are computed from pairwise projections:

$$P_{ij} = \operatorname{Sigmoid}(\operatorname{MLP}([h_i \parallel h_j \parallel e_{ij}]))$$

Beam search then decodes the heatmap by expanding partial tours in order of cumulative edge probability, pruning branches that violate capacity or time-window constraints. Because the GCN runs only once, inference is fast and scales to instances far beyond what autoregressive models can decode greedily.

This architecture and its variants serve as the backbone of most subsequent predict-once work; NARGNN (Section 2) is a direct descendant.

### DIFUSCO (Diffusion-based Combinatorial Optimization)
*Sun & Yang, 2023*

Applies denoising diffusion probabilistic models (DDPMs) — the same family behind image generation — to combinatorial optimization by framing the solution heatmap as a structured image to be denoised.

**Mechanism.** The training process diffuses a binary edge-solution matrix $x_0 \in \{0,1\}^{N \times N}$ (the ground-truth tour) forward with Gaussian noise over $T$ timesteps to produce $x_T \approx \mathcal{N}(0, I)$. A graph transformer learns the reverse process, predicting $x_0$ from $x_t$ conditioned on the instance graph $G$:

$$p_\theta(x_{t-1} \mid x_t, G) = \mathcal{N}(x_{t-1};\, \mu_\theta(x_t, t, G),\, \Sigma_\theta(x_t, t, G))$$

At inference, the model runs the full reverse diffusion chain (predict-once across all $T$ denoising steps), producing an edge probability heatmap that is decoded via greedy tour assembly or MCTS. DIFUSCO outperforms GCN-based heatmap models on TSP-500 and TSP-1000 without any fine-tuning, because the iterative denoising naturally integrates global tour consistency across the generation process.

The full reverse-diffusion chain is still a single "prediction episode" in the predict-once sense — the OR solver (MCTS, greedy) receives the completed heatmap and runs independently.

### Neural Candidate List Generation for LKH
Multiple works (*Hottung et al.; Fu et al.*) train a GNN to produce a sparse candidate adjacency list — a small set of promising successor edges for each node — that replaces LKH3's internal greedy-distance heuristic for candidate generation.

**Mechanism.** LKH3 normally considers the $k$-nearest neighbors (by Euclidean distance) as candidate next moves. A GNN trained on historical optimal tours replaces this with a *learned* $k$-candidate list per node, where candidates are ranked by predicted edge inclusion probability. LKH3 then runs its full local search unchanged, but now explores a richer, constraint-aware neighborhood rather than a purely geometric one.

The quality of LKH3's solution on CVRP and VRPTW improves by 1–3% at equal iteration budgets, because the learned candidates steer the $k$-opt chains away from clearly suboptimal edges early in the search.

### ML-guided Branch and Bound
*Khalil et al., 2016; Gasse et al., 2019*

A family of methods that learn to predict branching variable scores or node selection priorities from historical B&B traces, then fix those predictions at the start of solving so the B&B tree runs without any further neural inference.

**Mechanism.** The predictor (a GNN operating on the bipartite constraint–variable graph of the LP relaxation) is evaluated once at the root node or at fixed tree depths, outputting a ranking over candidate branching variables. The B&B solver uses this ranking as its variable-selection policy for the remainder of the solve, replacing the default strong-branching or pseudo-cost rules.

For routing MIPs (column generation, set-partitioning formulations of VRP), this approach reduces the number of B&B nodes explored by up to an order of magnitude on instances drawn from the same distribution as the training set.

---

## 6. Predict-Many Solvers

Predict-many solvers follow the same predict-then-search philosophy as predict-once, but the neural model is called *repeatedly* during the search — each call consuming the current search state and outputting updated guidance that restricts or redirects the solver's next steps. This tighter coupling between the neural predictor and the search loop enables progressive narrowing of the solution space at the cost of additional inference overhead.

The trade-off is explicit: predict-once solvers amortize neural cost to a single call (fast, less adaptive); predict-many solvers pay per search step (slower, but the guidance stays synchronized with where the search currently is).

### DPDP (Dynamic Programming with Deep Learning)
*Kool et al., 2022*

Already introduced in Section 2 as a non-autoregressive model, DPDP is more precisely a predict-many solver: the GNN is queried at every stage of the beam-DP expansion to score candidate partial solutions, and only the top-$k$ are retained. Each DP stage triggers a fresh GNN forward pass conditioned on the current set of surviving partial solutions.

**Mechanism.** At DP stage $t$, the surviving beam $\mathcal{B}_t$ (the top-$k$ partial tours from the previous stage) is passed to the GNN, which scores every possible extension $(s, j)$ — partial tour $s$ extended by node $j$. Only the top-$k$ extensions across all $(s, j)$ pairs survive to form $\mathcal{B}_{t+1}$. This is repeated until all nodes are assigned.

The GNN's predict-many loop means it sees progressively more complete solutions at each stage, allowing later calls to exploit information about which routes were pruned — information that a predict-once heatmap computed at the empty-tour stage cannot have.

### Neural MCTS (Monte Carlo Tree Search with Neural Guidance)
Analogous to AlphaGo's combination of a policy network and value network with MCTS, applied to routing. The neural network is not invoked once before search; it is queried at every MCTS node expansion and rollout.

**Mechanism.** Two networks — a *policy network* $p_\theta(a \mid s)$ that scores candidate next nodes to expand, and a *value network* $v_\phi(s)$ that estimates the expected tour cost from partial solution state $s$ — guide the Upper Confidence Bound (UCT) criterion:

$$\operatorname{UCT}(s, a) = Q(s,a) + c \cdot p_\theta(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

where $Q(s,a)$ is the empirical mean tour cost from simulations through action $a$, and $N(s)$, $N(s,a)$ are visit counts. Every tree node expansion calls $p_\theta$; every leaf rollout calls $v_\phi$. The repeated neural calls allow the search to concentrate on the most promising subtrees far more precisely than random rollouts, at the cost of one neural forward pass per MCTS iteration.

Neural MCTS achieves near-optimal results on TSP-100 when given sufficient iteration budgets, and scales naturally to VRP by enforcing capacity feasibility as a hard constraint on legal actions at each state.

### Iterative Graph Sparsification
A class of methods that progressively prune the instance graph — removing edges the model deems unpromising — over multiple rounds of prediction, feeding each pruned graph back into the neural model for the next round.

**Mechanism.** Starting from the complete graph $G^{(0)}$, the GNN predicts edge inclusion probabilities $P^{(0)}_{ij}$. Edges below a threshold $\epsilon$ are pruned to produce $G^{(1)}$. The GNN is re-run on $G^{(1)}$ — now a much sparser graph with a qualitatively different topology — to produce $P^{(1)}_{ij}$, and another round of pruning follows. After $R$ rounds, a classical solver (LKH, beam search) runs on $G^{(R)}$, whose edge count may be $\mathcal{O}(N)$ rather than $\mathcal{O}(N^2)$.

The key insight is that re-running the GNN on the pruned graph is not redundant: the changed neighborhood structure alters message-passing aggregations, so the model effectively sees a different problem geometry at each round and can correct early over-pruning errors. This approach scales to TSP-10000 and VRP-5000, well beyond the range of standard constructive models.

### Confidence-based Candidate Elimination
*Related to iterative narrowing and curriculum-guided search*

A predict-many strategy specifically for heatmap-decoded models: the neural model is re-queried on progressively smaller candidate sets, eliminating edges whose predicted inclusion probability has not exceeded a confidence threshold across multiple independent rollouts.

**Mechanism.** $M$ stochastic forward passes produce $M$ independent heatmaps. Edges where fewer than $\tau \cdot M$ passes assign probability above $\epsilon$ are eliminated from the candidate graph. The model is then re-queried on the reduced graph — now with a lower-entropy input distribution — to produce sharper edge scores for the remaining candidates. Iteration continues until the candidate graph is sparse enough for exact or near-exact decoding.

This approach is particularly effective for CVRP variants with hard constraints (time windows, precedence) where early false-positive edges in the heatmap corrupt feasibility; the multi-pass filtering catches these before the decoder commits to them.

---

## 7. Active Search and Fine-Tuning Methods

These methods improve solution quality for a *specific test instance* by updating model parameters at inference time — bridging the gap between amortized heuristics and instance-level exact solvers.

### EAS (Efficient Active Search)
*Hottung et al., 2022*

Active Search (Bello et al., 2016) fine-tunes the full model on each test instance via policy gradients, but this is prohibitively slow. EAS achieves comparable quality at a fraction of the cost by only fine-tuning a small set of instance-specific embedding parameters (an "adapter") while keeping the backbone frozen.

**Mechanism.** A lightweight adapter layer is injected into the pretrained AM or POMO model. At test time, only the adapter weights are optimized via REINFORCE on the specific instance. Because the adapter has far fewer parameters than the full model, gradient steps are cheap and convergence is fast — typically 20–50× faster than full Active Search.

### COMPASS (Combination of Parallel Search Strategies)
*Luo et al., 2023*

A search-time strategy that runs multiple constructive policies in parallel — each initialized from a different symmetry augmentation and/or temperature — and applies local improvement operators asynchronously to the best solutions across the population. COMPASS is model-agnostic and can wrap any pretrained NCO backbone.

---

## 8. Multi-Task and Generalist Models

A significant research direction since 2023 is training a *single* model that generalizes across VRP variant families without instance-type specification at inference time.

### MVMoE (Multi-Task VRP with Mixture of Experts)
*Jiao et al., 2024*

Handles 16+ VRP variants (CVRP, VRPTW, OVRP, VRPB, mixed combinations) from a single set of weights using a sparse Mixture-of-Experts routing mechanism.

**Mechanism.** The encoder backbone is shared. A gating network routes each input token to one of $E$ expert feedforward modules, where each expert specializes in a different constraint topology (time windows, open routes, backhauls, etc.). The gating is learned end-to-end, and at inference, variant identity is not provided explicitly — the model infers it from the instance features. MVMoE achieves competitive performance with single-task models across all 16 variants while using roughly the same parameter budget as a single-task AM.

### UniFIED / Omni-VRP Style Architectures

A class of models that encode VRP constraints as natural language or structured token sequences appended to the instance graph, enabling zero-shot generalization to novel constraint combinations not seen during training.

**Mechanism.** The instance (node coordinates + demands) is encoded spatially; constraint specifications (e.g., "time windows active, fleet heterogeneous, open routes") are encoded as additional tokens and cross-attended with the spatial embedding. This effectively makes the model a *conditional* routing policy, conditioned on the constraint vocabulary, rather than a fixed-variant policy.

---

## 9. Inventory Routing and Multi-Period Specific Models

The IRP adds a temporal dimension — decisions made today affect feasibility and cost tomorrow — that pure routing architectures do not natively handle.

### L2D (Learning to Dispatch)
A reinforcement learning model for IRP that decouples the two interdependent decision spaces: *when and how much to deliver* (dispatch policy) vs. *in what order to visit customers* (routing policy).

**Mechanism.** The RL agent observes current inventory levels, demand forecasts, and vehicle positions, and outputs a dispatch decision (which customers to serve and what quantity to deliver this period). The geometric routing sub-problem is solved independently by a fast classical TSP heuristic or NCO model. Decoupling avoids the curse of dimensionality that arises from jointly optimizing route sequences and inventory levels over $T$ periods.

### BGN (Bipartite Graph Network)
Designed for the IRP and Location-Routing Problem, where the fundamental structure is a bipartite interaction between supply nodes (depots, warehouses) and demand nodes (customers).

**Mechanism.** Separate embedding layers encode supply-side and demand-side features. Cross-attention layers then compute a soft assignment matrix between depots and customers, which is used to initialize the routing policy. This explicit supply-demand separation avoids the representational bias of homogeneous GNNs, which treat all nodes identically.

### Hierarchical Temporal Attention Networks

A family of architectures for multi-period routing that uses a two-level attention hierarchy: a *period-level* transformer attends over the $T$-period planning horizon to decide visit schedules, while a *within-period* transformer handles the geometric routing for a single day given the schedule.

**Mechanism.** Period-level attention captures inter-day inventory dynamics (the replenishment wave propagating forward through time). Within-period attention captures spatial routing efficiency. The two levels are trained jointly with a shared reward signal — total horizon cost — so the schedule level learns to create daily workloads that the routing level can execute efficiently.

This architecture is the natural neural counterpart to the Progressive Hedging / Benders decomposition approaches in the classical SIRP literature, and it is the most directly applicable NCO family to multi-period VRP.

---
