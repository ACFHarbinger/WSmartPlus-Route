use crate::params::Params;

#[derive(Clone, Debug)]
pub struct CostEval {
    pub distance: f64,
    pub capacity_excess: f64,
    pub penalized_cost: f64, // Used for internal optimization (fitness)
    pub profit: f64,         // True VRPP Profit
    pub is_feasible: bool,
}

#[derive(Clone)]
pub struct Individual {
    pub giant_tour: Vec<usize>,
    // Linked List representation for O(1) LS moves
    // successors[i] = next node after i
    // predecessors[i] = prev node before i
    // 0 is the depot.
    pub successors: Vec<usize>,
    pub predecessors: Vec<usize>,
    pub chrom_r: Vec<Vec<usize>>, // Routes in vector form (for Crossover/Split compatibility)
    pub eval: CostEval,
    pub biased_fitness: f64,
    pub diversity_contribution: f64,
}

impl Individual {
    pub fn new(giant_tour: Vec<usize>) -> Self {
        Self {
            giant_tour,
            successors: Vec::new(),
            predecessors: Vec::new(),
            chrom_r: Vec::new(),
            eval: CostEval {
                distance: 0.0,
                capacity_excess: 0.0,
                penalized_cost: 0.0,
                profit: 0.0,
                is_feasible: false,
            },
            biased_fitness: 0.0,
            diversity_contribution: 0.0,
        }
    }

    pub fn evaluate(&mut self, params: &Params) {
        self.eval.distance = 0.0;
        self.eval.capacity_excess = 0.0;
        let mut revenue = 0.0;

        // Reset linked lists if not sized correct (e.g. fresh individual)
        // or just traverse existing?
        // Actually, evaluate is usually called AFTER chrom_r is set by Split.
        // So we should build links FROM chrom_r here, or assume they are built?
        // In C++, split fills chromR, then evaluateCompleteCost builds links.
        // Let's match C++: build links from chrom_r.

        if self.successors.len() != params.n_clients + 1 {
            self.successors = vec![0; params.n_clients + 1];
            self.predecessors = vec![0; params.n_clients + 1];
        }

        // Initialize depot links: 0 -> 0 (empty) is valid?
        // Actually C++ maintains depot per vehicle. Rust simplified version might just have one depot node 0?
        // If we have multi-routes, they all start/end at 0.
        // But 0 can't point to multiple next nodes in a simple array.
        // C++ uses separate Depots for each vehicle! `depots` vector.
        // In Rust, we might need a more complex structure if we want O(1) for multiple routes.
        // OR we just use chrom_r for validation and cost, and only use links inside LS logic?
        // The requirement is "Refactor LocalSearch (Index-based Linked List)".
        // Meaning LS converts to links, optimizes, convert back.
        // OR Individual keeps links persistent?
        // C++ Individual has `successors` vector of size NB_CLIENTS+1.
        // Wait, how does it handle multiple routes connected to depot?
        // params.nbClients+1 size.
        // Ah, C++ code: `predecessors[chromR[r][0]] = 0;` ... `successors[chromR[r][last]] = 0;`
        // So all routes start from 0 and end at 0.
        // But `successors[0]` can only hold ONE value.
        // So you can't traverse ALL routes from 0 using just `successors` array.
        // You iterate `chromR` (routes) to find heads.
        // Links are valid for clients 1..N.
        // 0 -> client A? No, `predecessors[A] = 0` means A is start of route.
        // `successors[B] = 0` means B is end of route.
        // You cannot go 0 -> A using this array.
        // This is "Path representation" for clients mostly.

        for route in &self.chrom_r {
            if route.is_empty() {
                continue;
            }

            let mut r_dist = params.dist_matrix[0][route[0]];
            let mut r_load = params.demands[route[0]];
            revenue += r_load * params.r_coeff;

            self.predecessors[route[0]] = 0; // Start of route

            for k in 0..route.len() - 1 {
                let u = route[k];
                let v = route[k + 1];
                r_dist += params.dist_matrix[u][v];
                r_load += params.demands[v];
                revenue += params.demands[v] * params.r_coeff;

                self.successors[u] = v;
                self.predecessors[v] = u;
            }
            let last = route[route.len() - 1];
            r_dist += params.dist_matrix[last][0];
            self.successors[last] = 0; // End of route

            self.eval.distance += r_dist;
            if r_load > params.vehicle_capacity {
                self.eval.capacity_excess += r_load - params.vehicle_capacity;
            }
        }
        self.eval.is_feasible = self.eval.capacity_excess < 1e-3;

        let transport_cost = self.eval.distance * params.c_coeff;
        self.eval.profit = revenue - transport_cost;
        // penalized_cost is calculated externally or we add method with penalty arg?
        // C++ calculate penalized cost inside evaluateCompleteCost using stored params penalties.
        // We will update this in `compute_penalized_cost`.
    }

    pub fn compute_penalized_cost(
        &mut self,
        penalty_capacity: f64,
        c_coeff: f64,
        _r_coeff: f64,
        revenue: f64,
    ) {
        let transport_cost = self.eval.distance * c_coeff;
        // Profit is Revenue - Cost
        // But "penalized cost" (fitness) is Cost - Revenue + Penalties (Minimization context)
        // Since we optimize Profit, we want Maximize Profit - Penalties.
        // Or Minimize Cost - Revenue + Penalties.
        // Let's stick to C++ Minimize Cost logic for internal fitness?
        // Rust HGS original was Maximize Profit.
        // Plan says "maintaining the VRPP objective".
        // So Fitness should be Penalized Profit = Profit - Penalties.
        // But `penalized_cost` implies cost (minimization).
        // Let's use `penalized_cost` as "Negative Penalized Profit" for minimization sort compatibility?
        // Or just cost + penalties - revenue? Yes.
        self.eval.penalized_cost =
            transport_cost - revenue + (penalty_capacity * self.eval.capacity_excess);
    }
}

impl PartialEq for Individual {
    fn eq(&self, other: &Self) -> bool {
        (self.eval.penalized_cost - other.eval.penalized_cost).abs() < 1e-6
    }
}
impl Eq for Individual {}
