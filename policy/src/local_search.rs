/*!
 * Local search operators for route improvement.
 *
 * Implements neighborhood operators: Relocate, Swap, 2-Opt, SWAP*.
 * Uses linked lists for O(1) move evaluation.
 */

use crate::circle_sector::CircleSector;
use crate::individual::Individual;
use crate::params::Params;
use rand::rng;
use rand::seq::SliceRandom;

/**
 * Local search engine.
 *
 * # Operators
 *
 * - **Relocate**: Move 1-2 nodes between routes
 * - **Swap**: Exchange 1-2 nodes between routes
 * - **2-Opt**: Eliminate edge crossings within a route
 * - **2-Opt\***: Inter-route edge exchange
 * - **SWAP\***: Geometric sector-based node exchanges
 *
 * # Granular Search
 *
 * Only explores moves involving nodes in each other's `correlated_vertices`
 * (top-k nearest neighbors), reducing complexity from O(N²) to O(N×k).
 */
pub struct LocalSearch {
    penalty_capacity: f64,
}

impl LocalSearch {
    /** Creates a new local search instance. */
    pub fn new() -> Self {
        Self {
            penalty_capacity: 1.0,
        }
    }

    /**
     * Runs local search until no improving move is found.
     *
     * # Algorithm
     *
     * 1. Initialize route IDs and node order
     * 2. While improved and loop_cnt < max_loops:
     *    - Try relocate operators
     *    - Try swap operators
     *    - Try 2-Opt intra-route
     *    - Try 2-Opt* inter-route
     *    - Try SWAP* (if enabled)
     * 3. Rebuild `chrom_r` from linked lists
     * 4. Re-evaluate solution
     *
     * # Complexity
     *
     * O(M × N × k × L) where:
     * - M = number of operators
     * - N = number of customers
     * - k = granular neighborhood size
     * - L = average number of loops until convergence
     */
    pub fn run(&mut self, indiv: &mut Individual, params: &Params, penalty_capacity: f64) {
        self.penalty_capacity = penalty_capacity;

        let mut improved = true;
        let max_loops = 50;
        let mut loop_cnt = 0;

        // Build route_ids map for O(1) checks
        let mut route_ids = self.get_route_ids(indiv, params);

        let mut order_nodes: Vec<usize> = (1..=params.n_clients).collect();
        let mut rng = rng();
        order_nodes.shuffle(&mut rng);

        while improved && loop_cnt < max_loops {
            improved = false;
            loop_cnt += 1;

            if params.ap.time_limit > 0.0
                && params.start_time.elapsed().as_secs_f64() > params.ap.time_limit
            {
                break;
            }

            // Re-shuffle for variety?
            // order_nodes.shuffle(&mut rng);

            // 1. Relocate
            if self.relocate(indiv, params, &order_nodes, &mut route_ids) {
                improved = true;
                continue;
            }

            if self.relocate_two(indiv, params, &order_nodes, &mut route_ids) {
                improved = true;
                continue;
            }

            if self.relocate_two_reversed(indiv, params, &order_nodes, &mut route_ids) {
                improved = true;
                continue;
            }

            // 2. Swap
            if self.swap(indiv, params, &order_nodes, &mut route_ids) {
                improved = true;
                continue;
            }

            if self.swap_two_one(indiv, params, &order_nodes, &mut route_ids) {
                improved = true;
                continue;
            }

            if self.swap_two_two(indiv, params, &order_nodes, &mut route_ids) {
                improved = true;
                continue;
            }

            // 3. 2-Opt Intra
            if self.two_opt_intra(indiv, params, &order_nodes, &mut route_ids) {
                improved = true;
                continue;
            }

            // 4. 2-Opt* (Inter)
            if self.two_opt_star(indiv, params, &order_nodes, &mut route_ids) {
                improved = true;
                continue;
            }

            // 5. Swap*
            if params.ap.use_swap_star {
                if self.swap_star(indiv, params, &mut route_ids) {
                    improved = true;
                    continue;
                }
            }
        }

        // Sync chrom_r
        self.update_chrom_r(indiv, params);

        // Re-evaluate to get exact costs
        indiv.evaluate(params);
    }

    fn update_chrom_r(&self, indiv: &mut Individual, params: &Params) {
        indiv.chrom_r.clear();
        let n = params.n_clients;
        let mut visited = vec![false; n + 1];

        // Find route starts.
        for i in 1..=n {
            if !visited[i] && indiv.predecessors[i] == 0 {
                let mut route = Vec::new();
                let mut curr = i;
                let mut loop_det = 0;
                while curr != 0 {
                    if visited[curr] {
                        break;
                    } // Safe guard
                    visited[curr] = true;
                    route.push(curr);
                    curr = indiv.successors[curr];

                    loop_det += 1;
                    if loop_det > n + 1 {
                        break;
                    }
                }
                indiv.chrom_r.push(route);
            }
        }
    }

    fn get_route_ids(&self, indiv: &Individual, params: &Params) -> Vec<usize> {
        let mut ids = vec![0; params.n_clients + 1];
        let mut rid = 1;

        let n = params.n_clients;
        let mut visited = vec![false; n + 1];

        for i in 1..=n {
            if !visited[i] && indiv.predecessors[i] == 0 {
                let mut curr = i;
                while curr != 0 {
                    visited[curr] = true;
                    ids[curr] = rid;
                    curr = indiv.successors[curr];
                }
                rid += 1;
            }
        }
        ids
    }

    // Helper to incrementally update route ID map after a move
    // Actually, it's easier to just re-scan the changed routes?
    // Or just re-scan all since it's O(N).
    fn update_route_ids(
        &self,
        indiv: &Individual,
        params: &Params,
        route_ids: &mut Vec<usize>,
        _changed_nodes: &[usize],
    ) {
        // Simple strategy: re-trace from the route heads of the changed nodes.
        // Find head of route for a node
        // But if we merged routes, two heads became one.
        // If we split, one head became two.
        // Re-calculating all is safest for now (O(N)).
        *route_ids = self.get_route_ids(indiv, params);
    }

    fn calculate_route_load(&self, indiv: &Individual, params: &Params, node: usize) -> f64 {
        let mut curr = node;
        while indiv.predecessors[curr] != 0 {
            curr = indiv.predecessors[curr];
        }

        let mut load = 0.0;
        while curr != 0 {
            load += params.demands[curr];
            curr = indiv.successors[curr];
        }
        load
    }

    fn get_penalty(&self, load: f64, capacity: f64) -> f64 {
        if load > capacity {
            (load - capacity) * self.penalty_capacity
        } else {
            0.0
        }
    }

    // --- RELOCATE ---
    fn relocate(
        &self,
        indiv: &mut Individual,
        params: &Params,
        order_nodes: &[usize],
        route_ids: &mut Vec<usize>,
    ) -> bool {
        for &u in order_nodes {
            let rid_u = route_ids[u];
            for &v in &params.correlated_vertices[u] {
                let rid_v = route_ids[v];

                // Move U after V?
                if rid_u == rid_v {
                    // Intra-route
                    let u_prev = indiv.predecessors[u];
                    if v == u_prev {
                        continue;
                    } // Already after V
                    let u_next = indiv.successors[u];
                    if v == u_next {
                        continue;
                    } // V is U's successor, can't insert U after V (would be same spot)

                    if u == v {
                        continue;
                    }

                    let v_next = indiv.successors[v];
                    let cost_rem = params.dist_matrix[u_prev][u]
                        + params.dist_matrix[u][u_next]
                        + params.dist_matrix[v][v_next];
                    let cost_add = params.dist_matrix[u_prev][u_next]
                        + params.dist_matrix[v][u]
                        + params.dist_matrix[u][v_next];

                    let delta = (cost_add - cost_rem) * params.c_coeff;
                    if delta < -1e-5 {
                        self.apply_relocate(indiv, u, v, u_prev, u_next, v_next);
                        // route_ids unchanged
                        return true;
                    }
                } else {
                    // Inter-route
                    let dem_u = params.demands[u];
                    let load_u = self.calculate_route_load(indiv, params, u);
                    let load_v = self.calculate_route_load(indiv, params, v);

                    let u_prev = indiv.predecessors[u];
                    let u_next = indiv.successors[u];
                    let v_next = indiv.successors[v];

                    let cost_rem = params.dist_matrix[u_prev][u]
                        + params.dist_matrix[u][u_next]
                        + params.dist_matrix[v][v_next];
                    let cost_add = params.dist_matrix[u_prev][u_next]
                        + params.dist_matrix[v][u]
                        + params.dist_matrix[u][v_next];

                    let delta_dist = (cost_add - cost_rem) * params.c_coeff;

                    let pen_u_old = self.get_penalty(load_u, params.vehicle_capacity);
                    let pen_v_old = self.get_penalty(load_v, params.vehicle_capacity);
                    let pen_u_new = self.get_penalty(load_u - dem_u, params.vehicle_capacity);
                    let pen_v_new = self.get_penalty(load_v + dem_u, params.vehicle_capacity);

                    let delta = delta_dist + (pen_u_new + pen_v_new - pen_u_old - pen_v_old);

                    if delta < -1e-5 {
                        self.apply_relocate(indiv, u, v, u_prev, u_next, v_next);
                        self.update_route_ids(indiv, params, route_ids, &[]);
                        return true;
                    }
                }
            }
        }
        false
    }

    fn apply_relocate(
        &self,
        indiv: &mut Individual,
        u: usize,
        v: usize,
        u_prev: usize,
        u_next: usize,
        v_next: usize,
    ) {
        // Remove U
        indiv.successors[u_prev] = u_next;
        if u_next != 0 {
            indiv.predecessors[u_next] = u_prev;
        }

        // Insert U after V
        indiv.successors[v] = u;
        indiv.predecessors[u] = v;
        indiv.successors[u] = v_next;
        if v_next != 0 {
            indiv.predecessors[v_next] = u;
        }
    }

    fn relocate_two(
        &self,
        indiv: &mut Individual,
        params: &Params,
        order_nodes: &[usize],
        route_ids: &mut Vec<usize>,
    ) -> bool {
        for &u in order_nodes {
            let x = indiv.successors[u];
            if x == 0 {
                continue;
            }

            let rid_u = route_ids[u];
            let rid_x = route_ids[x];
            // Ensure structural integrity: Node X (successor of U) must be in the same route as U
            // Ensure structural integrity: Node X (successor of U) must be in the same route as U
            assert_eq!(
                rid_u, rid_x,
                "Structural Error: U ({}) and Successor X ({}) must be in same route ({} vs {})",
                u, x, rid_u, rid_x
            );

            for &v in &params.correlated_vertices[u] {
                let rid_v = route_ids[v];

                // Intra-route checks
                if rid_u == rid_v {
                    // Cannot move U-X after U, after X.
                    // Also if V is predecessor of U? Then U-X is already after V.
                    if v == u || v == x {
                        continue;
                    }
                    if indiv.successors[v] == u {
                        continue;
                    } // Already there
                    if indiv.predecessors[u] == v {
                        continue;
                    }
                }

                let dem_u = params.demands[u];
                let dem_x = params.demands[x];

                if rid_u != rid_v {
                    let load_u = self.calculate_route_load(indiv, params, u);
                    let load_v = self.calculate_route_load(indiv, params, v);

                    let u_prev = indiv.predecessors[u];
                    let x_next = indiv.successors[x];
                    let v_next = indiv.successors[v];

                    if v == u_prev {
                        continue;
                    }

                    let cost_rem = params.dist_matrix[u_prev][u]
                        + params.dist_matrix[x][x_next]
                        + params.dist_matrix[v][v_next];
                    let cost_add = params.dist_matrix[u_prev][x_next]
                        + params.dist_matrix[v][u]
                        + params.dist_matrix[x][v_next];

                    let delta_dist = (cost_add - cost_rem) * params.c_coeff;

                    let pen_u_old = self.get_penalty(load_u, params.vehicle_capacity);
                    let pen_v_old = self.get_penalty(load_v, params.vehicle_capacity);
                    let pen_u_new =
                        self.get_penalty(load_u - dem_u - dem_x, params.vehicle_capacity);
                    let pen_v_new =
                        self.get_penalty(load_v + dem_u + dem_x, params.vehicle_capacity);

                    let delta = delta_dist + (pen_u_new + pen_v_new - pen_u_old - pen_v_old);

                    if delta < -1e-5 {
                        self.apply_relocate_two(indiv, u, x, v, u_prev, x_next, v_next);
                        self.update_route_ids(indiv, params, route_ids, &[]);
                        return true;
                    }
                }
            }
        }
        false
    }

    fn apply_relocate_two(
        &self,
        indiv: &mut Individual,
        u: usize,
        x: usize,
        v: usize,
        u_prev: usize,
        x_next: usize,
        v_next: usize,
    ) {
        // Remove U-X
        indiv.successors[u_prev] = x_next;
        if x_next != 0 {
            indiv.predecessors[x_next] = u_prev;
        }

        // Insert U-X after V
        indiv.successors[v] = u;
        indiv.predecessors[u] = v;
        // Internal U->X is preserved
        indiv.successors[x] = v_next;
        if v_next != 0 {
            indiv.predecessors[v_next] = x;
        }
    }

    fn relocate_two_reversed(
        &self,
        indiv: &mut Individual,
        params: &Params,
        order_nodes: &[usize],
        route_ids: &mut Vec<usize>,
    ) -> bool {
        for &u in order_nodes {
            let x = indiv.successors[u];
            if x == 0 {
                continue;
            }

            let rid_u = route_ids[u];

            for &v in &params.correlated_vertices[u] {
                let rid_v = route_ids[v];

                if rid_u == rid_v {
                    if v == u || v == x {
                        continue;
                    }
                    if indiv.successors[v] == u {
                        continue;
                    }
                }

                let dem_u = params.demands[u];
                let dem_x = params.demands[x];

                if rid_u != rid_v {
                    let load_u = self.calculate_route_load(indiv, params, u);
                    let load_v = self.calculate_route_load(indiv, params, v);

                    let u_prev = indiv.predecessors[u];
                    let x_next = indiv.successors[x];
                    let v_next = indiv.successors[v];

                    if v == u_prev {
                        continue;
                    }

                    let cost_rem = params.dist_matrix[u_prev][u]
                        + params.dist_matrix[u][x]
                        + params.dist_matrix[x][x_next]
                        + params.dist_matrix[v][v_next];
                    let cost_add = params.dist_matrix[u_prev][x_next]
                        + params.dist_matrix[v][x]
                        + params.dist_matrix[x][u]
                        + params.dist_matrix[u][v_next];

                    let delta_dist = (cost_add - cost_rem) * params.c_coeff;

                    let pen_u_old = self.get_penalty(load_u, params.vehicle_capacity);
                    let pen_v_old = self.get_penalty(load_v, params.vehicle_capacity);
                    let pen_u_new =
                        self.get_penalty(load_u - dem_u - dem_x, params.vehicle_capacity);
                    let pen_v_new =
                        self.get_penalty(load_v + dem_u + dem_x, params.vehicle_capacity);

                    let delta = delta_dist + (pen_u_new + pen_v_new - pen_u_old - pen_v_old);

                    if delta < -1e-5 {
                        self.apply_relocate_two_reversed(indiv, u, x, v, u_prev, x_next, v_next);
                        self.update_route_ids(indiv, params, route_ids, &[]);
                        return true;
                    }
                } else {
                    // Intra logic for reversed
                    let u_prev = indiv.predecessors[u];
                    if v == u_prev {
                        continue;
                    }

                    let x_next = indiv.successors[x];
                    let v_next = indiv.successors[v];

                    let cost_rem = params.dist_matrix[u_prev][u]
                        + params.dist_matrix[u][x]
                        + params.dist_matrix[x][x_next]
                        + params.dist_matrix[v][v_next];
                    let cost_add = params.dist_matrix[u_prev][x_next]
                        + params.dist_matrix[v][x]
                        + params.dist_matrix[x][u]
                        + params.dist_matrix[u][v_next];

                    let delta = (cost_add - cost_rem) * params.c_coeff;
                    if delta < -1e-5 {
                        self.apply_relocate_two_reversed(indiv, u, x, v, u_prev, x_next, v_next);
                        return true;
                    }
                }
            }
        }
        false
    }

    fn apply_relocate_two_reversed(
        &self,
        indiv: &mut Individual,
        u: usize,
        x: usize,
        v: usize,
        u_prev: usize,
        x_next: usize,
        v_next: usize,
    ) {
        // Remove U-X
        indiv.successors[u_prev] = x_next;
        if x_next != 0 {
            indiv.predecessors[x_next] = u_prev;
        }

        // Insert X-U after V
        indiv.successors[v] = x;
        indiv.predecessors[x] = v;

        indiv.successors[x] = u;
        indiv.predecessors[u] = x;

        indiv.successors[u] = v_next;
        if v_next != 0 {
            indiv.predecessors[v_next] = u;
        }
    }

    // --- SWAP ---
    fn swap(
        &self,
        indiv: &mut Individual,
        params: &Params,
        order_nodes: &[usize],
        route_ids: &mut Vec<usize>,
    ) -> bool {
        for &u in order_nodes {
            let rid_u = route_ids[u];
            for &v in &params.correlated_vertices[u] {
                let rid_v = route_ids[v];

                if rid_u > rid_v {
                    continue;
                }
                if rid_u == rid_v && u == v {
                    continue;
                }

                let dem_u = params.demands[u];
                let dem_v = params.demands[v];

                if rid_u != rid_v {
                    let load_u = self.calculate_route_load(indiv, params, u);
                    let load_v = self.calculate_route_load(indiv, params, v);

                    let u_prev = indiv.predecessors[u];
                    let u_next = indiv.successors[u];
                    let v_prev = indiv.predecessors[v];
                    let v_next = indiv.successors[v];

                    let cost_rem_u = params.dist_matrix[u_prev][u] + params.dist_matrix[u][u_next];
                    let cost_rem_v = params.dist_matrix[v_prev][v] + params.dist_matrix[v][v_next];
                    let cost_add_u = params.dist_matrix[v_prev][u] + params.dist_matrix[u][v_next];
                    let cost_add_v = params.dist_matrix[u_prev][v] + params.dist_matrix[v][u_next];

                    let delta_dist =
                        (cost_add_u + cost_add_v - cost_rem_u - cost_rem_v) * params.c_coeff;

                    let pen_u_old = self.get_penalty(load_u, params.vehicle_capacity);
                    let pen_v_old = self.get_penalty(load_v, params.vehicle_capacity);
                    let pen_u_new =
                        self.get_penalty(load_u - dem_u + dem_v, params.vehicle_capacity);
                    let pen_v_new =
                        self.get_penalty(load_v - dem_v + dem_u, params.vehicle_capacity);

                    let delta = delta_dist + (pen_u_new + pen_v_new - pen_u_old - pen_v_old);
                    if delta < -1e-5 {
                        self.apply_swap(indiv, u, v, u_prev, u_next, v_prev, v_next);
                        self.update_route_ids(indiv, params, route_ids, &[]);
                        return true;
                    }
                } else {
                    // Intra-route swap
                    let u_prev = indiv.predecessors[u];
                    let u_next = indiv.successors[u];
                    let v_prev = indiv.predecessors[v];
                    let v_next = indiv.successors[v];

                    let delta = if u_next == v {
                        let cost_rem = params.dist_matrix[u_prev][u]
                            + params.dist_matrix[u][v]
                            + params.dist_matrix[v][v_next];
                        let cost_add = params.dist_matrix[u_prev][v]
                            + params.dist_matrix[v][u]
                            + params.dist_matrix[u][v_next];
                        cost_add - cost_rem
                    } else if v_next == u {
                        let cost_rem = params.dist_matrix[v_prev][v]
                            + params.dist_matrix[v][u]
                            + params.dist_matrix[u][u_next];
                        let cost_add = params.dist_matrix[v_prev][u]
                            + params.dist_matrix[u][v]
                            + params.dist_matrix[v][u_next];
                        cost_add - cost_rem
                    } else {
                        let cost_rem_u =
                            params.dist_matrix[u_prev][u] + params.dist_matrix[u][u_next];
                        let cost_rem_v =
                            params.dist_matrix[v_prev][v] + params.dist_matrix[v][v_next];
                        let cost_add_u =
                            params.dist_matrix[v_prev][u] + params.dist_matrix[u][v_next];
                        let cost_add_v =
                            params.dist_matrix[u_prev][v] + params.dist_matrix[v][u_next];
                        cost_add_u + cost_add_v - cost_rem_u - cost_rem_v
                    };

                    if delta * params.c_coeff < -1e-5 {
                        self.apply_swap(indiv, u, v, u_prev, u_next, v_prev, v_next);
                        return true;
                    }
                }
            }
        }
        false
    }

    fn apply_swap(
        &self,
        indiv: &mut Individual,
        u: usize,
        v: usize,
        u_prev: usize,
        u_next: usize,
        v_prev: usize,
        v_next: usize,
    ) {
        if u_next == v {
            indiv.successors[u_prev] = v;
            if v != 0 {
                indiv.predecessors[v] = u_prev;
            }
            indiv.successors[v] = u;
            indiv.predecessors[u] = v;
            indiv.successors[u] = v_next;
            if v_next != 0 {
                indiv.predecessors[v_next] = u;
            }
        } else if v_next == u {
            indiv.successors[v_prev] = u;
            if u != 0 {
                indiv.predecessors[u] = v_prev;
            }
            indiv.successors[u] = v;
            indiv.predecessors[v] = u;
            indiv.successors[v] = u_next;
            if u_next != 0 {
                indiv.predecessors[u_next] = v;
            }
        } else {
            indiv.successors[u_prev] = v;
            if v != 0 {
                indiv.predecessors[v] = u_prev;
            }
            indiv.successors[v] = u_next;
            if u_next != 0 {
                indiv.predecessors[u_next] = v;
            }

            indiv.successors[v_prev] = u;
            if u != 0 {
                indiv.predecessors[u] = v_prev;
            }
            indiv.successors[u] = v_next;
            if v_next != 0 {
                indiv.predecessors[v_next] = u;
            }
        }
    }

    fn swap_two_one(
        &self,
        indiv: &mut Individual,
        params: &Params,
        order_nodes: &[usize],
        route_ids: &mut Vec<usize>,
    ) -> bool {
        for &u in order_nodes {
            let x = indiv.successors[u];
            if x == 0 {
                continue;
            }
            let rid_u = route_ids[u];
            let dem_u = params.demands[u];
            let dem_x = params.demands[x];

            for &v in &params.correlated_vertices[u] {
                let rid_v = route_ids[v];

                if rid_u == rid_v {
                    if v == u || v == x {
                        continue;
                    }
                    let u_prev = indiv.predecessors[u];
                    if v == u_prev {
                        continue;
                    }
                    let v_next = indiv.successors[v];
                    if u == v_next {
                        continue;
                    }
                }

                let dem_v = params.demands[v];

                if rid_u != rid_v {
                    let load_u = self.calculate_route_load(indiv, params, u);
                    let load_v = self.calculate_route_load(indiv, params, v);

                    let u_prev = indiv.predecessors[u];
                    let x_next = indiv.successors[x];
                    let v_prev = indiv.predecessors[v];
                    let v_next = indiv.successors[v];

                    let cost_rem = params.dist_matrix[u_prev][u]
                        + params.dist_matrix[x][x_next]
                        + params.dist_matrix[v_prev][v]
                        + params.dist_matrix[v][v_next];
                    let cost_add = params.dist_matrix[u_prev][v]
                        + params.dist_matrix[v][x_next]
                        + params.dist_matrix[v_prev][u]
                        + params.dist_matrix[x][v_next];

                    let delta_dist = (cost_add - cost_rem) * params.c_coeff;

                    let pen_u_old = self.get_penalty(load_u, params.vehicle_capacity);
                    let pen_v_old = self.get_penalty(load_v, params.vehicle_capacity);
                    let pen_u_new =
                        self.get_penalty(load_u - dem_u - dem_x + dem_v, params.vehicle_capacity);
                    let pen_v_new =
                        self.get_penalty(load_v - dem_v + dem_u + dem_x, params.vehicle_capacity);

                    let delta = delta_dist + (pen_u_new + pen_v_new - pen_u_old - pen_v_old);

                    if delta < -1e-5 {
                        self.apply_swap_two_one(indiv, u, x, v, u_prev, x_next, v_prev, v_next);
                        self.update_route_ids(indiv, params, route_ids, &[]);
                        return true;
                    }
                } else {
                    // Intra logic
                    let u_prev = indiv.predecessors[u];
                    let x_next = indiv.successors[x];
                    let v_prev = indiv.predecessors[v];
                    let v_next = indiv.successors[v];

                    let cost_rem = params.dist_matrix[u_prev][u]
                        + params.dist_matrix[x][x_next]
                        + params.dist_matrix[v_prev][v]
                        + params.dist_matrix[v][v_next];
                    let cost_add = params.dist_matrix[u_prev][v]
                        + params.dist_matrix[v][x_next]
                        + params.dist_matrix[v_prev][u]
                        + params.dist_matrix[x][v_next];

                    let delta = (cost_add - cost_rem) * params.c_coeff;

                    if delta < -1e-5 {
                        self.apply_swap_two_one(indiv, u, x, v, u_prev, x_next, v_prev, v_next);
                        return true;
                    }
                }
            }
        }
        false
    }

    fn apply_swap_two_one(
        &self,
        indiv: &mut Individual,
        u: usize,
        x: usize,
        v: usize,
        u_prev: usize,
        x_next: usize,
        v_prev: usize,
        v_next: usize,
    ) {
        // Place V at U's spot: u_prev -> V -> x_next
        indiv.successors[u_prev] = v;
        if v != 0 {
            indiv.predecessors[v] = u_prev;
        }
        indiv.successors[v] = x_next;
        if x_next != 0 {
            indiv.predecessors[x_next] = v;
        }

        // Place U-X at V's spot: v_prev -> U -> X -> v_next
        indiv.successors[v_prev] = u;
        if u != 0 {
            indiv.predecessors[u] = v_prev;
        }
        // U->X is internal, already linked
        indiv.successors[x] = v_next;
        if v_next != 0 {
            indiv.predecessors[v_next] = x;
        }
    }

    fn swap_two_two(
        &self,
        indiv: &mut Individual,
        params: &Params,
        order_nodes: &[usize],
        route_ids: &mut Vec<usize>,
    ) -> bool {
        for &u in order_nodes {
            let x = indiv.successors[u];
            if x == 0 {
                continue;
            }
            let rid_u = route_ids[u];
            let dem_u = params.demands[u];
            let dem_x = params.demands[x];

            for &v in &params.correlated_vertices[u] {
                let y = indiv.successors[v];
                if y == 0 {
                    continue;
                }

                let rid_v = route_ids[v];

                // Avoid redundancy
                if rid_u > rid_v {
                    continue;
                }
                if rid_u == rid_v {
                    // Intra-route: avoid double check and overlap
                    if u == v {
                        continue;
                    }
                    // Overlap logic:
                    // U, X, V, Y.
                    if v == x {
                        continue;
                    } // Adjacent U->X(V)->Y
                    if u == y {
                        continue;
                    } // Adjacent V->Y(U)->X
                }

                let dem_v = params.demands[v];
                let dem_y = params.demands[y];

                if rid_u != rid_v {
                    let load_u = self.calculate_route_load(indiv, params, u);
                    let load_v = self.calculate_route_load(indiv, params, v);

                    let u_prev = indiv.predecessors[u];
                    let x_next = indiv.successors[x];
                    let v_prev = indiv.predecessors[v];
                    let y_next = indiv.successors[y];

                    // Special case: adjacent blocks?
                    // U->X -> V->Y.
                    // u_prev -> U. X->V. V_prev=X.
                    // We don't support swapping adjacent blocks directly here to simplify. C++ likely prunes too.
                    if v == x_next || u == y_next {
                        continue;
                    }

                    let cost_rem = params.dist_matrix[u_prev][u]
                        + params.dist_matrix[x][x_next]
                        + params.dist_matrix[v_prev][v]
                        + params.dist_matrix[y][y_next];
                    let cost_add = params.dist_matrix[u_prev][v]
                        + params.dist_matrix[y][x_next]
                        + params.dist_matrix[v_prev][u]
                        + params.dist_matrix[x][y_next];

                    let delta_dist = (cost_add - cost_rem) * params.c_coeff;

                    let pen_u_old = self.get_penalty(load_u, params.vehicle_capacity);
                    let pen_v_old = self.get_penalty(load_v, params.vehicle_capacity);
                    let pen_u_new = self.get_penalty(
                        load_u - dem_u - dem_x + dem_v + dem_y,
                        params.vehicle_capacity,
                    );
                    let pen_v_new = self.get_penalty(
                        load_v - dem_v - dem_y + dem_u + dem_x,
                        params.vehicle_capacity,
                    );

                    let delta = delta_dist + (pen_u_new + pen_v_new - pen_u_old - pen_v_old);

                    if delta < -1e-5 {
                        self.apply_swap_two_two(indiv, u, x, v, y, u_prev, x_next, v_prev, y_next);
                        self.update_route_ids(indiv, params, route_ids, &[]);
                        return true;
                    }
                }
            }
        }
        false
    }

    fn apply_swap_two_two(
        &self,
        indiv: &mut Individual,
        u: usize,
        x: usize,
        v: usize,
        y: usize,
        u_prev: usize,
        x_next: usize,
        v_prev: usize,
        y_next: usize,
    ) {
        // Place V-Y at U's spot: u_prev -> V -> Y -> x_next
        indiv.successors[u_prev] = v;
        if v != 0 {
            indiv.predecessors[v] = u_prev;
        }
        // V->Y internal kept
        indiv.successors[y] = x_next;
        if x_next != 0 {
            indiv.predecessors[x_next] = y;
        }

        // Place U-X at V's spot: v_prev -> U -> X -> y_next
        indiv.successors[v_prev] = u;
        if u != 0 {
            indiv.predecessors[u] = v_prev;
        }
        // U->X internal kept
        indiv.successors[x] = y_next;
        if y_next != 0 {
            indiv.predecessors[y_next] = x;
        }
    }

    // --- 2-OPT Intra ---
    fn two_opt_intra(
        &self,
        indiv: &mut Individual,
        params: &Params,
        order_nodes: &[usize],
        route_ids: &mut Vec<usize>,
    ) -> bool {
        for &u in order_nodes {
            let rid_u = route_ids[u];
            for &v in &params.correlated_vertices[u] {
                let rid_v = route_ids[v];
                if rid_u != rid_v {
                    continue;
                }
                if u == v {
                    continue;
                }

                let u_next = indiv.successors[u];
                let v_next = indiv.successors[v];

                if u_next == v || v_next == u {
                    continue;
                }

                let cost_rem = params.dist_matrix[u][u_next] + params.dist_matrix[v][v_next];
                let cost_add = params.dist_matrix[u][v] + params.dist_matrix[u_next][v_next];

                let delta = (cost_add - cost_rem) * params.c_coeff;

                if delta < -1e-5 {
                    if self.is_forward(indiv, u, v) {
                        self.apply_two_opt(indiv, u, v, u_next, v_next);
                        return true;
                    }
                }
            }
        }
        false
    }

    fn is_forward(&self, indiv: &Individual, u: usize, v: usize) -> bool {
        let mut curr = u;
        while curr != 0 {
            if curr == v {
                return true;
            }
            curr = indiv.successors[curr];
        }
        false
    }

    fn apply_two_opt(
        &self,
        indiv: &mut Individual,
        u: usize,
        v: usize,
        u_next: usize,
        v_next: usize,
    ) {
        let mut curr = u_next;
        let mut reversed_nodes = Vec::new();
        while curr != v_next && curr != 0 {
            reversed_nodes.push(curr);
            if curr == v {
                break;
            }
            curr = indiv.successors[curr];
        }

        if let Some(last) = reversed_nodes.last() {
            indiv.successors[u] = *last;
            indiv.predecessors[*last] = u;
        }

        for i in (1..reversed_nodes.len()).rev() {
            let node = reversed_nodes[i];
            let next = reversed_nodes[i - 1];
            indiv.successors[node] = next;
            indiv.predecessors[next] = node;
        }

        if let Some(first) = reversed_nodes.first() {
            indiv.successors[*first] = v_next;
            if v_next != 0 {
                indiv.predecessors[v_next] = *first;
            }
        }
    }

    fn two_opt_star(
        &self,
        _indiv: &mut Individual,
        _params: &Params,
        _order_nodes: &[usize],
        _route_ids: &mut Vec<usize>,
    ) -> bool {
        false
    }

    // --- SWAP* Operator ---
    fn swap_star(
        &self,
        indiv: &mut Individual,
        params: &Params,
        route_ids: &mut Vec<usize>,
    ) -> bool {
        let improved = false;
        let n_routes = indiv.chrom_r.len();

        struct RouteData {
            r_idx: usize,
            sector: CircleSector,
            load: f64,
            nodes: Vec<usize>,
        }

        // Construct RouteData from linked list for accurate fresh state
        // Scan for heads
        let mut routes_data = Vec::with_capacity(n_routes + 5);
        let mut visited = vec![false; params.n_clients + 1];

        for i in 1..=params.n_clients {
            if !visited[i] && indiv.predecessors[i] == 0 {
                let mut nodes = Vec::new();
                let mut curr = i;
                let mut load = 0.0;
                while curr != 0 {
                    visited[curr] = true;
                    nodes.push(curr);
                    load += params.demands[curr];
                    curr = indiv.successors[curr];
                }

                // Route must have nodes if we are here
                let mut sector = CircleSector::new(params.polar_angles[nodes[0]]);
                for &node in &nodes {
                    sector.extend(params.polar_angles[node]);
                }

                routes_data.push(RouteData {
                    r_idx: routes_data.len(),
                    sector,
                    load,
                    nodes,
                });
            }
        }

        // O(N^2) Loop over pairs with geometric pruning
        for r1_idx in 0..routes_data.len() {
            for r2_idx in (r1_idx + 1)..routes_data.len() {
                let r1 = &routes_data[r1_idx];
                let r2 = &routes_data[r2_idx];

                if !CircleSector::overlap(&r1.sector, &r2.sector) {
                    continue;
                }

                // Integrity checks (cheap integer comparison)
                assert_eq!(r1.r_idx, r1_idx);
                assert_eq!(r2.r_idx, r2_idx);

                if self.swap_star_pair(indiv, params, &r1.nodes, &r2.nodes, r1.load, r2.load) {
                    // If move applied, we should update route_ids as other operators rely on them.
                    self.update_route_ids(indiv, params, route_ids, &[]);
                    return true; // Restart LS
                }
            }
        }

        improved
    }

    fn swap_star_pair(
        &self,
        indiv: &mut Individual,
        params: &Params,
        r1_nodes: &Vec<usize>,
        r2_nodes: &Vec<usize>,
        load1: f64,
        load2: f64,
    ) -> bool {
        for &u in r1_nodes {
            let dem_u = params.demands[u];
            // Unused result of cost_u_in_2 ignored to prevent warning
            // let (cost_u_in_2, pos_u_in_2) = self.find_best_insertion_cost(u, r2_nodes, params);

            for &v in r2_nodes {
                let dem_v = params.demands[v];

                // Soft constraint: Calculate penalty change
                let pen_1_old = self.get_penalty(load1, params.vehicle_capacity);
                let pen_2_old = self.get_penalty(load2, params.vehicle_capacity);
                let pen_1_new = self.get_penalty(load1 - dem_u + dem_v, params.vehicle_capacity);
                let pen_2_new = self.get_penalty(load2 - dem_v + dem_u, params.vehicle_capacity);

                let (cost_v_in_1, pos_v_in_1) =
                    self.find_best_insertion_without(v, r1_nodes, u, params);
                // We ideally want insertion cost of U into R2 WITHOUT V.
                let (cost_u_in_2_real, pos_u_in_2_real) =
                    self.find_best_insertion_without(u, r2_nodes, v, params);

                let cost_rem_u = self.removal_cost(indiv, params, u);
                let cost_rem_v = self.removal_cost(indiv, params, v);

                let delta_dist = cost_v_in_1 + cost_u_in_2_real - cost_rem_u - cost_rem_v;
                let delta =
                    delta_dist * params.c_coeff + (pen_1_new + pen_2_new - pen_1_old - pen_2_old);

                if delta < -1e-5 {
                    self.apply_swap_star_move(indiv, u, v, pos_v_in_1, pos_u_in_2_real);
                    return true;
                }
            }
        }
        false
    }

    fn find_best_insertion_without(
        &self,
        node: usize,
        route: &Vec<usize>,
        ignore_node: usize,
        params: &Params,
    ) -> (f64, usize) {
        let mut best_cost = f64::INFINITY;
        let mut best_after = 0;

        let mut prev = 0;
        for &curr in route {
            if curr == ignore_node {
                continue;
            }
            let delta = params.dist_matrix[prev][node] + params.dist_matrix[node][curr]
                - params.dist_matrix[prev][curr];
            if delta < best_cost {
                best_cost = delta;
                best_after = prev;
            }
            prev = curr;
        }
        // Last node -> 0
        let curr = 0;
        let delta = params.dist_matrix[prev][node] + params.dist_matrix[node][curr]
            - params.dist_matrix[prev][curr];
        if delta < best_cost {
            best_cost = delta;
            best_after = prev;
        }

        (best_cost, best_after)
    }

    fn removal_cost(&self, indiv: &Individual, params: &Params, u: usize) -> f64 {
        let u_prev = indiv.predecessors[u];
        let u_next = indiv.successors[u];
        params.dist_matrix[u_prev][u] + params.dist_matrix[u][u_next]
            - params.dist_matrix[u_prev][u_next]
    }

    fn apply_swap_star_move(
        &self,
        indiv: &mut Individual,
        u: usize,
        v: usize,
        after_v_in_r1: usize,
        after_u_in_r2: usize,
    ) {
        // Remove u
        let u_prev = indiv.predecessors[u];
        let u_next = indiv.successors[u];
        indiv.successors[u_prev] = u_next;
        if u_next != 0 {
            indiv.predecessors[u_next] = u_prev;
        }

        // Remove v
        let v_prev = indiv.predecessors[v];
        let v_next = indiv.successors[v];
        indiv.successors[v_prev] = v_next;
        if v_next != 0 {
            indiv.predecessors[v_next] = v_prev;
        }

        // Insert v in r1
        let target_1 = after_v_in_r1;
        let target_1_next = indiv.successors[target_1];

        indiv.successors[target_1] = v;
        indiv.predecessors[v] = target_1;
        indiv.successors[v] = target_1_next;
        if target_1_next != 0 {
            indiv.predecessors[target_1_next] = v;
        }

        // Insert u in r2
        let target_2 = after_u_in_r2;
        let target_2_next = indiv.successors[target_2];

        indiv.successors[target_2] = u;
        indiv.predecessors[u] = target_2;
        indiv.successors[u] = target_2_next;
        if target_2_next != 0 {
            indiv.predecessors[target_2_next] = u;
        }
    }
}
