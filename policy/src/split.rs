/*!
 * Split algorithm for giant tour decomposition.
 *
 * Converts a giant tour (permutation of customers) into feasible vehicle routes
 * using dynamic programming to maximize profit.
 */

use crate::individual::Individual;
use crate::params::Params;
use std::sync::Arc;

/**
 * Split algorithm for route construction.
 *
 * # Problem
 *
 * Given a giant tour (permutation of customers), find the optimal way to
 * partition it into vehicle routes to maximize profit while respecting capacity.
 *
 * # Variants
 *
 * - **Simple Split**: Unlimited fleet, maximizes total profit
 * - **Limited Fleet Split**: Fixed `max_vehicles`, uses 2D DP
 */
pub struct Split {
    params: Arc<Params>,
}

impl Split {
    /** Creates a new Split algorithm instance. */
    pub fn new(params: Arc<Params>) -> Self {
        Self { params }
    }

    /**
     * Splits the giant tour into optimal routes.
     *
     * Chooses between Simple Split (unlimited fleet) and Limited Fleet Split
     * based on `max_vehicles` parameter.
     */
    pub fn split(&self, indiv: &mut Individual, penalty_capacity: f64) {
        if self.params.max_vehicles > 0 {
            self.split_lf(indiv, penalty_capacity);
        } else {
            self.split_simple(indiv, penalty_capacity);
        }
    }

    /**
     * Simple Split: Unlimited fleet, maximize profit.
     *
     * # Dynamic Programming
     *
     * ```text
     * v[i] = maximum profit serving first i customers
     * v[i] = max over j < i of:
     *         v[j] + profit(route from tour[j..i])
     *
     * profit(route) = revenue - cost - penalties
     * ```
     *
     * # Complexity
     *
     * O(N²) where N is the number of customers.
     */
    fn split_simple(&self, indiv: &mut Individual, penalty_capacity: f64) {
        let n = self.params.n_clients;
        let tour = &indiv.giant_tour;

        let mut v = vec![-f64::INFINITY; n + 1];
        let mut p = vec![0; n + 1];
        v[0] = 0.0;

        for i in 0..n {
            let mut load = 0.0;
            let mut dist = 0.0;
            let mut revenue = 0.0;

            for j in (i + 1)..=n {
                let node_idx = tour[j - 1];
                load += self.params.demands[node_idx];

                // Soft constraint: Break only if load is way too high (optimization)
                if load > self.params.vehicle_capacity * 2.0 {
                    break;
                }

                revenue += self.params.demands[node_idx] * self.params.r_coeff;

                if j == i + 1 {
                    dist = self.params.dist_matrix[0][node_idx];
                } else {
                    let prev_idx = tour[j - 2];
                    dist += self.params.dist_matrix[prev_idx][node_idx];
                }

                let cost_segment =
                    (dist + self.params.dist_matrix[node_idx][0]) * self.params.c_coeff;

                let penalty = if load > self.params.vehicle_capacity {
                    (load - self.params.vehicle_capacity) * penalty_capacity
                } else {
                    0.0
                };

                let profit_segment = revenue - cost_segment - penalty;

                if v[i] + profit_segment > v[j] {
                    v[j] = v[i] + profit_segment;
                    p[j] = i;
                }
            }
        }

        self.reconstruct(indiv, n, &p);
    }

    /** Limited Fleet Split (Fixed max_vehicles) */
    fn split_lf(&self, indiv: &mut Individual, penalty_capacity: f64) {
        let n = self.params.n_clients;
        let tour = &indiv.giant_tour;
        let max_k = self.params.max_vehicles;

        // v[k][i] = max profit serving first i clients with k routes
        let mut v = vec![vec![-f64::INFINITY; n + 1]; max_k + 1];
        let mut p = vec![vec![0; n + 1]; max_k + 1];

        v[0][0] = 0.0;

        for k in 0..max_k {
            for i in 0..n {
                if v[k][i] == -f64::INFINITY {
                    continue;
                }

                let mut load = 0.0;
                let mut dist = 0.0;
                let mut revenue = 0.0;

                for j in (i + 1)..=n {
                    let node_idx = tour[j - 1];
                    load += self.params.demands[node_idx];

                    // Soft constraint: Break only if load is way too high (optimization)
                    if load > self.params.vehicle_capacity * 10.0 {
                        break;
                    }

                    revenue += self.params.demands[node_idx] * self.params.r_coeff;

                    if j == i + 1 {
                        dist = self.params.dist_matrix[0][node_idx];
                    } else {
                        let prev_idx = tour[j - 2];
                        dist += self.params.dist_matrix[prev_idx][node_idx];
                    }

                    let cost_segment =
                        (dist + self.params.dist_matrix[node_idx][0]) * self.params.c_coeff;

                    let penalty = if load > self.params.vehicle_capacity {
                        (load - self.params.vehicle_capacity) * penalty_capacity
                    } else {
                        0.0
                    };

                    let profit_segment = revenue - cost_segment - penalty;

                    if v[k][i] + profit_segment > v[k + 1][j] {
                        v[k + 1][j] = v[k][i] + profit_segment;
                        p[k + 1][j] = i;
                    }
                }
            }
        }

        // Find best number of vehicles (1..=max_k)
        let mut best_profit = -f64::INFINITY;
        let mut best_k = 0;
        for k in 1..=max_k {
            if v[k][n] > best_profit {
                best_profit = v[k][n];
                best_k = k;
            }
        }

        if best_profit == -f64::INFINITY {
            indiv.chrom_r.clear();
            return;
        }

        // Reconstruct from p[k]
        indiv.chrom_r.clear();
        let mut curr = n;
        let mut k = best_k;
        while curr > 0 && k > 0 {
            let prev = p[k][curr];
            let route = tour[prev..curr].to_vec();
            indiv.chrom_r.push(route);
            curr = prev;
            k -= 1;
        }
        indiv.chrom_r.reverse();
    }

    fn reconstruct(&self, indiv: &mut Individual, n: usize, p: &[usize]) {
        indiv.chrom_r.clear();
        let mut curr = n;
        // Safety check to prevent infinite loop if unreachable
        let mut visited = 0;
        while curr > 0 {
            let prev = p[curr];
            let route = indiv.giant_tour[prev..curr].to_vec();
            indiv.chrom_r.push(route);
            curr = prev;

            visited += 1;
            if visited > n + 1 {
                // Should not happen
                indiv.chrom_r.clear();
                return;
            }
        }
        indiv.chrom_r.reverse();
    }
}
