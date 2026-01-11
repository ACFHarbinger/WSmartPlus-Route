/*!
 * Population management with diversity control.
 *
 * Maintains separate pools of feasible and infeasible solutions,
 * using biased fitness (cost + diversity) for survivor selection.
 */

use crate::individual::Individual;
use crate::params::Params;
use std::cmp::Ordering;

/**
 * Population of solutions with diversity management.
 *
 * # Structure
 *
 * - **Feasible pool**: Solutions satisfying all constraints
 * - **Infeasible pool**: Solutions with constraint violations
 * - **Best solution**: Best feasible solution found so far
 *
 * # Biased Fitness
 *
 * Each individual's fitness is a combination of:
 * - **Cost rank**: Position when sorted by penalized cost
 * - **Diversity rank**: Average distance to `nb_close` nearest neighbors
 *
 * This prevents premature convergence by rewarding diverse solutions.
 */
pub struct Population {
    pub feasible: Vec<Individual>,
    pub infeasible: Vec<Individual>,
    pub best_solution: Option<Individual>,
    _max_size: usize,
}

impl Population {
    /** Creates a new empty population. */
    pub fn new(params: &Params) -> Self {
        Self {
            feasible: Vec::with_capacity(params.ap.mu + params.ap.lambda),
            infeasible: Vec::with_capacity(params.ap.mu + params.ap.lambda),
            best_solution: None,
            _max_size: params.ap.mu + params.ap.lambda,
        }
    }

    /**
     * Adds an individual to the population.
     *
     * - Updates best solution if this is a better feasible solution
     * - Adds to feasible or infeasible pool based on constraint satisfaction
     * - Skips if an identical solution (same penalized cost) already exists
     */
    pub fn add(&mut self, indiv: Individual) {
        if indiv.eval.is_feasible {
            match &self.best_solution {
                Some(best) => {
                    if indiv.eval.penalized_cost < best.eval.penalized_cost {
                        self.best_solution = Some(indiv.clone());
                    }
                }
                None => {
                    self.best_solution = Some(indiv.clone());
                }
            }
        }

        let list = if indiv.eval.is_feasible {
            &mut self.feasible
        } else {
            &mut self.infeasible
        };

        for other in list.iter() {
            if (other.eval.penalized_cost - indiv.eval.penalized_cost).abs() < 1e-5 {
                return;
            }
        }

        list.push(indiv);
    }

    /**
     * Updates biased fitness for all individuals.
     *
     * # Steps
     *
     * 1. Rank solutions by penalized cost (cost rank)
     * 2. Calculate diversity contribution (average distance to nearest neighbors)
     * 3. Rank solutions by diversity (diversity rank)
     * 4. Compute biased fitness: `cost_rank + (1 - elite_ratio) × diversity_rank`
     *
     * Elite solutions (top `nb_elite`) get lower diversity weight.
     */
    pub fn update_biased_fitnesses(&mut self, params: &Params) {
        // Fix E0502: Use associated functions to avoid borrowing self while borrowing fields
        Self::rank_by_cost_static(&mut self.feasible);
        Self::rank_by_cost_static(&mut self.infeasible);

        Self::compute_diversity_static(&mut self.feasible, params);
        Self::compute_diversity_static(&mut self.infeasible, params);

        Self::assign_biased_fitness_static(&mut self.feasible, params);
        Self::assign_biased_fitness_static(&mut self.infeasible, params);
    }

    fn rank_by_cost_static(list: &mut Vec<Individual>) {
        list.sort_by(|a, b| {
            a.eval
                .penalized_cost
                .partial_cmp(&b.eval.penalized_cost)
                .unwrap_or(Ordering::Equal)
        });
    }

    fn compute_diversity_static(list: &mut Vec<Individual>, params: &Params) {
        let size = list.len();
        if size < 2 {
            return;
        }

        let n_clients = params.n_clients;
        let nb_close = params.ap.nb_close;

        // Collect routes for distance calculation to avoid multiple borrows
        let all_routes: Vec<Vec<Vec<usize>>> = list.iter().map(|ind| ind.chrom_r.clone()).collect();

        for i in 0..size {
            let mut distances: Vec<f64> = Vec::with_capacity(size);
            for j in 0..size {
                if i == j {
                    continue;
                }
                let dist =
                    Self::broken_pairs_distance_static(&all_routes[i], &all_routes[j], n_clients);
                distances.push(dist);
            }
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            let k = std::cmp::min(nb_close, distances.len());
            let sum_dist: f64 = distances.iter().take(k).sum();
            list[i].diversity_contribution = if k > 0 { sum_dist / k as f64 } else { 0.0 };
        }
    }

    fn assign_biased_fitness_static(list: &mut Vec<Individual>, params: &Params) {
        let size = list.len();
        if size == 0 {
            return;
        }

        let mut pointers: Vec<usize> = (0..size).collect();
        pointers.sort_by(|&a, &b| {
            list[b]
                .diversity_contribution
                .partial_cmp(&list[a].diversity_contribution)
                .unwrap_or(Ordering::Equal)
        });

        let mut diversity_ranks = vec![0; size];
        for (rank, &idx) in pointers.iter().enumerate() {
            diversity_ranks[idx] = rank;
        }

        let elite_ratio = params.ap.nb_elite as f64 / params.ap.mu as f64;
        for i in 0..size {
            list[i].biased_fitness = (i as f64) + (1.0 - elite_ratio) * (diversity_ranks[i] as f64);
        }
    }

    fn broken_pairs_distance_static(
        routes1: &[Vec<usize>],
        routes2: &[Vec<usize>],
        _n_clients: usize,
    ) -> f64 {
        let mut edges1 = std::collections::HashSet::new();
        for r in routes1 {
            if r.is_empty() {
                continue;
            }
            edges1.insert(Self::normalize_edge(0, r[0]));
            for k in 0..r.len() - 1 {
                edges1.insert(Self::normalize_edge(r[k], r[k + 1]));
            }
            edges1.insert(Self::normalize_edge(r[r.len() - 1], 0));
        }

        let mut common = 0;
        for r in routes2 {
            if r.is_empty() {
                continue;
            }
            if edges1.contains(&Self::normalize_edge(0, r[0])) {
                common += 1;
            }
            for k in 0..r.len() - 1 {
                if edges1.contains(&Self::normalize_edge(r[k], r[k + 1])) {
                    common += 1;
                }
            }
            if edges1.contains(&Self::normalize_edge(r[r.len() - 1], 0)) {
                common += 1;
            }
        }

        (edges1.len() - common) as f64
    }

    fn normalize_edge(u: usize, v: usize) -> (usize, usize) {
        if u < v {
            (u, v)
        } else {
            (v, u)
        }
    }

    pub fn prune(&mut self, mu: usize) {
        self.feasible.sort_by(|a, b| {
            a.biased_fitness
                .partial_cmp(&b.biased_fitness)
                .unwrap_or(Ordering::Equal)
        });
        self.infeasible.sort_by(|a, b| {
            a.biased_fitness
                .partial_cmp(&b.biased_fitness)
                .unwrap_or(Ordering::Equal)
        });

        if self.feasible.len() > mu {
            self.feasible.truncate(mu);
        }
        if self.infeasible.len() > mu {
            self.infeasible.truncate(mu);
        }
    }

    pub fn get_random_pair(&self) -> Option<(&Individual, &Individual)> {
        let total_size = self.feasible.len() + self.infeasible.len();
        if total_size < 2 {
            return None;
        }

        use rand::Rng;
        let mut rng = rand::rng();

        // Fix logic: Use mut for closure as it captures and mutates rng
        let mut get_tournament = || -> &Individual {
            let idx1 = rng.random_range(0..total_size);
            let idx2 = rng.random_range(0..total_size);

            let ind1 = if idx1 < self.feasible.len() {
                &self.feasible[idx1]
            } else {
                &self.infeasible[idx1 - self.feasible.len()]
            };
            let ind2 = if idx2 < self.feasible.len() {
                &self.feasible[idx2]
            } else {
                &self.infeasible[idx2 - self.feasible.len()]
            };

            if ind1.biased_fitness < ind2.biased_fitness {
                ind1
            } else {
                ind2
            }
        };

        let p1 = get_tournament();
        let p2 = get_tournament();
        Some((p1, p2))
    }
}
