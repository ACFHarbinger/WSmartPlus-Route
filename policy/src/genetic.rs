use crate::individual::Individual;
use crate::local_search::LocalSearch;
use crate::params::Params;
use crate::population::Population;
use crate::split::Split;
use rand::prelude::*;
use std::sync::Arc;

pub struct Genetic {
    params: Arc<Params>,
    population: Population,
    split: Split,
    local_search: LocalSearch,
    penalty_capacity: f64,
}

impl Genetic {
    pub fn new(params: Arc<Params>) -> Self {
        Self {
            population: Population::new(&params),
            split: Split::new(params.clone()),
            local_search: LocalSearch::new(),
            params,
            penalty_capacity: 100.0,
        }
    }

    pub fn run(&mut self) -> (Vec<Vec<usize>>, f64, f64) {
        let mut rng = StdRng::seed_from_u64(self.params.ap.seed);
        let pop_size = self.params.ap.mu;

        // 1. Initialize Population
        let init_pop_size = 4 * pop_size;

        for _ in 0..init_pop_size {
            // Check time limit during initialization too
            if self.params.ap.time_limit > 0.0
                && self.params.start_time.elapsed().as_secs_f64() > self.params.ap.time_limit
            {
                break;
            }

            let mut giant_tour: Vec<usize> = (1..=self.params.n_clients).collect();
            giant_tour.shuffle(&mut rng);

            let mut indiv = Individual::new(giant_tour);
            self.split.split(&mut indiv, self.penalty_capacity);
            indiv.evaluate(&self.params);

            self.local_search
                .run(&mut indiv, &self.params, self.penalty_capacity);

            // Repair if infeasible (chance: 50%)
            if !indiv.eval.is_feasible && rng.random_bool(0.5) {
                let repair_penalty = self.penalty_capacity * 10.0;
                self.local_search
                    .run(&mut indiv, &self.params, repair_penalty);
            }

            // Must re-evaluate with current penalty for correct population addition?
            // add() re-checks feasibility.
            // But penalized cost inside indiv depends on penalty passed?
            // Update: indiv.evaluate doesn't take penalty anymore.
            // We need to set penalized_cost externally or pass it?
            // Re-check Individual: compute_penalized_cost.
            indiv.compute_penalized_cost(
                self.penalty_capacity,
                self.params.c_coeff,
                self.params.r_coeff,
                indiv.eval.profit,
            );

            self.population.add(indiv);
        }

        self.population.update_biased_fitnesses(&self.params);
        self.population.prune(self.params.ap.mu);

        // 2. Evolution Loop
        let mut iter = 0;

        while iter < self.params.ap.nb_iter {
            // Time Limit Check using Params.start_time
            if self.params.ap.time_limit > 0.0
                && self.params.start_time.elapsed().as_secs_f64() > self.params.ap.time_limit
            {
                break;
            }

            iter += 1;

            // Selection
            let parents = self.population.get_random_pair();
            if parents.is_none() {
                break;
            }
            let (p1, p2) = parents.unwrap();

            // Crossover
            let child_tour = self.crossover_ox(&p1.giant_tour, &p2.giant_tour, &mut rng);
            let mut child = Individual::new(child_tour);

            // Split
            self.split.split(&mut child, self.penalty_capacity);

            // Local Search
            child.evaluate(&self.params);
            self.local_search
                .run(&mut child, &self.params, self.penalty_capacity);

            // Repair if infeasible
            if !child.eval.is_feasible && rng.random_bool(0.5) {
                let repair_penalty = self.penalty_capacity * 10.0;
                self.local_search
                    .run(&mut child, &self.params, repair_penalty);
            }

            child.compute_penalized_cost(
                self.penalty_capacity,
                self.params.c_coeff,
                self.params.r_coeff,
                child.eval.profit,
            );

            // Survivor Selection
            self.population.add(child);
            self.population.update_biased_fitnesses(&self.params);
            self.population.prune(self.params.ap.mu);

            // Penalty Management
            if iter % self.params.ap.nb_iter_penalty_management == 0 {
                self.adjust_penalties();
            }
        }

        // Return best
        if let Some(best) = &self.population.best_solution {
            let cost = best.eval.distance * self.params.c_coeff;
            (best.chrom_r.clone(), best.eval.profit, cost)
        } else {
            if !self.population.infeasible.is_empty() {
                let best_inf = &self.population.infeasible[0];
                let cost = best_inf.eval.distance * self.params.c_coeff;
                (best_inf.chrom_r.clone(), best_inf.eval.profit, cost)
            } else {
                (vec![], 0.0, 0.0)
            }
        }
    }

    // ... [adjust_penalties and crossover_ox remain unchanged] ...
    fn adjust_penalties(&mut self) {
        let n_feasible = self.population.feasible.len();
        let total = n_feasible + self.population.infeasible.len();
        let ratio = if total > 0 {
            n_feasible as f64 / total as f64
        } else {
            0.0
        };

        if ratio < self.params.ap.target_feasible - 0.05 {
            self.penalty_capacity *= self.params.ap.penalty_increase;
        } else if ratio > self.params.ap.target_feasible + 0.05 {
            self.penalty_capacity *= self.params.ap.penalty_decrease;
        }

        if self.penalty_capacity < 0.1 {
            self.penalty_capacity = 0.1;
        }
        if self.penalty_capacity > 10000.0 {
            self.penalty_capacity = 10000.0;
        }
    }

    fn crossover_ox(&self, p1: &[usize], p2: &[usize], rng: &mut StdRng) -> Vec<usize> {
        let n = p1.len();
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);
        let (start, end) = if i < j { (i, j) } else { (j, i) };

        let mut child = vec![0; n];
        let mut in_child = vec![false; self.params.n_clients + 1];

        for k in start..=end {
            let node = p1[k];
            child[k] = node;
            in_child[node] = true;
        }

        let mut p2_idx = (end + 1) % n;
        let mut child_idx = (end + 1) % n;

        for _ in 0..n {
            let node = p2[p2_idx];
            if !in_child[node] {
                child[child_idx] = node;
                child_idx = (child_idx + 1) % n;
            }
            p2_idx = (p2_idx + 1) % n;
        }
        child
    }
}
