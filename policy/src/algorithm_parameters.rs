/**
 * Algorithm hyperparameters for Hybrid Genetic Search.
 *
 * Controls the behavior of the genetic algorithm, including population size,
 * diversity management, penalty adjustment, and termination criteria.
 *
 * # Default Values
 *
 * ```
 * AlgorithmParameters {
 *     nb_granular: 20,         // Granular search neighborhood size
 *     mu: 25,                  // Population size
 *     lambda: 40,              // Offspring population size
 *     nb_elite: 4,             // Number of elite solutions
 *     nb_close: 5,             // Neighbors for diversity calculation
 *     nb_iter_penalty_management: 100,  // Penalty adjustment frequency
 *     target_feasible: 0.2,    // Target ratio of feasible solutions
 *     penalty_decrease: 0.85,  // Penalty decrease multiplier
 *     penalty_increase: 1.2,   // Penalty increase multiplier
 *     seed: 0,                 // Random seed
 *     nb_iter: 20000,          // Maximum iterations
 *     time_limit: 0.0,         // Time limit in seconds (0 = no limit)
 *     use_swap_star: true,     // Enable SWAP* operator
 *     max_vehicles: 0,         // Fleet size limit (0 = unlimited)
 * }
 * ```
 */
#[derive(Clone)]
pub struct AlgorithmParameters {
    pub nb_granular: usize,
    pub mu: usize,
    pub lambda: usize,
    pub nb_elite: usize,
    pub nb_close: usize,
    pub nb_iter_penalty_management: usize,
    pub target_feasible: f64,
    pub penalty_decrease: f64,
    pub penalty_increase: f64,
    pub seed: u64,
    pub nb_iter: usize,
    pub time_limit: f64,
    pub use_swap_star: bool,
    pub max_vehicles: usize,
}

impl Default for AlgorithmParameters {
    fn default() -> Self {
        Self {
            nb_granular: 20,
            mu: 25,
            lambda: 40,
            nb_elite: 4,
            nb_close: 5,
            nb_iter_penalty_management: 100,
            target_feasible: 0.2,
            penalty_decrease: 0.85,
            penalty_increase: 1.2,
            seed: 0,
            nb_iter: 20000,
            time_limit: 0.0,
            use_swap_star: true,
            max_vehicles: 0,
        }
    }
}
