/*!
 * # Rust Policy Module - Hybrid Genetic Search for VRPP
 *
 * This Rust module implements a high-performance **Hybrid Genetic Search (HGS)** algorithm
 * for solving the **Vehicle Routing Problem with Profits (VRPP)**. It is compiled as a
 * Python extension using PyO3, providing the WSmart-Route framework with fast baseline solvers.
 *
 * ## Architecture
 *
 * The HGS algorithm combines:
 * - **Genetic Algorithm** - Population-based evolutionary search
 * - **Local Search** - Neighborhood-based improvement operators
 * - **Split Algorithm** - Giant tour decomposition for route construction
 * - **Geometric Optimization** - Angular sector pruning for efficiency
 *
 * ## Objective Function
 *
 * Maximizes **Profit = Revenue - Cost - Penalties**
 * - **Revenue**: `R × Σ demands_served`
 * - **Cost**: `C × total_distance`
 * - **Penalties**: Soft constraint violations (capacity excess)
 *
 * ## Python Interface
 *
 * ```python
 * from policy import solve, solve_batch
 *
 * # Single instance
 * routes, profit, cost = solve(
 *     dist_matrix=[[0, 10], [10, 0]],
 *     demands=[0, 5],
 *     coords=[(0, 0), (1, 1)],
 *     capacity=100.0,
 *     r=1.0,
 *     c=1.0,
 *     time_limit=10.0,
 *     pop_size=25,
 *     seed=42,
 *     max_vehicles=3
 * )
 *
 * # Batch processing (parallel)
 * routes_batch, profits, costs = solve_batch(
 *     dist_matrices=[...],
 *     demands_batch=[...],
 *     coords_batch=[...],
 *     capacity=100.0,
 *     r=1.0,
 *     c=1.0,
 *     time_limit=10.0,
 *     nb_granular=20,
 *     seed=42,
 *     max_vehicles=3
 * )
 * ```
 */

mod algorithm_parameters;
mod circle_sector;
mod genetic;
mod individual;
mod local_search;
mod params;
mod population;
mod split;

use crate::algorithm_parameters::AlgorithmParameters;
use crate::genetic::Genetic;
use crate::params::Params;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

/**
 * Solves a single VRPP instance using the Hybrid Genetic Search algorithm.
 *
 * # Arguments
 *
 * * `dist_matrix` - Symmetric distance matrix (N+1 × N+1, index 0 is depot)
 * * `demands` - Node demands (depot = 0)
 * * `coords` - Node coordinates for geometric optimization
 * * `capacity` - Vehicle capacity
 * * `r` - Revenue coefficient
 * * `c` - Cost coefficient per unit distance
 * * `time_limit` - Maximum runtime in seconds
 * * `nb_granular` - Neighborhood size for granular search
 * * `seed` - Random seed for reproducibility
 * * `max_vehicles` - Fleet size limit (0 = unlimited)
 *
 * # Returns
 *
 * A tuple containing:
 * - `routes`: List of routes (each route is a list of node indices)
 * - `profit`: Total profit (revenue - cost)
 * - `cost`: Total transportation cost
 */
fn solve_instance(
    dist_matrix: Vec<Vec<f64>>,
    demands: Vec<f64>,
    coords: Vec<(f64, f64)>,
    capacity: f64,
    r: f64,
    c: f64,
    time_limit: f64,
    nb_granular: usize,
    seed: u64,
    max_vehicles: usize,
) -> (Vec<Vec<usize>>, f64, f64) {
    let mut ap = AlgorithmParameters::default();
    ap.time_limit = time_limit;
    ap.nb_granular = nb_granular;
    ap.seed = seed;
    ap.max_vehicles = max_vehicles;
    ap.nb_iter = 20000;

    let params = Arc::new(Params::new(
        dist_matrix,
        demands,
        coords,
        capacity,
        r,
        c,
        ap,
        max_vehicles,
    ));

    let mut ga = Genetic::new(params);
    ga.run()
}

/**
 * Solves a single VRPP instance (Python API).
 *
 * # Arguments
 *
 * * `dist_matrix` - Symmetric distance matrix (N+1 × N+1, index 0 is depot)
 * * `demands` - Node demands (depot = 0)
 * * `coords` - Node coordinates for geometric optimization
 * * `capacity` - Vehicle capacity
 * * `r` - Revenue coefficient (internally boosted by 100× to ensure profitability)
 * * `c` - Cost coefficient per unit distance
 * * `time_limit` - Maximum runtime in seconds
 * * `_pop_size` - Unused (kept for API compatibility)
 * * `seed` - Optional random seed (default: 0)
 * * `max_vehicles` - Optional fleet size limit (default: 0 = unlimited)
 *
 * # Returns
 *
 * A PyResult containing:
 * - `routes`: List of routes
 * - `profit`: Total profit
 * - `cost`: Total cost
 *
 * # Example
 *
 * ```python
 * routes, profit, cost = solve(
 *     dist_matrix=[[0, 10], [10, 0]],
 *     demands=[0, 5],
 *     coords=[(0, 0), (1, 1)],
 *     capacity=100.0,
 *     r=1.0,
 *     c=1.0,
 *     time_limit=10.0,
 *     pop_size=25,
 *     seed=42,
 *     max_vehicles=3
 * )
 * ```
 */
#[pyfunction]
fn solve(
    dist_matrix: Vec<Vec<f64>>,
    demands: Vec<f64>,
    coords: Vec<(f64, f64)>,
    capacity: f64,
    r: f64,
    c: f64,
    time_limit: f64,
    _pop_size: usize,
    seed: Option<u64>,
    max_vehicles: Option<usize>,
) -> PyResult<(Vec<Vec<usize>>, f64, f64)> {
    let actual_seed = seed.unwrap_or(0);
    let actual_max_vehicles = max_vehicles.unwrap_or(0);
    let (routes, profit, cost) = solve_instance(
        dist_matrix,
        demands,
        coords,
        capacity,
        r * 100.0, // Aggressively boost revenue to ensure serving clients is profitable (avoid empty routes)
        c, // Revert cost to original to maintain balance relative to penalty (but dominated by R for service)
        time_limit,
        20,
        actual_seed,
        actual_max_vehicles,
    );
    Ok((routes, profit, cost))
}

/**
 * Solves multiple VRPP instances in parallel using Rayon (Python API).
 *
 * Each instance is solved independently with a unique seed (`seed + i`).
 * This function is optimized for batch processing of similar instances.
 *
 * # Arguments
 *
 * * `dist_matrices` - List of distance matrices
 * * `demands_batch` - List of demand arrays
 * * `coords_batch` - List of coordinate arrays (may be empty)
 * * `capacity` - Vehicle capacity (shared across all instances)
 * * `r` - Revenue coefficient
 * * `c` - Cost coefficient
 * * `time_limit` - Time limit per instance
 * * `nb_granular` - Granular search neighborhood size
 * * `seed` - Base random seed (incremented for each instance)
 * * `max_vehicles` - Fleet size limit
 *
 * # Returns
 *
 * A PyResult containing:
 * - `routes_batch`: List of route solutions
 * - `profits_batch`: List of profits
 * - `costs_batch`: List of costs
 *
 * # Errors
 *
 * Returns `PyValueError` if batch sizes don't match.
 */
#[pyfunction]
fn solve_batch(
    dist_matrices: Vec<Vec<Vec<f64>>>,
    demands_batch: Vec<Vec<f64>>,
    coords_batch: Vec<Vec<(f64, f64)>>,
    capacity: f64,
    r: f64,
    c: f64,
    time_limit: f64,
    nb_granular: usize,
    seed: Option<u64>,
    max_vehicles: Option<usize>,
) -> PyResult<(Vec<Vec<Vec<usize>>>, Vec<f64>, Vec<f64>)> {
    let batch_size = dist_matrices.len();
    if demands_batch.len() != batch_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Batch size mismatch between dists and demands",
        ));
    }
    // Handle coords batch if present or generate empty
    let use_coords = !coords_batch.is_empty();
    if use_coords && coords_batch.len() != batch_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Batch size mismatch between dists and coords",
        ));
    }

    let actual_seed = seed.unwrap_or(0);
    let actual_max_vehicles = max_vehicles.unwrap_or(0);

    let results: Vec<(Vec<Vec<usize>>, f64, f64)> = dist_matrices
        .into_par_iter()
        .zip(demands_batch.into_par_iter())
        .enumerate()
        .map(|(i, (dist, dem))| {
            let coords = if use_coords {
                coords_batch[i].clone()
            } else {
                vec![]
            };
            solve_instance(
                dist,
                dem,
                coords,
                capacity,
                r,
                c,
                time_limit,
                nb_granular,
                actual_seed + i as u64,
                actual_max_vehicles,
            )
        })
        .collect();

    // Unzip results
    let mut routes_batch = Vec::with_capacity(batch_size);
    let mut profits_batch = Vec::with_capacity(batch_size);
    let mut costs_batch = Vec::with_capacity(batch_size);

    for (routes, profit, cost) in results {
        routes_batch.push(routes);
        profits_batch.push(profit);
        costs_batch.push(cost);
    }

    Ok((routes_batch, profits_batch, costs_batch))
}

/**
 * Python module definition.
 *
 * Exposes two functions to Python:
 * - `solve`: Single instance solver
 * - `solve_batch`: Parallel batch solver
 */
#[pymodule]
fn policy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch, m)?)?;
    Ok(())
}
