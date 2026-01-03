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

/// Single Instance Solver
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

#[pymodule]
fn policy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch, m)?)?;
    Ok(())
}
