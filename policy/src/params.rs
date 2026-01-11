/*!
 * Problem instance parameters and precomputed optimization structures.
 *
 * This module defines the `Params` struct, which encapsulates all problem instance data
 * and precomputed structures needed for efficient optimization.
 */
use crate::algorithm_parameters::AlgorithmParameters;
use std::time::Instant;

/**
 * Problem instance parameters.
 *
 * This struct stores all problem data and precomputed optimization structures.
 * It is shared across algorithm components using `Arc<Params>` for thread-safe access.
 *
 * # Fields
 *
 * * `dist_matrix` - Symmetric distance matrix (N+1 × N+1, depot at index 0)
 * * `demands` - Node demands (depot = 0)
 * * `coords` - Node coordinates for geometric optimization
 * * `polar_angles` - Precomputed polar angles (0-65535) relative to depot
 * * `vehicle_capacity` - Maximum load capacity per vehicle
 * * `r_coeff` - Revenue coefficient (profit per unit of demand served)
 * * `c_coeff` - Cost coefficient (cost per unit of distance)
 * * `n_clients` - Number of customers (excludes depot)
 * * `max_vehicles` - Fleet size limit (0 = unlimited)
 * * `correlated_vertices` - Granular search neighborhoods (top-k nearest neighbors)
 * * `ap` - Algorithm hyperparameters
 * * `start_time` - Algorithm start timestamp for time limit enforcement
 */
pub struct Params {
    pub ap: AlgorithmParameters,
    pub dist_matrix: Vec<Vec<f64>>,
    pub demands: Vec<f64>,       // 0-based, index 0 is depot
    pub coords: Vec<(f64, f64)>, // Coordinates for geometric optimization
    pub polar_angles: Vec<i32>,  // 0-65535 angle for geometric optimization
    pub vehicle_capacity: f64,
    pub r_coeff: f64,                         // Revenue per unit of demand
    pub c_coeff: f64,                         // Cost per unit of distance
    pub n_clients: usize,                     // Excludes depot
    pub max_vehicles: usize,                  // Limited Fleet size (0 = unlimited)
    pub correlated_vertices: Vec<Vec<usize>>, // Granular search neighborhoods
    pub start_time: Instant,
}

impl Params {
    /**
     * Creates a new problem instance with precomputed optimization structures.
     *
     * # Arguments
     *
     * * `dist_matrix` - Distance matrix (must be symmetric)
     * * `demands` - Node demands
     * * `coords` - Node coordinates (may be empty)
     * * `vehicle_capacity` - Vehicle capacity
     * * `r` - Revenue coefficient
     * * `c` - Cost coefficient
     * * `ap` - Algorithm parameters
     * * `max_vehicles` - Fleet size limit
     *
     * # Initialization
     *
     * 1. Computes problem size (`n_clients`)
     * 2. Builds **granular search neighborhoods** (top-k closest neighbors for each node)
     * 3. Computes **polar angles** for geometric sector optimization
     * 4. Validates geometric consistency (if coordinates provided)
     *
     * # Granular Search
     *
     * For each customer, stores the indices of the `nb_granular` closest neighbors.
     * This reduces local search complexity from O(N²) to O(N×k).
     *
     * # Polar Angles
     *
     * Maps each node's angle (relative to depot) to 0-65535 range.
     * Used in SWAP* operator for geometric pruning.
     */
    pub fn new(
        dist_matrix: Vec<Vec<f64>>,
        demands: Vec<f64>,
        coords: Vec<(f64, f64)>, // New: Coordinates
        vehicle_capacity: f64,
        r: f64,
        c: f64,
        ap: AlgorithmParameters,
        max_vehicles: usize, // New: Max vehicles
    ) -> Self {
        let n_clients = dist_matrix.len() - 1;
        let mut correlated_vertices = vec![vec![]; n_clients + 1];

        // Precompute Granular Search neighborhoods (Correlated Vertices)
        // For each client, find the closest neighbors
        for i in 1..=n_clients {
            let mut neighbors: Vec<(usize, f64)> = (1..=n_clients)
                .filter(|&j| i != j)
                .map(|j| (j, dist_matrix[i][j]))
                .collect();

            // Sort by distance (ascending)
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Keep top nb_granular
            let count = std::cmp::min(ap.nb_granular, neighbors.len());
            correlated_vertices[i] = neighbors.iter().take(count).map(|x| x.0).collect();
        }

        // Calculate polar angles
        // Depot is at coords[0]
        let mut polar_angles = vec![0; n_clients + 1];
        if !coords.is_empty() {
            let depot_x = coords[0].0;
            let depot_y = coords[0].1;
            for i in 1..=n_clients {
                let dx = coords[i].0 - depot_x;
                let dy = coords[i].1 - depot_y;
                let angle = dy.atan2(dx); // -pi to pi
                                          // Map to 0-65535
                let normalized = (angle / std::f64::consts::PI / 2.0 + 0.5) * 65536.0;
                polar_angles[i] = normalized as i32 % 65536;
            }
        }

        let params = Self {
            ap,
            dist_matrix,
            demands,
            coords,
            polar_angles,
            vehicle_capacity,
            r_coeff: r,
            c_coeff: c,
            n_clients,
            max_vehicles,
            correlated_vertices,
            start_time: Instant::now(),
        };

        params.check_geometric_validity();
        params
    }

    /**
     * Validates that the distance matrix is consistent with Euclidean distances.
     *
     * Warns if `dist_matrix[i][j] < euclidean_distance(i, j)`, which indicates
     * a geometric inconsistency (e.g., road network distances vs. straight-line).
     */
    fn check_geometric_validity(&self) {
        if !self.coords.is_empty() {
            for i in 0..=self.n_clients {
                for j in 0..=self.n_clients {
                    if i == j {
                        continue;
                    }
                    let dx = self.coords[i].0 - self.coords[j].0;
                    let dy = self.coords[i].1 - self.coords[j].1;
                    let euclidean = (dx * dx + dy * dy).sqrt();
                    // A small tolerance for floating point errors
                    if self.dist_matrix[i][j] < euclidean - 1e-3 {
                        eprintln!("Warning: Geometric consistency violation at {}->{}: dist_matrix={} < euclidean={}", 
                            i, j, self.dist_matrix[i][j], euclidean);
                    }
                }
            }
        }
    }
}
