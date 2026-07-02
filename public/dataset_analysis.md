# WSmart+ Route — Dataset Analysis Report

> **Scope:** NPZ simulator datasets and TensorDict training datasets
> **Cities:** Figueira da Foz, Rio Maior
> **Distributions:** Empirical, Gamma-3
> **Horizons analysed:** 30 days, 90 days
> **Total NPZ dataset entries:** 20
> **Generated:** <!-- date -->

---

## Table of Contents

1. [Training Data (TD)](#1-training-data-td)
2. [Figueira da Foz NPZ Datasets](#2-figueira-da-foz-npz-datasets)
3. [Rio Maior NPZ Datasets](#3-rio-maior-npz-datasets)
4. [City Comparison](#4-city-comparison)
5. [TD vs NPZ Alignment](#5-td-vs-npz-alignment)

---


## 1. Training Data (TD)

Training data used for supervised learning models (stored as TensorDict `.td` files).
Each entry contains normalised waste values in [0, 1] (divide by 100 to convert to kg/kg).

<figure style="display:block;width:100%;margin:0.8em 0;padding:0;"><img src="figures/datasets/td_stats_comparison.png" alt="Waste Statistics Comparison" width="100%" style="width:100% !important;max-width:100% !important;height:auto !important;display:block !important;margin:0;" /></figure>

**Figure 1:** *Mean, std, and skewness of training waste values per network size and distribution.*

<figure style="display:block;width:100%;margin:0.8em 0;padding:0;"><img src="figures/datasets/td_waste_distributions.png" alt="Training Data Waste Distributions" width="100%" style="width:100% !important;max-width:100% !important;height:auto !important;display:block !important;margin:0;" /></figure>

**Figure 2:** *Bar chart of mean and std waste fractions per network size.*

### TD Statistics Summary

**Table 1:** *Training data (TD) statistics — mean, std, and skewness of normalised waste values per network size and distribution.*

| N | Distribution | Instances | Mean Waste | Std Waste | Skewness |
|---|-------------|-----------|------------|-----------|---------|
| 20 | Empirical | 12,800 | 0.0476 | 0.1110 | 2.971 |
| 20 | Gamma-3 | 12,800 | 0.1380 | 0.1207 | 1.457 |
| 50 | Empirical | 12,800 | 0.0459 | 0.1011 | 2.607 |
| 50 | Gamma-3 | 12,800 | 0.1380 | 0.1207 | 1.453 |
| 100 | Empirical | 12,800 | 0.0463 | 0.1113 | 2.798 |
| 100 | Gamma-3 | 12,800 | 0.1379 | 0.1206 | 1.453 |
| 170 | Empirical | 12,800 | 0.0510 | 0.1117 | 2.678 |
| 170 | Gamma-3 | 12,800 | 0.1380 | 0.1207 | 1.452 |

<!-- [ANALYSIS: Insert your observations here] -->

---

## 2. Figueira da Foz NPZ Datasets
**Network sizes:** N = 350  **Distributions:** Empirical, Gamma-3  **Horizons:** 30 days, 90 days
<figure style="display:block;width:100%;margin:0.8em 0;padding:0;"><img src="figures/datasets/npz_stats_bar.png" alt="NPZ Statistics Bar Chart" width="100%" style="width:100% !important;max-width:100% !important;height:auto !important;display:block !important;margin:0;" /></figure>

**Figure 3:** *Mean, std, max waste and overflow percentage per city and distribution.*
<figure style="display:block;width:100%;margin:0.8em 0;padding:0;"><img src="figures/datasets/npz_size_scaling.png" alt="Statistics vs Network Size" width="100%" style="width:100% !important;max-width:100% !important;height:auto !important;display:block !important;margin:0;" /></figure>

**Figure 4:** *How mean waste, std, and skewness vary with network size.*
<figure style="display:block;width:100%;margin:0.8em 0;padding:0;"><img src="figures/datasets/npz_horizon_comparison.png" alt="30-day vs 90-day Horizon Comparison" width="100%" style="width:100% !important;max-width:100% !important;height:auto !important;display:block !important;margin:0;" /></figure>

**Figure 5:** *Comparison of 30-day and 90-day horizon statistics.*

### Statistics Summary — Figueira da Foz (30-day horizon)
**Table 2:** *NPZ dataset statistics for Figueira da Foz — mean, std, max waste and overflow percentage per network size and distribution (30-day horizon).*
| City | N | Distribution | Mean kg | Std kg | Max kg | Overflow % | Skewness |
|------|---|-------------|---------|--------|--------|------------|---------|
| Figueira da Foz | 350 | Empirical | 7.15 | 10.06 | 61.0 | 0.000 | 1.366 |
| Figueira da Foz | 350 | Gamma-3 | 13.88 | 12.25 | 100.0 | 0.000 | 1.548 |
<!-- [ANALYSIS: Insert your observations here] -->
## 3. Rio Maior NPZ Datasets
**Network sizes:** N = 20, 50, 100, 170  **Distributions:** Empirical, Gamma-3  **Horizons:** 30 days, 90 days
<figure style="display:block;width:100%;margin:0.8em 0;padding:0;"><img src="figures/datasets/npz_stats_bar.png" alt="NPZ Statistics Bar Chart" width="100%" style="width:100% !important;max-width:100% !important;height:auto !important;display:block !important;margin:0;" /></figure>

**Figure 6:** *Mean, std, max waste and overflow percentage per city and distribution.*
<figure style="display:block;width:100%;margin:0.8em 0;padding:0;"><img src="figures/datasets/npz_size_scaling.png" alt="Statistics vs Network Size" width="100%" style="width:100% !important;max-width:100% !important;height:auto !important;display:block !important;margin:0;" /></figure>

**Figure 7:** *How mean waste, std, and skewness vary with network size.*
<figure style="display:block;width:100%;margin:0.8em 0;padding:0;"><img src="figures/datasets/npz_horizon_comparison.png" alt="30-day vs 90-day Horizon Comparison" width="100%" style="width:100% !important;max-width:100% !important;height:auto !important;display:block !important;margin:0;" /></figure>

**Figure 8:** *Comparison of 30-day and 90-day horizon statistics.*

### Statistics Summary — Rio Maior (30-day horizon)
**Table 3:** *NPZ dataset statistics for Rio Maior — mean, std, max waste and overflow percentage per network size and distribution (30-day horizon).*
| City | N | Distribution | Mean kg | Std kg | Max kg | Overflow % | Skewness |
|------|---|-------------|---------|--------|--------|------------|---------|
| Rio Maior | 20 | Empirical | 5.27 | 11.60 | 100.0 | 0.000 | 3.219 |
| Rio Maior | 20 | Gamma-3 | 13.47 | 11.92 | 83.3 | 0.000 | 1.741 |
| Rio Maior | 50 | Empirical | 5.46 | 10.71 | 61.0 | 0.000 | 2.267 |
| Rio Maior | 50 | Gamma-3 | 13.36 | 11.61 | 83.3 | 0.000 | 1.485 |
| Rio Maior | 100 | Empirical | 5.54 | 12.03 | 93.0 | 0.000 | 2.656 |
| Rio Maior | 100 | Gamma-3 | 13.67 | 12.10 | 100.0 | 0.000 | 1.690 |
| Rio Maior | 170 | Empirical | 5.80 | 11.47 | 100.0 | 0.000 | 2.528 |
| Rio Maior | 170 | Gamma-3 | 13.75 | 12.12 | 100.0 | 0.000 | 1.540 |
<!-- [ANALYSIS: Insert your observations here] -->



## 4. City Comparison

<figure style="display:block;width:100%;margin:0.8em 0;padding:0;"><img src="figures/datasets/npz_city_comparison.png" alt="City Comparison Overview" width="100%" style="width:100% !important;max-width:100% !important;height:auto !important;display:block !important;margin:0;" /></figure>

**Figure 9:** *Key statistics across cities and distributions.*

### Statistics Summary — All Cities (30-day horizon)

**Table 4:** *NPZ dataset statistics across all cities — mean, std, max waste and overflow percentage per city and distribution (30-day horizon).*

| City | N | Distribution | Mean kg | Std kg | Max kg | Overflow % | Skewness |
|------|---|-------------|---------|--------|--------|------------|---------|
| Figueira da Foz | 350 | Empirical | 7.15 | 10.06 | 61.0 | 0.000 | 1.366 |
| Figueira da Foz | 350 | Gamma-3 | 13.88 | 12.25 | 100.0 | 0.000 | 1.548 |
| Rio Maior | 20 | Empirical | 5.27 | 11.60 | 100.0 | 0.000 | 3.219 |
| Rio Maior | 20 | Gamma-3 | 13.47 | 11.92 | 83.3 | 0.000 | 1.741 |
| Rio Maior | 50 | Empirical | 5.46 | 10.71 | 61.0 | 0.000 | 2.267 |
| Rio Maior | 50 | Gamma-3 | 13.36 | 11.61 | 83.3 | 0.000 | 1.485 |
| Rio Maior | 100 | Empirical | 5.54 | 12.03 | 93.0 | 0.000 | 2.656 |
| Rio Maior | 100 | Gamma-3 | 13.67 | 12.10 | 100.0 | 0.000 | 1.690 |
| Rio Maior | 170 | Empirical | 5.80 | 11.47 | 100.0 | 0.000 | 2.528 |
| Rio Maior | 170 | Gamma-3 | 13.75 | 12.12 | 100.0 | 0.000 | 1.540 |

<!-- [ANALYSIS: Insert your observations here] -->

---


## 5. TD vs NPZ Alignment

<figure style="display:block;width:100%;margin:0.8em 0;padding:0;"><img src="figures/datasets/npz_td_alignment.png" alt="Training (TD) vs Simulator (NPZ) Mean Waste Alignment" width="100%" style="width:100% !important;max-width:100% !important;height:auto !important;display:block !important;margin:0;" /></figure>

*Comparison of mean waste levels between TD training data (normalised × 100) and NPZ simulator
data. Close alignment validates that training distribution matches simulation.*

<!-- [ANALYSIS: Insert your observations here] -->

---

*Figures are stored in `figures/datasets/`.*
*Raw statistics: `public/global/datasets/td_stats.csv` and `public/global/datasets/npz_stats.csv`.*

## Interactive Charts

- [NPZ Statistics — Mean vs Std Scatter](private/datasets/npz_stats_interactive.html)
- [Waste Distribution by City and Network Size](private/datasets/waste_distribution_interactive.html)
- [City & Network Comparison](private/datasets/city_network_comparison_interactive.html)
    