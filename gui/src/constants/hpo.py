"""
Hyperparameter optimization constants for the GUI.
"""

# Hyper-Parameter Optimization
HPO_METRICS = {
    "Training Loss": "loss",
    "Validation Loss": "val_loss",
    "Training Mean Reward": "mean_reward",
    "Mean Absolute Error (MAE)": "mae",
    "Mean Squared Error (MSE)": "mse",
    "Root MSE (RMSE)": "rmse",
    "Episode Mean Reward": "episode_reward_mean",
    "Efficiency (Kilogram per Kilometer)": "kg/km",
    "Number of Overflows": "overflows",
    "Efficiency and Overflows": "both",
}

HPO_METHODS = {
    "": None,
    "Distributed Evolutionary Algorithm (DEA)": "dea",
    "Bayesian Optimization (BO)": "bo",
    "Hyperband Optimization (HBO)": "hbo",
    "Random Search (RS)": "rs",
    "Grid Search (GS)": "gs",
    "Differential Evolutionary Hyperband Optimization (DEHBO)": "dehbo",
}
