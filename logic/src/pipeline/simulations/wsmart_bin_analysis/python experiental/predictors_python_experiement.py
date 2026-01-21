"""
Unsuded module. Experimental script on how to all the arima ftting and predictions in python.

To run predict using this module need to use setup Predictor() on training data and
then run predictor.predict() and predictor.update(current_data) alternately; to allow the object to update errorson curent data
"""

from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm


def fit_sarima(data, order, seasonal_order):
    """
    Fit a SARIMA model to the provided data.

    Args:
        data (pd.Series or np.ndarray): Time series data.
        order (tuple): The (p,d,q) order of the model.
        seasonal_order (tuple): The (P,D,Q,s) seasonal order of the model.

    Returns:
        tuple: (aic, params) where aic is the AIC score and params are the model parameters.
               Returns (np.inf, None) if fitting fails.
    """
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(low_memory=True)
        return model_fit.aic, model_fit.params
    except Exception:
        return np.inf, None  # If the model fails, return a high AIC score


def grid_search_sarima(data, param_grid):
    """
    Perform grid search to find best SARIMA hyperparameters.

    Args:
        data (pd.Series): Time series data.
        param_grid (list): List of parameter dictionaries.

    Returns:
        tuple: (best_order, best_seasonal_order, best_model_params)
    """
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None

    for params in param_grid:
        order = (params["p"], params["d"], params["q"])
        seasonal_order = (params["P"], params["D"], params["Q"], params["m"])

        aic, model = fit_sarima(data, order, seasonal_order)
        if aic < best_aic:
            best_aic = aic
            best_order = order
            best_seasonal_order = seasonal_order
            best_model = model

    return best_order, best_seasonal_order, best_model


def parallel_grid_search(df: pd.DataFrame, process_func: callable):
    """
    Run grid search in parallel across dataframe columns.

    Args:
        df (pd.DataFrame): Dataframe where each column is a time series.
        process_func (callable): Function to process each column item.

    Returns:
        list: List of results from process_func.
    """
    print("Fitting Arima Models")
    with Pool() as pool:
        results = list(tqdm(pool.imap_unordered(process_func, df.items()), total=len(df.columns)))
    return results


def serial_grid_search(df: pd.DataFrame, process_func: callable):
    """
    Run grid search serially across dataframe columns.

    Args:
        df (pd.DataFrame): Dataframe where each column is a time series.
        process_func (callable): Function to process each column item.

    Returns:
        list: List of results from process_func.
    """
    print("Fitting Arima Models")
    return list(tqdm(map(process_func, df.items()), total=len(df.columns)))


def build_predictor(runtype, df: pd.DataFrame, params: dict = None):
    """
    Build a Predictor object by fitting SARIMA models to the dataframe columns.

    Args:
        runtype (str): 'serial' or 'parallel' execution mode.
        df (pd.DataFrame): Input time series data.
        params (dict, optional): Dictionary defining hyperparameter search space.

    Returns:
        Predictor: Initialized Predictor object.

    Raises:
        AssertionError: If runtype is invalid or params are malformed.
    """
    KEYS = ["serial", "parallel"]

    assert runtype in KEYS, f"runtime must be one of {KEYS}"

    assert params is None or len(params["m"]) == 1, "Only supports trying one seasonal period"

    df = df.copy(deep=True)
    df.fillna(0, inplace=True)
    df.columns = list(range(len(df.columns)))

    if params is None:
        p = range(0, 4)
        d = range(0, 1)
        q = range(0, 4)
        P = range(0, 2)
        D = range(0, 1)
        Q = range(0, 2)
        m = [7]
    else:
        p = params["p"]
        d = params["d"]
        q = params["q"]
        P = params["P"]
        D = params["D"]
        Q = params["Q"]
        m = params["m"]

    param_grid = list(ParameterGrid({"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "m": m}))

    def process_column(column_data):
        """Internal worker function to process a single column."""
        data = column_data[1]  # Extract the series data
        column_name = column_data[0]
        best_order, best_seasonal_order, best_model = grid_search_sarima(data, param_grid)
        return {
            "column": column_name,
            "order": best_order,
            "s_order": best_seasonal_order,
            "params": best_model,
        }

    if "serial":
        models = serial_grid_search(df, process_column)
    elif "parallel":
        models = parallel_grid_search(df, process_column)
    else:
        raise f"runtime must be one of {KEYS}"

    return Predictor(df.to_numpy(), maxorder=max(max(P), max(Q)) * m[0], models=models)


class Predictor:
    """
    Class to handle time series predictions using pre-fitted SARIMA parameters.
    """

    def __init__(
        self,
        data: np.ndarray,
        maxorder: int,
        models: list,
        means_arr: np.ndarray = None,
    ):
        """
        Initialize the Predictor.

        Args:
            data (np.ndarray): Historical data.
            maxorder (int): Maximum order of the AR/MA components.
            models (list): List of model dictionaries containing parameters.
            means_arr (np.ndarray, optional): Array of mean values if needed.
        """
        self.history = None
        self.errors = None
        self.ar = None
        self.ma = None
        self.error_var = None
        self.cur_prediction = None

        self.n_bins = data.shape[1]
        self.maxorder = maxorder

        self.__pre_process_params(models)

        self.errors = np.zeros(self.n_bins, maxorder)
        self.history = data[:, -2 * self.maxorder : self.maxorder]

        self.cur_prediction = means_arr

        self.__setup_errors()

    def update(self, cur_data) -> np.ndarray:
        """
        Update the predictor with new observation data and compute errors.

        Args:
            cur_data (np.ndarray): New data points.

        Returns:
            np.ndarray: Updated errors.
        """
        self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1] = cur_data

        self.errors[:, :-1] = self.errors[:, 1:]
        self.errors[:, -1] = np.abs(cur_data - self.cur_prediction)

        return self.errors[:-1]

    def predict(self) -> np.ndarray:
        """
        Predict the next step.

        Returns:
        -------
        pred: np.ndarray
            The next step prediction given current data.
        """

        self.cur_prediction = np.einsum("ij,ij->i", self.history, self.ar) + np.einsum("ij,ij->i", self.errors, self.ma)

        return self.cur_prediction

    def get_normal_var(self) -> Optional[np.ndarray]:
        """
        Get the variance of the prediction errors.

        Returns
        -------
        var: np.ndarray
            Return the variance of the error distribution around the next step prediction.
        """

        return self.error_var

    def __setup_errors(self, data=None):
        """
        Initialize errors based on initial history.

        Args:
            data: Optional data override.
        """
        # Note: Logic implies this is called during init where history is set.
        # Original code loop implementation is vague without data input in signature or usage.
        # Assuming usage context or missing implementation details in original.
        pass

    def __pre_process_params(self, models: list[dict]):
        """
        Process list of model parameters into arrays for vectorized prediction.

        Args:
            models (list[dict]): List of model parameter dictionaries.
        """
        self.ma = np.zeros(np.zeros(self.n_bins, self.maxorder))
        self.ar = np.zeros(np.zeros(self.n_bins, self.maxorder))
        self.mean_error = np.zeros(self.n_bins)

        for model in models:
            idx = model["column"]
            counter = 0

            for p in range(model["order"][0]):
                self.ar[idx, p] = model["params"][counter]
                counter += 1

            for q in range(model["order"][2]):
                self.ma[idx, q] = model["params"][counter]
                counter += 1

            for P in range(model["s_order"][0]):
                self.ar[idx, (P + 1) * model["s_order"][3] - 1] = model["params"][counter]
                counter += 1

            for Q in range(model["s_order"][2]):
                self.ma[idx, (Q + 1) * model["s_order"][3] - 1] = model["params"][counter]
                counter += 1

            self.error_var[idx] = model["params"][-1]
