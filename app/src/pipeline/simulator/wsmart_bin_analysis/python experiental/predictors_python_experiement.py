"""
Unsuded module. Experimental script on how to all the arima ftting and predictions in python.

To run predict using this module need to use setup Predictor() on training data and 
then run predictor.predict() and predictor.update(current_data) alternately; to allow the object to update errorson curent data 
"""



import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from multiprocessing import Pool


def fit_sarima(data, order, seasonal_order):
    try:
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(low_memory=True)
        return model_fit.aic, model_fit.params
    except:
        return np.inf, None # If the model fails, return a high AIC score


def grid_search_sarima(data, param_grid):
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_model = None

    for params in param_grid:
        order = (params['p'], params['d'], params['q'])
        seasonal_order = (params['P'], params['D'], params['Q'], params['m'])
        
        aic, model = fit_sarima(data, order, seasonal_order)
        if aic < best_aic:
            best_aic = aic
            best_order = order
            best_seasonal_order = seasonal_order
            best_model = model
    
    return best_order, best_seasonal_order, best_model

def parallel_grid_search(df:pd.DataFrame, process_func:callable):

    print("Fitting Arima Models")
    with Pool() as pool:
        results = list(tqdm(pool.imap_unordered(process_func, df.items()), total=len(df.columns)))
    return results

def serial_grid_search(df:pd.DataFrame, process_func:callable):

    print("Fitting Arima Models")
    return list(tqdm(map(process_func, df.items()), total=len(df.columns)))
    

def build_predictor(runtype, df: pd.DataFrame, params:dict = None):

    KEYS = ['serial', 'parallel']

    assert runtype in KEYS, f"runtime must be one of {KEYS}"

    assert params==None or len(params['m']) == 1 , f"Only supports trying one seasonal period"

    df = df.copy(deep=True)
    df.fillna(0, inplace=True)
    df.columns = list(range(len(df.columns)))

    if params == None:
        p = range(0, 4); d = range(0, 1); q = range(0, 4)
        P = range(0, 2); D = range(0, 1); Q = range(0, 2)
        m = [7]
    else:
        p = params['p']; d = params['d']; q = params['q']
        P = params['P']; d = params['D']; q = params['Q']
        m = params['m']

    param_grid = list(ParameterGrid({
        'p': p, 'd': d, 'q': q,
        'P': P, 'D': D, 'Q': Q,
        'm': m
    }))

    def process_column(column_data):
        data = column_data[1]  # Extract the series data
        column_name = column_data[0]
        best_order, best_seasonal_order , best_model = grid_search_sarima(data, param_grid)
        return {
            "column": column_name,
            "order": best_order,
            "s_order": best_seasonal_order,
            "params": best_model
        }

    if 'serial':
        models = serial_grid_search(df, process_column)
    elif 'parallel':
        models = parallel_grid_search(df, process_column)
    else:
        raise f"runtime must be one of {KEYS}"

    return Predictor(df.to_numpy(), maxorder=max(max(P),max(Q))*m[0], models = models)


class Predictor():

    def __init__(self, data:np.ndarray, maxorder:int, models:list, means_arr:np.ndarray):
        
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
        self.history = data[:,-2*self.maxorder:self.maxorder]

        self.cur_prediction = means_arr
    
        self.__setup_errors()
    
    def update(self, cur_data) -> np.ndarray:
        self.history[:,:-1] = self.history[:,1:]
        self.history[:,-1] = cur_data

        self.errors[:,:-1] = self.errors[:,1:]
        self.errors[:,-1] = np.abs(cur_data - self.cur_prediction)

        return self.errors[:-1]

    def predict(self) -> np.ndarray:

        """
        Retruns
        -------
        pred: np.ndarray
            The next stp prediction given current data
        """
            
        self.cur_prediction =  np.einsum('ij,ij->i', self.history, self.ar) + np.einsum('ij,ij->i', self.errors, self.ma)

        return self.cur_prediction
    
    def get_normal_var(self) -> np.ndarray:

        """
        Returns
        -------
        var: np.ndarray
            Return the variance of the error dsitribution around the next step prediction
        """

        return self.error_var
    
    def __setup_errors(self, data):

        for i in range(self.maxorder):
            _ = self.update(data[:, -self.maxorder+i])
            self.cur_prediction = self.predict()
    
    def __pre_process_params(self, models:list[dict]):

        self.ma = np.zeros(np.zeros(self.n_bins, self.maxorder))
        self.ar = np.zeros(np.zeros(self.n_bins, self.maxorder))
        self.mean_error = np.zeros(self.n_bins)

        for model in models:
            idx = model["column"]
            counter = 0

            for p in range(model["order"][0]):
                self.ar[idx,p] = model["params"][counter]
                counter += 1

            for q in range(model["order"][2]):
                self.ma[idx,q] = model["params"][counter]
                counter += 1
            
            for P in range(model["s_order"][0]):
                self.ar[idx,(P+1)*model["s_order"][3]-1] = model["params"][counter]
                counter += 1
            
            for Q in range(model["s_order"][2]):
                self.ma[idx,(Q+1)*model["s_order"][3]-1] = model["params"][counter]
                counter += 1

            self.error_var[idx] = model["params"][-1]
