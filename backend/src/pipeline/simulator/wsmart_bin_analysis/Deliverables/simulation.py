import os
import numpy as np
import pandas as pd

from .predictors import Predictor
from .save_load import load_rate_series, load_info, load_rate_global_wrapper


class GridBase():
    """
    Class that manages the background behind a containers ensemble.

    It receives a csv file, and pre-processes the necessary data in order to allow sampling.
    """
    def __init__(self, ids, data_dir, rate_type, info_ver = None, names = None) -> None:
        self.data:pd.DataFrame = None
        self.__info:dict = None
        self.__freq_table:pd.DataFrame = None
        self.__data_dir:str = None

        self.__data_dir = data_dir
        self.data, self.__info = self.load_data(processed = False, 
                                                  ids=ids, data_dir=self.__data_dir, 
                                                  rate_type=rate_type, info_ver=info_ver, names=names)
        
        self.data = self.data.select_dtypes(include='number').round(0)
        self.__freq_table = self.cacl_freq_tables()

    def load_data(self, ids, data_dir, info_ver = None, names = None, rate_type = None, processed=True, same_file=False) -> tuple[pd.DataFrame, dict]:
        """
        Parameters
        ----------
        ids:list[int],
            list of container ids to load
        data_dir:str,
            files directory 
        rate_type:str,
            Can be 'mean' or 'crude' depending on the the file previously saved
        info_ver:str,
            ver suffix unused in info saving
        names:list[str]
            Names to use instead of the names generator. 
            All should come already with the .csv attached in the end
        processed: bool ,True
            set to True  by default; set if files hae been previously loaded
        same_file: bool, False,
            if the files are in the same file
        
        Returns
        -------

        rates: pd.Dataframe
            The dataframe of processed rates
        info:dict: dict
            Ids dictionary with info from each bin
        """
        waste_dir = os.path.join(data_dir, "bins_waste")
        coords_dir = os.path.join(data_dir, "coordinates")
        if processed:
            return self.data, self.__info
        elif same_file:
            rate = self.__data_preprocess_same_file(pd.read_csv(os.path.join(waste_dir, names[0])))
            rate = rate.loc[:, ids]
            info = pd.read_csv(os.path.join(coords_dir, names[1]))
            return rate, info
        else:
            rate_list = []
            info_dict = {}
            if names == None:
                for id in ids:
                    rate_list = rate_list + [load_rate_series(id, rate_type=rate_type, path=waste_dir)]
                    info_dict[id] = load_info(id, ver=info_ver, path=coords_dir)
            else:
                for i, id in enumerate(ids):
                    if 'rate' in names[i].keys():
                        rate_list = rate_list + [load_rate_series(id, rate_type=rate_type, path=waste_dir, name=names[i]['rate'])]
                    else:
                        rate_list = rate_list + [load_rate_series(id, rate_type=rate_type, path=waste_dir)]
                    
                    if 'info' in names[i].keys():
                        info_dict[id] = load_info(id, ver=info_ver, path=coords_dir, name=names[i]['info'])
                    else:
                        info_dict[id] = load_info(id, ver=info_ver, path=coords_dir)
            print("All loaded")
            return load_rate_global_wrapper(rate_list=rate_list), info_dict
        
    def __data_preprocess_same_file(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        -------
        data: pd.Dataframe,
            The dataframe with dates before preprocessing 

        Returns
        -------
        data: pd.Dataframe,
            The dataframe with dates as row indexes 
        """
        data['Date'] = pd.to_datetime(data['Date'], format = "%Y-%m-%d")
        data = data.set_index('Date')
        return data
        
    def cacl_freq_tables(self) -> pd.DataFrame:
        """
        Returns
        --------
        a dataframe whose columns are the distribution of each rate 
        """
        def count(series:pd.Series):
            return fix10pad(series.value_counts(normalize=True).fillna(0).cumsum())
        
        freq_table = self.data.agg(count,axis=0)
        return freq_table
    
    def get_mean_rate(self) -> dict[float]:
        """
        Returns
        -------
        mean: np.ndarray
            the mean of each bin
        """
        return self.data.mean(axis=0, skipna=True, numeric_only=True).to_numpy()

    def get_var_rate(self) -> np.ndarray:
        """
        Returns
        -------
        var: np.ndarray
            the varaince of each bin
        """
        return self.data.var(axis=0, skipna=True, numeric_only=True).to_numpy()
    
    def get_std_rate(self) -> np.ndarray:
        """
        Returns
        -------
        std: np.ndarray
            the stardard variation of each bin
        """
        return self.data.var(axis=0, skipna=True, numeric_only=True).transform('sqrt').to_numpy()
    
    def get_datarange(self) -> tuple[pd.Timestamp]:
        """
        Returns
        -------
        (start, end): pd.timestemp,
            start and end of the actual rate values
        """
        index = self.data.index
        return index[0], index[-1]
    
    def sample(self) -> np.ndarray:
        """
        Returns
        -------
        sample: np.ndarray
            a sample form each bins rate histogram
        """
        assert self.__freq_table is not None, "Freq tables should be calculated before calling this method. Call self.calc_freq_tables()"
        
        indexes = self.__freq_table.apply(lambda s: np.searchsorted(s, np.random.random(), side = 'right'), raw=True, axis=0)
        sample = self.__freq_table.index[indexes]
        return sample.to_numpy()
    
    def get_values_by_date(self, date, sample: bool = False) -> np.ndarray:
        """    
        Parameters
        ----------
        date: timestamp, str
            Datetime obect (e.g when looping through a datarange) or string in %d-%m-%Y format. 
        sample: bool,
            Weather The cointaners whose value is NaN should be filled with a sample form self.sample()

        Returns
        -------
        rate: np.ndarray
            each container rate in required date. 
        """
        date = pd.to_datetime(date, format='%d-%m-%Y', errors='raise')
        rate = self.___values_by_date(date)
        if sample:
            assert self.__freq_table is not None, \
                "Freq tables should be calculated before calling this method. Call self.calc_freq_tables()"

            samp = self.sample()
            mask = rate.isna()
            rate[mask] = samp[mask]
        return rate.to_numpy()
        
    def ___values_by_date(self, date:pd.Timestamp) -> pd.Series:
        """
        Returns
        -------
        rate: np.darray
            The actual rate row of each bin
        """
        return self.data.loc[date,:]
    
    def values_by_date_range(self, start:pd.Timestamp = None, end:pd.Timestamp = None) -> pd.DataFrame:
        """
        Returns
        -------
        rate: pd.Dataframe
            The actual rate row of each bin per date
        """
        if end == None:
            end = self.data.index[-1]
        else:
            end = pd.to_datetime(end, format='%d-%m-%Y', errors='raise')
        
        if start == None:
            start = self.data.index[0]
        else:
            start = pd.to_datetime(start, format='%d-%m-%Y', errors='raise')
        return self.data.loc[start:end,:]

    def get_info(self, i:int) -> dict:
        """
        Parameters
        ----------
        i: int
            index of the container to fect information from

        Returns
        -------
        info: dict
            the info of the container with the given index
        """
        return self.__info[self.data.columns[i]]
    
    def get_num_bins(self) -> int:
        """
        Returns
        -------
        num_bins: int
            the number of bins in the grid
        """
        return len(self.data.columns)


class Simulation(GridBase):
    def __init__(self, 
                 sim_type:str, 
                 ids:list, 
                 data_dir:str, 
                 train_split:str  = None, 
                 start_date:str   = None, 
                 end_date:str     = None, 
                 rate_type:str    = None,
                 predictQ:bool    = False, 
                 info_ver:str     = None, 
                 names:str        = None,
                 savefit_name:str = None):
        """
        In order to allow for predictions to be made easily, the rates a pre-simulated along the whole data range that is to be used.
        Keeping in mind that simulation, the training and testing of the predictor is done on that data.
        If one wants just the rates a each time step, he can just call *get_current_step* while calling advance_timestep.

        To ease up the work, a little interface was done to suuport trash pickup, simulating fill levels of the bins in Real time, but
        one could perfectly choose not to use it.

        Notice also that this is a superclass of the GrdiBase class which starts a GridBase and runs verything on top of it

        Parameters
        ----------
        ids: list,
            list of ids to load
        sim_type: str, ['sampled', 'real', 'real+samples'],
            If values are sampled, the real ones are used or a mixture of 'real+sample' wheere the real values dont exist
        data_dir:str,
            directory where the rate files are
        train_split: str, pd.datetime;
            split to use for testing/training data. Notie that all the real data is used for training; not taking into acount start/end
        start_date: str, pd.datetime;
            start date of the simulation 
        end_date: str, pd.datetime;
            end date of the simulation
        rate_type: str,
            if the rate type saved is a 'mean' aproximation or 'crude'
        predictQ:bool,
            weather to predict values or not. predictQ set to True will trigger model training
        info_ver: str, None,
            versrion indicative of info files
        names: str, None
            names to be used instead of automatic naming
        savefit_name: str, None
            name of he file with the fitted models. If none, weights are not saved.
        """
        Sim_Keys = ['sampled', 'real', 'real+sampled']
        assert sim_type in Sim_Keys, f"sim_type {sim_type} is no acceptected. Must be one from {Sim_Keys}"
        super().__init__(ids=ids, rate_type=rate_type, info_ver=info_ver, data_dir=data_dir, names=names)
        
        self.fill:np.ndarray         = np.zeros(self.get_num_bins())
        self.sim_type:str            = sim_type
        self.predictQ:bool           = predictQ
        
        self.start_date:pd.Timestamp    = pd.to_datetime(start_date, format="%d-%m-%Y", errors='raise')
        self.split:pd.Timestamp         = pd.to_datetime(train_split, format="%d-%m-%Y", errors='raise')
        self.end_date:pd.Timestamp      = pd.to_datetime(end_date, format="%d-%m-%Y", errors='raise')
        self.current_date:pd.Timestamp  = self.start_date

        self.rates:pd.DataFrame = self.pre_simulate_rates()
        if self.predictQ:
            print("Getting in the Pedictor")
            self.predictor:Predictor = Predictor(self.values_by_date_range(end = self.split), self.rates[self.split:], savefit_name)

    def pre_simulate_rates(self) -> pd.DataFrame:
        """
        Pre_simulates Rates according to set at init
        Returns
        -------
        rate: pd.Dataframe
            dataframe with simulated rates
        """
        date_range = pd.date_range(self.start_date, self.end_date)
        rate_list = []
        for date in date_range:
            if self.sim_type == 'sample':
                rate_list.append(self.sample())
            elif self.sim_type == 'real':
                rate_list.append(self.get_values_by_date(date=date))
            elif self.sim_type == 'real+sampled':
                rate_list.append(self.get_values_by_date(date=date, sample=True))
            else:
                raise "self.type not recognised"

        rate = pd.DataFrame(np.vstack(rate_list))
        rate.index = date_range
        return rate
    
    def reset_simulation(self):
        "Reset the simulation witout changing the pre-simulated rates and predictions"
        self.current_date = self.start_date
        self.fill         = np.zeros(self.get_num_bins())
    
    def get_current_step(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets rates and prediction at the given timestep
        Returns
        -------
        rate: np.ndarray
            current rate

        pred: np.ndarray
            current prediction
        error:
            current predicted error on the prediction
        """
        if self.predictQ and self.current_date > self.split: 
            pred, p_error = self.predictor.get_pred_values(self.current_date)
            return self.rates.loc[self.current_date, : ].to_numpy(), pred, p_error
        else: 
            return self.rates.loc[self.current_date, : ].to_numpy()

    def make_collections(self, bins_index_list:list[int] = None) -> np.ndarray:
        """
        Preforms collections on the bins specified by the index. The index is induced by the order of
        the dataframe.

        Returns
        -------
        collected_junk: np.ndarray
            the collected trash in each of the collected bins using bins_index_list as an index map. 
            It corresponds to the percentage of the bin
        """
        if bins_index_list == None:
            bins_index_list = []
        
        idx = np.array(bins_index_list)
        collected_junk = self.fill[idx]
        self.fill[idx] = 0
        return collected_junk
    
    def advance_timestep(self, date=None) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Advances the timeStep of the simulation by updating the fill level according to the rates the 
        update procedure defined at init.

        Parameters
        ----------
        date: str, datetime
            date to fetch values. string format %d-%m-%Y or datetime

        Returns
        -------
        n_overflows: int
            the number of overflows occurred in that step
        prediction: np.ndarray
            returns an array of prediction made to the corresponding bins
        error: np.ndarray
            returns an array with the mean of the error predicted; i.e 50% of the times, the actual value will
            be in interval [prediction-error, prediction+error]

        """
        self.current_date  = self.current_date + pd.Timedelta('1D')
        rate, prediction, p_error = self.get_current_step()
        
        self.fill = self.fill + rate
        self.fill[self.fill < 0] = 0

        overfmask = (self.fill > 100)
        self.fill[overfmask] = 100
        return np.sum(overfmask), prediction, p_error


### UTILS
def fix10pad(s:pd.Series) -> pd.Series:
    """
    returns a series will all its 1 values padded to 2 without changing the first 1

    a value is verifiied by checkking if is one if 0.9999999>1.

    1-0.9999999 < 1/365*3 ~ 0.0001 Hard Coded, change for much bigger datasets; same goes for zero
    """
    temp = s[s >= 0.9999999]
    temp.iloc[1:] = 2.
    s.loc[s >= 0.9999999] = temp 

    temp = s[s <= 0.0000001]
    temp.iloc[:-1] = -1
    s.loc[s <= 0.0000001] = temp 
    return s
