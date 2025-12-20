import os
import math
import torch
import pickle
import pandas
import numpy as np
import scipy.stats as stats

from .wsmart_bin_analysis import GridBase
from .loader import load_area_and_waste_type_params


class Bins:
    def __init__(self, n, data_dir, sample_dist="gamma", grid=None, area=None, waste_type=None, indices=None, waste_file=None):
        assert sample_dist in ["emp", "gamma"]
        self.n = n
        _, revenue, density, expenses, bin_volume = load_area_and_waste_type_params(area, waste_type)
        self.revenue = revenue
        self.density = density
        self.expenses = expenses
        self.volume = bin_volume

        self.c = np.zeros((n))
        self.means = np.zeros((n))
        self.std = np.zeros((n))
        self.day_count = 0
        self.square_diff = np.zeros((n))
        self.start_with_fill = False
        
        self.lost = np.zeros((n))
        self.distribution = sample_dist
        self.dist_param1 = np.ones((n))*10
        self.dist_param2 = np.ones((n))*10
        self.inoverflow = np.zeros((n))
        self.collected = np.zeros((n))
        self.ncollections = np.zeros((n))
        self.history = []
        self.travel = 0
        self.profit = 0
        self.ndays = 0
        self.collectdays = np.ones((n))*5
        self.collectlevl = np.ones((n))*80
        self.data_dir = data_dir
        if indices is None:
            self.indices = np.array(range(n))
        else:
            self.indices = np.array(indices)
        if grid is None and sample_dist == "emp":
            src_area = area.translate(str.maketrans('', '', '-_ ')).lower()
            waste_csv = f"out_rate_crude[{src_area}].csv"
            info_csv = f"out_info[{src_area}].csv"
            
            # Read info file to map indices to IDs
            info_df = pandas.read_csv(os.path.join(data_dir, 'coordinates', info_csv))
            real_ids = info_df.iloc[self.indices]['ID'].tolist()
            
            # Check ID type in waste csv
            waste_path = os.path.join(data_dir, 'bins_waste', waste_csv)
            waste_header = pandas.read_csv(waste_path, nrows=0).columns
            if pandas.api.types.is_string_dtype(waste_header):
                real_ids = [str(i) for i in real_ids]
            
            self.grid = GridBase(real_ids, data_dir, rate_type="crude", names=[waste_csv, info_csv, None], same_file=True)
            self.grid.ids_map = {i: real_id for i, real_id in zip(self.indices, real_ids)}
        else:
            self.grid = grid
        if waste_file is not None:
            with open(os.path.join(data_dir, waste_file), 'rb') as file:
                self.waste_fills = pickle.load(file)
        else:
            self.waste_fills = None

    def __get_stdev(self):
        if self.day_count > 1:
            variance = self.square_diff / (self.day_count - 1)
            return np.sqrt(variance)
        else:
            return np.zeros(self.n)

    def _predictdaystooverflow(self, ui, vi, f, cl):
        n = np.zeros(ui.shape[0]) + 31
        for ii in np.arange(1,31,1):
            k = ii*ui**2/vi
            th = vi/ui
            aux = np.zeros(ui.shape[0]) + 31
            p = 1-stats.gamma.cdf(100-f, k, scale=th)
            aux[np.nonzero(p > cl)[0]] = ii
            n = np.minimum(n, aux)
            if (p > cl).all():
                return n
    
    def set_statistics(self, stats_file):
        data = pandas.read_csv(os.path.join(self.data_dir, stats_file))
        self.means = np.maximum(data['Mean'].values.astype(np.float64), 0)
        self.std = np.maximum(data['StD'].values.astype(np.float64), 0)
        self.day_count = np.maximum(data.at[0, 'Count'].astype(np.int64), 0)
        self.square_diff = (self.std ** 2) * (self.day_count - 1)
        self.start_with_fill = True
    
    def is_stochastic(self):
        return self.waste_fills is None
    
    def get_fill_history(self, device=None):
        if device is not None:
            return torch.tensor(self.history, dtype=torch.float, device=device)
        else:
            return np.array(self.history)

    def predictdaystooverflow(self, cl):
        return self._predictdaystooverflow(self.means, self.std, self.c, cl)
    
    def set_indices(self, indices=None):
        if not indices is None:
            self.indices = indices
        else:
            self.indices = list(range(self.n))

    def set_sample_waste(self, sample_id):
        self.waste_fills = self.waste_fills[sample_id]
        if self.start_with_fill: self.c = self.waste_fills[0]

    def collect(self, idsfull, cost=0):
        ids = set(idsfull)
        total_collected = np.zeros((self.n))
        if len(ids) <= 2: 
            return total_collected, 0, 0, 0
        
        ids.remove(0)
        self.ndays += 1
        ids = np.array(list(ids)) - 1
        collected = (self.c[ids] / 100) * self.volume * self.density
        self.collected[ids] += collected
        self.ncollections[ids] += 1
        total_collected[ids] += collected
        self.c[ids] = 0
        self.travel += cost
        profit = np.sum(total_collected) * self.revenue - cost * self.expenses
        self.profit += profit 
        return total_collected, np.sum(collected), ids.size, profit
    
    def _process_filling(self, todaysfilling):
        """
        Processes the filling data, handles overflows, updates state variables, 
        and calculates returns.
        """
        # Update mean and standard deviation using Welford's method
        self.history.append(todaysfilling)
        todaysfilling = np.array(todaysfilling)
        old_means = self.means.copy()

        self.day_count += 1
        delta = todaysfilling - old_means
        self.means += delta / self.day_count
        self.square_diff += delta * (todaysfilling - self.means)
        self.std = self.__get_stdev()

        # Lost overflows
        todays_lost = (np.maximum(self.c + todaysfilling - 100, 0) / 100) * self.volume * self.density
        todaysfilling = np.minimum(todaysfilling, 100)    
        self.lost += todays_lost

        # New depositions for the overflow calculation
        self.c = np.minimum(self.c + todaysfilling, 100)
        self.c = np.maximum(self.c, 0)
        inoverflow = (self.c==100)
        self.inoverflow += (self.c==100)
        return int(np.sum(inoverflow)), np.array(todaysfilling), np.array(self.c), np.sum(todays_lost)

    def stochasticFilling(self, n_samples=1, only_fill=False):
        if self.distribution == 'gamma':
            todaysfilling = np.random.gamma(self.dist_param1, self.dist_param2, size=(n_samples, self.n))
            if n_samples <= 1: todaysfilling = todaysfilling.squeeze(0)
        elif self.distribution == 'emp':
            sampled_value = self.grid.sample(n_samples=n_samples)
            todaysfilling = np.maximum(sampled_value, 0)
  
        if only_fill:
            return np.minimum(todaysfilling, 100)
        else:
            return self._process_filling(todaysfilling)

    def deterministicFilling(self, date):
        todaysfilling = self.grid.get_values_by_date(date, sample=True)
        return self._process_filling(todaysfilling)
    
    def loadFilling(self, day):
        todaysfilling = self.waste_fills[day] if self.start_with_fill else self.waste_fills[day-1]
        return self._process_filling(todaysfilling)

    def __setDistribution(self, param1, param2):
        if len(param1)==1:
            self.dist_param1 = np.ones((self.n))*param1
            self.dist_param2 = np.ones((self.n))*param2
        else:
            self.dist_param1 = param1
            self.dist_param2 = param2
        self.setCollectionLvlandFreq()

    def setGammaDistribution(self, option=0):
        def __set_param(param):
            param_len = len(param)
            if self.n == param_len:
                return param
            
            param = param * math.ceil(self.n / param_len)
            if self.n % param_len != 0:
                param = param[:param_len-self.n % param_len]
            return param
    
        self.distribution = 'gamma'
        if option == 0:
            k = __set_param([5, 5, 5, 5, 5, 10, 10, 10, 10, 10])
            th = __set_param([5, 2])
        elif option == 1:
            k = __set_param([2, 2, 2, 2, 2, 6, 6, 6, 6, 6])
            th = __set_param([6, 4])
        elif option == 2:
            k = __set_param([1, 1, 1, 1, 1, 3, 3, 3, 3, 3])
            th = __set_param([8, 6])
        else:
            assert option == 3
            k = __set_param([5, 2])
            th = __set_param([10])
        self.__setDistribution(k, th)

    def freqvisit2(self, ui, vi, cf):
        # a = gamma.cdf(30, k, scale=th)
        # c = gamma.ppf(a, k, scale=th)
        # print(a,c)
        for n in range(1,50):
            k = n*ui**2/vi
            th = vi/ui
            if n==1:
                ov = 100-stats.gamma.ppf(1-cf, k, scale=th)

            v = stats.gamma.ppf(1-cf, k, scale=th)
            if v>100:
                return n, ov

    def setCollectionLvlandFreq(self, cf = 0.9):
        for ii in range(0,self.n):
            f2,lv2 = self.freqvisit2(self.dist_param1[ii]*self.dist_param2[ii],
                                    self.dist_param1[ii]*self.dist_param2[ii]**2,cf)
            self.collectdays[ii] = f2
            self.collectlevl[ii] = lv2
        return
