import os
import math
import pickle
import numpy as np
import scipy.stats as stats

from .wsmart_bin_analysis import OldGridBase


class Bins:
    def __init__(self, n, data_dir, sample_dist="gamma", grid=None, area=None, indices=None, waste_file=None):
        assert sample_dist in ["emp", "gamma"]
        self.n = n
        self.c = np.zeros((n))
        self.means = np.ones((n))*10
        self.std = np.ones((n))*1
        self.lost = np.zeros((n))
        self.distribution = sample_dist
        self.dist_param1 = np.ones((n))*10
        self.dist_param2 = np.ones((n))*10
        self.inoverflow = np.zeros((n))
        self.collected = np.zeros((n))
        self.ncollections = np.zeros((n))
        self.history = []
        self.travel = 0
        self.ndays = 0
        self.collectdays = np.ones((n))*5
        self.collectlevl = np.ones((n))*80
        self.data_dir = data_dir
        if indices is None:
            self.indices = np.array(range(n))
        else:
            self.indices = np.array(indices)
        if grid is None and sample_dist == "emp":
            self.grid = OldGridBase(data_dir, area)
        else:
            self.grid = grid
        if waste_file is not None:
            with open(os.path.join(data_dir, waste_file), 'rb') as file:
                self.waste_fills = pickle.load(file)
        else:
            self.waste_fills = None

    def _predictdaystooverflow(self, ui, vi, f, cl):
        n = np.zeros(ui.shape[0])+31
        for ii in np.arange(1,31,1):
            k = ii*ui**2/vi
            th = vi/ui
            aux = np.zeros(ui.shape[0])+31
            p = 1-stats.gamma.cdf(100-f, k, scale=th)
            aux[np.nonzero(p>cl)[0]]=ii
            n = np.minimum(n,aux)
            if (p>cl).all():
                return n
    
    def is_stochastic(self):
        return self.waste_fills is None
    
    def get_fill_history(self):
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

    def collect(self, idsfull):
        ids = set(idsfull)
        ids.remove(0)
        if len(ids) == 0: 
            return 0, 0
        
        ids = np.array(list(ids)) - 1
        self.collected[ids] += self.c[ids]
        self.ncollections[ids] += 1
        collected = np.sum(self.c[ids])
        self.c[ids] = 0
        return collected, ids.size

    def predictdaystooverflow(self, cl):
        return self._predictdaystooverflow(self.means, self.std, self.c, cl)

    def stochasticFilling(self, n_samples=1, only_fill=False):
        if self.distribution == 'gamma':
            todaysfilling = np.random.gamma(self.dist_param1, self.dist_param2, size=(n_samples, self.n))
            if n_samples <= 1: todaysfilling = todaysfilling.squeeze(0)
        elif self.distribution == 'emp':
            sampled_value = self.grid.sample(n_samples=n_samples)
            if n_samples <= 1:
                todaysfilling = np.maximum(np.take(sampled_value, self.indices), 0)
            else:
                todaysfilling = np.maximum(np.take(sampled_value, self.indices, axis=1), 0)

        # Lost overflows
        todays_lost = np.maximum(self.c + todaysfilling - 100, 0)
        todaysfilling = np.minimum(todaysfilling, 100)
        if only_fill:
            return todaysfilling            

        # Update history
        self.ndays += 1
        self.history.append(todaysfilling)
        self.lost += todays_lost

        # New depositions - do not change order otherwise
        # xq + vals + vals for the overflow calculation
        self.c = np.minimum(self.c + todaysfilling, 100)
        self.c = np.maximum(self.c, 0)
        self.inoverflow += (self.c==100)
        return np.sum(self.inoverflow), todaysfilling, np.sum(todays_lost)

    def deterministicFilling(self, date):
        todaysfilling = self.grid.get_values_by_date(date, sample=True)

        # Lost overflows
        todays_lost = np.maximum(self.c + todaysfilling - 100, 0)
        todaysfilling = np.minimum(todaysfilling, 100)
        self.lost += todays_lost

        # Update history
        self.ndays += 1
        self.history.append(todaysfilling)

        self.c = np.minimum(self.c + todaysfilling, 100)
        self.c = np.maximum(self.c, 0)
        self.inoverflow += (self.c==100)
        return np.sum(self.inoverflow), todaysfilling, np.sum(todays_lost)
    
    def loadFilling(self, day):
        todaysfilling = self.waste_fills[day]

        # Lost overflows
        todays_lost = np.maximum(self.c + todaysfilling - 100, 0)
        todaysfilling = np.minimum(todaysfilling, 100)
        self.lost += todays_lost

        # Update history
        self.ndays += 1
        self.history.append(todaysfilling)

        self.c = np.minimum(self.c + todaysfilling, 100)
        self.c = np.maximum(self.c, 0)
        self.inoverflow += (self.c==100)
        return np.sum(self.inoverflow), todaysfilling, np.sum(todays_lost)

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
