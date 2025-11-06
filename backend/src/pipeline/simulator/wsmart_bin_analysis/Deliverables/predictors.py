import os 
import warnings
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Predictor():
    def __init__(self, train_data:pd.DataFrame, test_data:pd.DataFrame, fit_name:str = None):
        """
        Initializes the predictor class

        Upon lauch, it will trigger the R scripts that will make the prediction accordign to the ARIMA models.ç
        The residuals will be save in the the class atributes. Only temporary files will be created thaht the script will clean
        to handle the interface. To make it verbose, provide a fit name and all the fit values will be saved in spearate 
        csv files, as well as the ARIMA model weights.
        """
        self.prediction:pd.DataFrame = None
        self.pred_error:pd.DataFrame = None
        self.real_error:pd.DataFrame = None
        self.mean39error:pd.DataFrame = None

        self.date_range:pd.DatetimeIndex = test_data.index.drop_duplicates()
        self.names:list[str]             = ["train.csv", "test.csv", "prediction.csv", "pred_error.csv", "real_error.csv"]

        self.fit_predict(train_data, test_data, fit_name)
        self.fit_39mean(train_data, test_data)
    
    def get_pred_values(self, date) -> tuple[np.ndarray, np.ndarray]:
        return  self.prediction.loc[date,:].to_numpy(), np.sqrt(self.pred_error.loc[date,:].to_numpy())

    def get_real_errors(self) -> np.ndarray:
        return self.real_error.to_numpy()

    def get_39mean_MSE(self) -> np.ndarray:
        return np.sqrt(np.nanmean(np.square(self.mean39error), axis=0))

    def get_MSE(self) -> np.ndarray:
        return  np.sqrt(np.nanmean(np.square(self.real_error), axis=0))

    def get_avg_dispersion(self) -> np.ndarray:
        return np.mean(np.abs(self.pred_error), axis=0)
    
    def get_pred_errors(self) -> np.ndarray:
        return np.abs(self.pred_error.to_numpy())
    
    def fit_39mean(self, train:pd.DataFrame, test:pd.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            train_filled = train.fillna(train.mean())
       
        if len(train_filled) < 40:
            column_means = np.nanmean(train_filled, axis=0)
            new_matrix = np.tile(column_means, (40-len(train_filled), 1))
            padding_rows = pd.DataFrame(new_matrix, columns=train.columns)
            last_40_train_rows = pd.concat([padding_rows, train_filled], ignore_index=True)
        else:
            last_40_train_rows = train_filled.tail(40)

        combined_df = pd.concat([last_40_train_rows, test], ignore_index=True, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            combined_df = combined_df.fillna(combined_df.mean())

        self.mean39error = (test - combined_df.rolling(window=40).mean().iloc[39:-1]).abs()
        # self.mean39error = (test - combined_df.iloc[39:-1]).abs()

    def fit_predict(self, train_data:pd.DataFrame, test_data:pd.DataFrame, fit_name:str = None):
        self.save_cache(train_data, test_data)
        try:
            if fit_name is None:
                subprocess.run(["Rscript", "Arima_predictor.R"] + self.names, check=True)
            else:
                subprocess.run(["Rscript", "Arima_predictor.R"] + self.names + [" " + fit_name], check=True)
            self.load_cache()
        except:
            raise "There is a problem with the R script"
        
        finally:
            self.deleate_cache()
    
    def save_cache(self, train_data:pd.DataFrame, test_data:pd.DataFrame):
        train_data.to_csv(self.names[0], index = False)
        test_data.to_csv(self.names[1],  index = False)
    
    def load_cache(self):
        self.prediction  = pd.read_csv(self.names[2])
        self.pred_error  = pd.read_csv(self.names[3])
        self.real_error  = pd.read_csv(self.names[4])

        self.prediction.index = self.date_range
        self.pred_error.index  = self.date_range
        self.real_error.index  = self.date_range

        print("Predicted Sucessfully for all bins!")
        
    def deleate_cache(self):
        for filename in self.names:
            try:
                os.remove(filename)
            except:
                continue
    
    def plot_predictions(self, index, start_date:pd.Timestamp, end_date:pd.Timestamp, real_data:pd.DataFrame, info:pd.DataFrame, residuals_header:str, ylim = (-20,60), fig_size:tuple = (9,6)):
        """
        Plots the fill level of the bins with associated collections 

        Parameters
        ----------
        fig_size: tuple of integers
            alias for plt.figure(figsize=fig_size)
        end/start_date: string with format %d/%m/%Y.
            Period to be analysed. First day is inclusive staring at 00:01 last day is exclusive.
        """
        start_date = pd.to_datetime(start_date, format = "%d-%m-%Y", errors = 'raise')
        end_date   = pd.to_datetime(end_date,   format = "%d-%m-%Y", errors = 'raise')

        real_values = real_data[start_date:end_date].iloc[:, index]
        predicted   = self.prediction[start_date:end_date].iloc[:, index]
        dispersion  = np.abs(self.pred_error[start_date:end_date].iloc[:, index])

        plt.figure(figsize=fig_size)
        plt.plot(real_values.index, real_values, linestyle='-', color='black', linewidth=0.8)
        plt.scatter(real_values.index, real_values, marker='o', color='red', s=12, label="real_data")

        (_ , caps, _ ) = plt.errorbar(real_values.index, predicted, yerr = dispersion, fmt='none', elinewidth=0.5, ecolor="black", capthick=2, capsize=4)
        plt.scatter(real_values.index, predicted, marker='o', label="Prediction", s=7, color="blue")
        # for cap in caps:
        #     cap.set_markeredgewidth(1)

        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Rate')
        plt.xticks(rotation=45)
        plt.title('Container ID:' + str(int(info['ID'].item())) + '; ' + str(info[residuals_header].item()))
        plt.legend(loc = 'upper left')
        plt.grid()
        plt.ylim(ylim[0],ylim[1])
        plt.show()
