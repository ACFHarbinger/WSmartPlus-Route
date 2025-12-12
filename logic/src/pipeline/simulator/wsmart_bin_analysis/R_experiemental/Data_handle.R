
####  IMPORT AND TREAT DATA ####

setwd("C:/Users/Utilizador/OneDriveUL/Desktop/Masters/Wsmart+Route/Initial_studies")

library(tseries)
library(data.table)
library(forecast)
library(xts)


df_lin <- read.csv("out_CRUDE_RATE.csv")

df_interval <- read.csv("out_INTERVAL_RATE.csv", row.names = 'name', check.names = FALSE)
colnames(df_interval) <- as.POSIXct(colnames(df_interval), format = "%Y-%m-%d %H:%M")
df_crude <- transpose(df_interval)
df_crude <- df_crude[order(rownames(df_crude)), ]

rownames(df_crude) <- colnames(df_interval)
colnames(df_crude) <- rownames(df_interval)

rm(df_interval)

df_info <- read.csv("out_INFO.csv", row.names = 'ID')
rownames(df_info) <- as.integer(rownames(df_info))


names(df_lin)[names(df_lin) == 'X'] <- 'Date'
#names(df_crude)[names(df_crude) == 'X'] <- 'Date'

colnames(df_lin) <- sub("^X", "", colnames(df_lin))
rownames(df_lin) <- as.POSIXct(df_lin$Date, format = "%Y-%m-%d ")
#colnames(df_crude) <- sub("^X", "", colnames(df_crude))

#df_crude$Date <- as.Date(df_crude$Date)

#df_crude[is.na(df_crude)] <- 0


ids <- c("1621","16540","17812","17821","17948","18177")

id = "13647"

df_lin$Date <- NULL
df_crudeM <- rowMeans(df_lin, na.rm = TRUE)

tipo = unique(df_info$Tipo.de.Residuos)

dfs_tipo <- list()

for (t in tipo){
  dfs_list[[t]] =  data.frame()
}

xts_avg = as.xts(df_crudeM, order.by = as.POSIXct(rownames(df_lin)))
moving_avg <- rollmean(xts_avg, k = 90, align = "center")

plot(xts_avg, lwd = 0.1, ylim=c(0,20))
lines(moving_avg, col = "darkblue", lwd = 3)

plot(df_crudeM)


### TRANSLTATE TO TIME SERIES DATA TYPE WITH XTS ###

ts_daily <- as.xts(na.omit(df_lin[[id]]), order.by = as.POSIXct(rownames(df_lin))[!is.na(df_lin[[id]])])

plot(ts_daily, main = paste("Rate of Bin", id))

par(mfrow=c(2,1))
acf1 = acf(coredata(ts_daily), plot=FALSE)
acf1$acf[1] <- 0
plot(acf1, main = paste("ACF for bin ", id))
pacf(coredata(ts_daily), main = paste("PACF for bin ", id) )



moving_avg <- rollmean(na.omit(ts_daily), k = 30, align = "center")

plot(na.omit(ts_daily), lwd = 1.5, main = paste(paste("Plot",id), df_info[id,"Tipo.de.Residuos"]))
lines(moving_avg, col = "darkblue", lwd = 3)

# Plot a histogram of the variances
hist(variances)

ts_4H <- as.xts(df_crude[[id]], order.by = as.POSIXct(rownames(df_crude)))

calculate_variance <- function(column) {
  var(na.omit(column))
}

variances <- numeric(length(df_lin))

for (i in seq_along(df_lin)) {
  # Remove NA values and calculate the variance for each column
  variances[i] <- var(na.omit(df_lin[[i]]))
}

hist(variances)


acf(na.omit(coredata(ts_daily)))

model = auto.arima(na.omit(coredata(ts_daily)))

model

checkresiduals((model))

new_data <- simulate(model, nsim=30, future= TRUE)

plot(new_data)

### AVERAGE OF ALL BINS ###



####  FITTING MODELS TO IDS  ####



i = 1
for (id in ids){
  
  
  
  # par(mfrow=c(1,1))
  # plot(ts_4H, xlab = "Date", ylab = "Rate", main = paste(paste("Plot",id), df_info[id,"Tipo.de.Residuos"]))
  
  # par(mfrow=c(2,1))
  # acf2<- acf(na.omit(coredata(ts_4H)), plot = FALSE, main = paste(c("ACF_4H for", id)))
  # 
#   # acf2$acf[1] = 0
#   # plot(acf2, main = paste(c("ACF_M for", id)))
#   # 
#   # pacf2<- pacf(na.omit(coredata(ts_4H)), plot = TRUE, main = paste(c("PACF_4H for", id)))
#   
#   # my_model = auto.arima(na.omit(coredata(ts_4H)))
#   
#   # tsdiag(my_model)
# 
#   # print(my_model)
#   
#   # print(accuracy(Arima(na.omit(coredata(ts_4H)), order = c(0,0,0))))
# #  print(accuracy(my_model))
#   
#   
#   ##### Monthly + Weekly cenas ####
#   par(mfrow=c(1,1))
#   ts_daily <- as.xts(df_lin[[id]], order.by = df_lin$Date)
#   
#   # plot(ts_daily)
#   
#   ts_weekly <- apply.weekly(ts_daily, FUN = colMeans);
#   
#   plot(ts_weekly)
#   # 
#   ts_month <- apply.monthly(ts_daily, FUN = colMeans);
#   
#   plot(ts_month)
#   
#   plot(ts_4H, xlab = "Date", ylab = "Rate", main = paste(paste("Plot",id), df_info[id,"Tipo.de.Residuos"]))
#   #
#   
#   par(mfrow=c(2,1))
#   acf2<- acf(na.omit(coredata(ts_4H)), plot = FALSE, main = paste(c("ACF_4H for", id)))
# 
#   acf2$acf[1] = 0
#   plot(acf2, main = paste(c("ACF_M for", id)))
# 
#   pacf2<- pacf(na.omit(coredata(ts_4H)), plot = TRUE, main = paste(c("PACF_4H for", id)))
#   
#   par(mfrow=c(2,1))
#   acf2<- acf(na.omit(coredata(ts_weekly)), plot = TRUE, main = paste(c("ACF_Week for", id)))
#   #plot(acf2, main = paste(id," acf "))
# 
#   pacf2<- pacf(na.omit(coredata(ts_weekly)), plot = TRUE, main = paste(c("PACF_Week for", id)))
#   
#   acf2<- acf(na.omit(coredata(ts_month)), plot = TRUE, main = paste(c("ACF_Month for", id)))
#   #plot(acf2, main = paste(id," acf "))
#   
#   pacf2<- pacf(na.omit(coredata(ts_month)), plot = TRUE, main = paste(c("PACF_Month for", id)))
  
  #plot(pacf2, main = paste(id," pacf "))

  # base_model = Arima(ts_weekly, order=c(0,0,0))
  # my_M_model = auto.arima(ts_weekly)
  
  # checkresiduals(df_lin[[id]])
  
  # 
  # tsdiag(my_M_model, main = id)
  # 
  # print(my_M_model)
  # print(accuracy(my_M_model))
  # print(accuracy(base_model))
  
  # forecast_values2 <- forecast(my_M_model, h=3)
  
  # par(mfrow=c(1,1))
  # plot(forecast_values2, main = paste("Predict 0: ", id), xlim=c(140,160))
      #      xlim=c(tail(time(ts_weekly),1) - 4, tail(time(ts_weekly),1) + 4))
      # axis.Date(1, at = pretty(time(ts_weekly)), format = "%Y-%m-%d")
  # 
  # #acf2<- acf(ts_weekly, plot = TRUE, main = paste(c("ACF_W for", id)))
  # #plot(acf2, main = paste(id," acf "))
  # 
  # #pacf2<- pacf(ts_weekly, plot = TRUE, main = paste(c("PACF_W for", id)))
  # #plot(pacf2, main = paste(id," pacf "))
  # 
  # base_model = Arima(ts_weekly, order=c(0,0,0))
  # my_W_model = auto.arima(ts_weekly)
  # 
  # tsdiag(my_W_model, main = id)
  # 
  # print(my_W_model)
  # print(accuracy(my_W_model))
  # print(accuracy(base_model))
  
  # acf2<- acf(ts_daily, plot = TRUE, main = paste(c("ACF_D for", id)))
  # #plot(acf2, main = paste(id," acf "))
  # 
  # pacf2<- pacf(ts_daily, plot = TRUE, main = paste(c("PACF_D for", id)))
  # #plot(pacf2, main = paste(id," pacf ")) 
  
  
  # forecast_values2 <- forecast(Arima(ts_series, order=c(0,0,0)), h=10)
  # forecast_values <- forecast(auto.arima(ts_series), h=10)
  # 
  # plot(forecast_values2, main = paste("Predict 0: ", id),
  #      xlim=c(tail(time(ts_series),1) - 20, tail(time(ts_series),1) + 10))
  # axis.Date(1, at = pretty(time(ts_series)), format = "%Y-%m-%d")
  # 
  # plot(forecast_values, main = paste("Predict ARIMA auto: ", id),
  #      xlim=c(tail(time(ts_series),1) - 20, tail(time(ts_series),1) + 10))
  # axis.Date(1, at = pretty(time(ts_series)), format = "%Y-%m-%d")
  


  #### Preety Plot ####
  
  # par(mfrow=c(1,1))
  # plot(ts_series, xlab = "Date", ylab = "Rate", main = paste(paste("Plot",id), df_info[id,"Tipo.de.Residuos"]), xaxt = "n")
  # axis.Date(1, at = pretty(time(ts_series)), format = "%Y-%m-%d")
  
  ####  PLOT ACF and PACF   #####
  
  #acf2<- acf(ts_month, plot = TRUE, main = paste(c("ACF_M for", id)))
  #plot(acf2, main = paste(id," acf "))

  #pacf2<- pacf(ts_month, plot = TRUE, main = paste(c("PACF_M for", id)))
  #plot(pacf2, main = paste(id," pacf "))

  
  #### Fiting ARIMA MODELS ####
  
  # par(mfrow=c(2,1))
  # print(paste("Forecast for ID: ", id))
  # 
  # mymodel_MA = Arima(ts_series, order=c(0,0,1))
  # 
  # print(paste("MA(1)", id))
  # 
  # print(mymodel_MA)
  # tsdiag(mymodel_MA)
  # 
  # print(accuracy(mymodel_MA))
  # 
  # forecast_values <- forecast(mymodel_MA, h=10)
  # 
  # #### Differncing TO EXTRACTSEASONAL TERMS ###
  # 
  # diff_series = Arima(ts_series, order = c(0,0,1), seasonal = list(order=c(2,0,0), period=resid[i]))
  # 
  # print(paste("ARIMA Seasonal", id))
  # 
  # print(diff_series)
  # print(accuracy(diff_series))
  # 
  # tsdiag(diff_series)
  # 
  # forecast_values2 <- forecast(diff_series, h=10)
  # 
  # 
  # par(mfrow=c(2,1))
  # plot(forecast_values2, main = paste("Predict Seasonal: ", id),
  #      xlim=c(tail(time(ts_series),1) - 20, tail(time(ts_series),1) + 10))
  # axis.Date(1, at = pretty(time(ts_series)), format = "%Y-%m-%d")
  # 
  # plot(forecast_values, main = paste("Predict MA: ", id),
  #      xlim=c(tail(time(ts_series),1) - 20, tail(time(ts_series),1) + 10))
  # axis.Date(1, at = pretty(time(ts_series)), format = "%Y-%m-%d")
  # 
  # 
  
  
  
  
  i = i +1
  
}














