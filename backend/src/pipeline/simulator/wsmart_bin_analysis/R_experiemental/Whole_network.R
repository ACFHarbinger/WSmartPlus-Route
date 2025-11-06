setwd("C:/Users/Utilizador/OneDriveUL/Desktop/Masters/Wsmart+Route/Initial_studies")

library(tseries)
library(data.table)
library(forecast)
library(xts)

#### IMPORT DATA HEADER #####

#DATA DAY
df_lin <- read.csv("out_CRUDE_RATE.csv", header = TRUE)
names(df_lin)[names(df_lin) == 'X'] <- 'Date'
colnames(df_lin) <- sub("^X", "", colnames(df_lin))
rownames(df_lin) <- as.POSIXct(df_lin$Date, format = "%Y-%m-%d ")

#DATA 4H
df_interval <- read.csv("out_INTERVAL_RATE.csv", row.names = 'name', check.names = FALSE)
colnames(df_interval) <- as.POSIXct(colnames(df_interval), format = "%Y-%m-%d %H:%M")
df_crude <- transpose(df_interval)
df_crude <- df_crude[order(rownames(df_crude)), ]
rownames(df_crude) <- colnames(df_interval)
colnames(df_crude) <- rownames(df_interval)
rm(df_interval)

#INFO
df_info <- read.csv("out_INFO.csv", row.names = 'ID')
rownames(df_info) <- as.integer(rownames(df_info))

### SPLITING PAPEL PLASTICO ####

tipo = unique(df_info$Tipo.de.Residuos)

filtered_ids_c <- rownames(df_info)[which(df_info[, "Tipo.de.Residuos"] == tipo[[1]])]
filtered_ids_p <- rownames(df_info)[which(df_info[, "Tipo.de.Residuos"] == tipo[[2]])]

df_lin_p <- df_lin[,filtered_ids_p]
df_lin_c <- df_lin[,filtered_ids_c]

### GAUSSIAN RESIDUALS FOR RANDOM PALSTIC/CARD BINS ##

id_p <- sample(colnames(df_lin_p), 1)
id_c <- sample(colnames(df_lin_c), 1)
id_p <- "13491"
id_c <- "10835"

checkresiduals(df_lin_p[[id_p]], main = paste("plastic: ", id_p))
checkresiduals(df_lin_c[[id_c]], main = paste("plastic: ", id_c))

### DISCRETIZATION OF THE VALUES ###
# Customizing the histogram
hist(df_lin_p[[id_p]],
     breaks = 200,       # Number of breaks (bins)
     main = "Histogram of Sample Data",  # Title of the plot
     xlab = "Values",   # Label for the x-axis
     ylab = "Frequency",# Label for the y-axis
     col = "skyblue",  # Color of the bars
     border = "black",  # Border color of the bars
     xlim = c(-100, 100)    # Limits for the x-axis
)



### PLOTING WITH MOVING AVERAGE ###
par(mfrow=c(1,1))
xts_p <- as.xts(df_lin_p[[id_p]] , order.by = as.POSIXct(rownames(df_lin_p)))
xts_c <- as.xts(df_lin_c[[id_c]] , order.by = as.POSIXct(rownames(df_lin_c)))

moving_avg <- rollmean(xts_p, k = 60, align = "center")
moving_avg2 <- rollmean(xts_c, k = 60, align = "center")

plot(xts_p, lwd = 0.1, col = "darkblue", ylim=c(-20,60),
     main = id_p)
lines(moving_avg, col = "darkorange", lwd = 3.5)

plot(xts_c, lwd = 0.1, col = "darkblue", ylim=c(-20,60),
     main = id_c)
lines(moving_avg2, col = "darkorange", lwd = 3.5)

pacf(na.omit(coredata(xts_p)))
pacf(na.omit(coredate(xts_c)))

### PLOTTING VALUES FOR THE WHOLE BINS ### 

xts_avg_p <- as.xts(rowMeans(df_lin_p, na.rm = TRUE), order.by = as.POSIXct(rownames(df_lin)))
xts_avg_c <- as.xts(rowMeans(df_lin_c, na.rm = TRUE), order.by = as.POSIXct(rownames(df_lin)))

moving_avg <- rollmean(xts_avg_p, k = 60, align = "center")

moving_avg2 <- rollmean(xts_avg_c, k = 60, align = "center")

plot(xts_avg_c, lwd = 0.1, col = "darkred", ylim=c(0,20),
        main = paste(paste(tipo[[2]], " (red); "),paste(tipo[[1]], " (green); window = 60 days") ))
lines(xts_avg_p, lwd = 0.01, col = "darkgreen")
lines(moving_avg2, col = "red", lwd = 4)

lines(moving_avg, col = "green", lwd = 3)

### ACF AND PACF FOR THE WHOLE###

par(mfrow=c(2,2))
acf1 <- acf(coredata(xts_avg_c), plot = FALSE, lag.max = 60)
acf2 <- acf(coredata(xts_avg_p), plot = FALSE, lag.max = 60)
acf1$acf[1] = 0
acf2$acf[1] = 0
plot(acf1, main = tipo[[1]])
plot(acf2, main = tipo[[2]])

par(mfrow=c(1,2))
pacf(coredata(xts_avg_c), main = tipo[[1]],lag.max = 60) 
pacf(coredata(xts_avg_p), main = tipo[[2]], lag.max = 60)

### FITTING ###

ts_p <- xts_avg_p
ts_c <- xts_avg_c

bm_p = Arima(ts_p, order=c(0,0,0))
bm_c = Arima(ts_c, order=c(0,0,0))

accuracy(bm_p)
bm_p
accuracy(bm_c)
bm_c

bm_p_ma = Arima(ts_p, order=c(0,0,1))
bm_c_ma = Arima(ts_c, order=c(0,0,1))

accuracy(bm_p_ma)
bm_p_ma
accuracy(bm_c_ma)
bm_c_ma

bm_p_ar = Arima(ts_p, order=c(1,0,0))
bm_c_ar = Arima(ts_c, order=c(1,0,0))

accuracy(bm_p_ar)
bm_p_ar
accuracy(bm_c_ar)
bm_c_ar

bm_p_arma = Arima(ts_p, order=c(1,0,1))
bm_c_arma = Arima(ts_c, order=c(1,0,1))


tsdiag(bm_p_arma)
accuracy(bm_p_arma)
bm_p_arma
accuracy(bm_c_arma)
bm_c_arma

bm_p_a = auto.arima(ts_p)
bm_c_a = auto.arima(ts_c)

accuracy(bm_p_a)
bm_p_a
accuracy(bm_c_a)
bm_c_a

### Fitting with Seasoning ###
bm_p_s7 = Arima(head(ts_p,-15),order=c(4,0,1), seasonal=list(order=c(1,0,1),period=7))
bm_c_s7 = Arima(ts_c,order=c(4,0,5), seasonal=list(order=c(1,0,1),period=7))



accuracy(bm_p_s7)
bm_p_s7
accuracy(bm_c_s7)
bm_c_s7


forecast_values <- forecast(bm_p_s7, h = 30)

# Plot the forecast along with the last 10 days
plot(forecast_values, xlim = c(3.3*10^6, 3.8*10^6), xlab = "Time", ylab = "Values", main = "SARMA Forecast for Average Plastic Bins")
lines(window(fitted(bm_p_s7), start = length(bm_p_s7$x) - 300), col = "blue")  # Plot the last 10 days in blue

# Add a legend
legend("topright", legend = c("Real Values", "Fited Process"), col = c("black", "blue"), lty = 1)


# Plot just the last 10 days along with the forecast
last_10_days <- tail(time(ts_p), 10)  # Assuming 'series' is your time series data
plot(last_10_days, type = "l", xlab = "Time", ylab = "Values", main = "Last 10 Days with Zoomed Forecast")

# Zoom in on the forecast by setting xlim
xlim_forecast <- c(last_10_days[1], last_10_days[length(last_10_days)] + 10)  # Extend by 10 days for forecast
lines(forecast_values$mean, col = "blue", xlim = xlim_forecast)

# Add a legend
legend("topright", legend = c("Last 10 Days", "Forecast"), col = c("black", "blue"), lty = 1)


