
setwd("C:/Users/Utilizador/OneDriveUL/Desktop/Masters/Wsmart+Route/Initial_studies")

library("fdapace")
library("fda")
library("lubridate")
library("dplyr")
library("dbscan")
library("EMCluster")
library("forecast")
library("dtwclust")
library("SBAGM")
library(stats)
library(rugarch)

#DATA DAY
df_lin <- read.csv("out_CRUDE_RATE.csv", header = TRUE)
names(df_lin)[names(df_lin) == 'X'] <- 'Date'
colnames(df_lin) <- sub("^X", "", colnames(df_lin))
df_lin$Date <- as.Date(as.POSIXct(df_lin$Date, format = "%Y-%m-%d"))
df_lin <- df_lin[order(df_lin$Date), ]

##Remove first and last days <- RUN EVERYTHING AGAIN
df_lin <- df_lin[-c(1:12),]
df_lin <- df_lin[1:(nrow(df_lin) - 11), ]

#INFO
df_info <- read.csv("out_INFO.csv", row.names = 'ID')
rownames(df_info) <- as.integer(rownames(df_info))

tipo = unique(df_info$Tipo.de.Residuos)

filtered_ids_plastic <- rownames(df_info)[which(df_info[, "Tipo.de.Residuos"] == tipo[[1]])]
filtered_ids_paper <- rownames(df_info)[which(df_info[, "Tipo.de.Residuos"] == tipo[[2]])]

df_lin_plastic <- df_lin[,filtered_ids_plastic]
df_lin_paper <- df_lin[,filtered_ids_paper]

df_lin_plastic$Date <- df_lin$Date
df_lin_paper$Date <- df_lin$Date

#Remove outlier from the df
variance_values <- apply(df_lin_plastic, 2, var)

argmax_variance <- which.max(variance_values)
argmax_variance
plot(df_lin_plastic$"11821", type='l')

df_lin_plastic$"11821" <- NULL
df_lin$"11821" <- NULL

#plot histogram and variance
mean_values <- apply(df_lin_plastic, 2, mean)
variance_values <- apply(df_lin_plastic, 2, var)

hist(mean_values, main = "Histogram of Mean Plastic Values",breaks = 20, xlab = "Mean", col = "blue")
hist(variance_values, main = "Histogram of Variance Plastic Values",breaks = 20, xlab = "Variance", col = "red")

mean_values <- apply(df_lin_paper, 2, mean)
variance_values <- apply(df_lin_paper, 2, var)

hist(mean_values, main = "Histogram of Mean Paper Values",breaks = 20, xlab = "Mean", col = "blue")
hist(variance_values, main = "Histogram of Variance Paper Values",breaks = 20, xlab = "Variance", col = "red")

### functions for FPCA
convert_to_vector <- function(col) {
  if (any(is.na(col))) {
    # Return NULL if the column contains NA values
    return(NULL)
  } else {
    # Convert the column to a vector
    return(as.vector(col))
  }
}

my_acf <- function(col){
  m <- acf(na.omit(col^2), plot= FALSE, lag.max = 30)

  m$acf[2:30]
}

#IN ACF SPACE

FPCAobj <- FPCA(Ly = lapply(df_lin[,2:ncol(df_lin)], my_acf),
                Lt = replicate(ncol(df_lin)-1 , 2:18, simplify = FALSE))
plot(FPCAobj)

FPCAobj_paper <- FPCA(Ly = lapply(df_lin_paper, my_acf),
                Lt = replicate(ncol(df_lin_paper) , 2:30, simplify = FALSE))
plot(FPCAobj_paper)

FPCAobj_plastic <- FPCA(Ly = lapply(df_lin_plastic, my_acf),
                Lt = replicate(ncol(df_lin_plastic) , 2:30, simplify = FALSE))
plot(FPCAobj_plastic)

#Get representation has function values

plastic_predict <- predict(object = FPCAobj_plastic,
                           newLy = lapply(df_lin_plastic, my_acf),
                           newLt = lapply(df_lin_plastic, function(x){2:18}),
                           xiMethod = 'IN')

paper_predict <- predict(object = FPCAobj_paper,
                         newLy = lapply(df_lin_paper, my_acf),
                         newLt = lapply(df_lin_paper, function(x){2:18}),
                         xiMethod = 'IN')

#Just a check
plot(my_acf(df_lin_plastic[[21]]), col='blue', type='l')
lines(plastic_predict[["predCurves"]][21,])

plot(plastic_predict$"scores"[,1], rep(2,length(plastic_predict$scores[,2])))

plot(plastic_predict$"scores"[,2], rep(2,length(plastic_predict$scores[,2])))

plot(paper_predict$"scores"[,1], rep(2,length(paper_predict$scores[,2])))

plot(paper_predict$"scores"[,2], rep(2,length(paper_predict$scores[,2])))

plot(1:length(plastic_predict$scores[,3]), plastic_predict$"scores"[,3])

plot(1:length(plastic_predict$scores[,4]), plastic_predict$"scores"[,4])

plot(1:length(plastic_predict$scores[,5]), plastic_predict$"scores"[,5])


#apply dbscan to what we want

emobj <- simple.init(plastic_predict$scores, nclass = 10)
mixture <- shortemcluster(plastic_predict$scores, emobj)

kNNdistplot(plastic_predict$score, minPts = 4)










#SPLIT VECTOR BASIS

#all_dates <- seq(as.Date("2021-01-01"), as.Date("2023-12-31"), by = "day")
#df_lin <- full_join(data.frame(Date = all_dates), df_lin, by = "Date")

df_lin$years <- year(df_lin$Date)
df_lin_plastic$period <- cut(df_lin_plastic$Date, breaks = seq(as.Date("2021-01-14"),
                by = "4 week", length.out = nrow(df_lin_plastic)), include.lowest = TRUE)

df_lin_paper$period <- cut(df_lin_paper$Date, breaks = seq(as.Date("2021-01-14"),
                by = "4 week", length.out = nrow(df_lin_paper)), include.lowest = TRUE)

df_lin_plastic$period <- format(as.Date(df_lin_plastic$period), "%Y-%m-%d")
df_lin_paper$period <- format(as.Date(df_lin_paper$period), "%Y-%m-%d")

ldfs_plastic <- split(df_lin_plastic, df_lin_plastic$period)
ldfs_paper <- split(df_lin_paper, df_lin_paper$period)

ldfs_plastic <- lapply(ldfs_plastic, function(df) df[, !(names(df) %in% c("years", "Date", "period"))])
ldfs_paper <- lapply(ldfs_paper, function(df) df[, !(names(df) %in% c("years", "Date", "period"))])

##To save for later
bins_fpc_plastic <- lapply(names(ldfs_plastic[[1]]), function(col_name) {
  lapply(ldfs_plastic, function(df) as.vector(df[[col_name]]))
})
names(bins_fpc_plastic) <- names(ldfs_plastic[[1]])

##To save for later
bins_fpc_paper <- lapply(names(ldfs_paper[[1]]), function(col_name) {
  lapply(ldfs_paper, function(df) as.vector(df[[col_name]]))
})
names(bins_fpc_paper) <- names(ldfs_paper[[1]])


yy_plastic <- Reduce(c, lapply(ldfs_plastic, function(x) {lapply(x, as.vector)}))
yy_plastic <- yy_plastic[sapply(yy_plastic, function(x) !any(is.na(x)) && length(x) == 28 )]
#yy_plastic <- c(yy_plastic, list(rep(9, 15)))

yy_paper <- Reduce(c, lapply(ldfs_paper, function(x) {lapply(x, as.vector)}))
yy_paper <- yy_paper[sapply(yy_paper, function(x) !any(is.na(x)) && length(x) == 28 )]
#yy_paper <- c(yy_paper, list(rep(9, 15)))

ldfs_plastic <- NULL
ldfs_paper <- NULL

FPCAobj_plastic <- FPCA(Ly = yy_plastic,
                Lt = lapply(yy_plastic, function(x) {1:length(x)}))
FPCAobj_paper <- FPCA(Ly = yy_paper,
                Lt = lapply(yy_paper, function(x) {1:length(x)}))

plot(FPCAobj_plastic)
plot(FPCAobj_paper)


###Get the eigenvalue score sequence


fpc_temp_plastic <- lapply(bins_fpc_plastic, function(ts) {
  mask <- sapply(ts, function(x) !any(is.na(x)) && length(x) == 28)
  ts_temp <- ts[mask]
  ts_temp <-predict(
    object = FPCAobj_plastic,
    newLy = ts_temp,
    newLt = lapply(ts_temp, function(x) 1:28),
  )
})

fpc_temp_paper <- lapply(bins_fpc_paper, function(ts) {
  mask <- sapply(ts, function(x) !any(is.na(x)) && length(x) == 28)
  ts_temp <- ts[mask]
  ts_temp <-predict(
    object = FPCAobj_paper,
    newLy = ts_temp,
    newLt = lapply(ts_temp, function(x) 1:28),
  )
})

to_pad_matrix <- function(x,i){
  matrix_score <- lapply(x, function(x) x[["scores"]][,i])
  max_length = max(sapply(matrix_score,length))

  matrix_score <- lapply(matrix_score, function(x){
    length(x)<-max_length
    x <- as.vector(x)
    return(x)
  })

  return(matrix_score)
}

convolutional_l2_distance <- function(vector1, vector2) {
  smaller_vector <- ifelse(length(vector1) < length(vector2), vector1, vector2)
  bigger_vector <- ifelse(length(vector1) < length(vector2), vector2, vector1)

  min_distance <- -1

  for (i in 0:(length(bigger_vector) - length(smaller_vector))) {
    cut_bigger_vector <- bigger_vector[i + 1:length(smaller_vector)]
    distance <- sqrt(sum((smaller_vector - cut_bigger_vector)^2))

    if(min_distance < 0){min_distance <- distance}
    else{ if(distance < min_distance){min_distance <- distance}}
  }

  return(min_distance)
}

get_matrix_dist <- function(vectors){
  n <- length(vectors)
  dist_matrix <- matrix(0, n, n)

  for (i in 1:n) {
    for (j in 1:n) {
      dist_matrix[i, j] <- convolutional_l2_distance(vectors[[i]], vectors[[j]])
    }
  }

  return(as.dist(dist_matrix))
}

score_list_plastic <- lapply(1:6, function(i) to_pad_matrix(fpc_temp_plastic,i))
score_list_paper <- lapply(1:5, function(i) to_pad_matrix(fpc_temp_paper,i))

mydtw_plastic <- proxy::dist(lapply(fpc_temp_paper, function(x) x[["scores"]][,1]), method = "SBD")
mydtw2_plastic <- proxy::dist(lapply(fpc_temp_paper, function(x) x[["scores"]][,2]), method = "SBD")
mydtw3_plastic <- proxy::dist(lapply(fpc_temp_paper, function(x) x[["scores"]][,3]), method = "SBD")
mydtw4_plastic <- proxy::dist(lapply(fpc_temp_paper, function(x) x[["scores"]][,4]), method = "SBD")

mydtw_paper <- proxy::dist(lapply(fpc_temp_paper, function(x) x[["scores"]][,1]), method = "SBD")


m <- lapply(fpc_temp_paper, function(x) mean(x[["scores"]][,1]))
m2 <- lapply(fpc_temp_paper, function(x) mean(x[["scores"]][,2]))

hist(Reduce(c, m))

my_d_plastic <- get_matrix_dist(lapply(fpc_temp_plastic, function(x) x[["scores"]][,1]))
my_d_paper <- get_matrix_dist(lapply(fpc_temp_paper, function(x) x[["scores"]][,1]))

par(mfrow=c(1,1))
kNNdistplot(my_d_plastic, minPts = 10)
kNNdistplot(my_d_paper, minPts = 10)
grid()

db_plastic <- dbscan(my_d_plastic, minPts = 10, eps=1.8)
db_paper <- dbscan(my_d_paper, minPts = 10, eps=1)

db_plastic
db_paper

cluster_plastic <- db_plastic$cluster
cluster_paper <- db_paper$cluster
fit <- cmdscale(my_d_plastic, eig=FALSE, k=2)
fit <- cmdscale(my_d_paper, eig=FALSE, k=2)

x <- fit[,1]
y <- fit[,2]

colors <- c("red", "blue", "black")
cluster_labels <- c("Noise", "Cluster 1", "Cluster 2")
point_colors <- colors[cluster_plastic + 1]

plot(x, y, col = point_colors, pch=16, main="Score 1 plastic with l2 distance")
plot(x, y, pch=16, main="Score 1 plastic with l2 distance")




x?proxy::dist




#####EXAMPLE WITH SIMULaTED ARIMA

# Set the number of time series and measures
num_series <- 10
num_measures <- 60

# Create an empty list to store the time series
time_series <- list()

# Generate ARIMA time series for each series
for (i in 1:num_series) {
  # Generate random coefficients
  ar_coefs <- runif(2, -0.5, 0.5)
  ma_coefs <- runif(2, -0.5, 0.5)

  # Generate ARIMA time series
  arima_series <- arima.sim(model = list(ar = ar_coefs, ma = ma_coefs), n = num_measures)

  # Store the time series in the list
  time_series[[i]] <- arima_series

}

# Plot all the time series in the list with legend
plot(x = NULL, y = NULL, xlim = c(1, num_measures), ylim = range(unlist(time_series)), xlab = "Time", ylab = "Value")
for (i in 1:1) {
  lines(time_series[[i]], col = i, lty = i)
}

# Add legend
legend("topright", legend = paste("Series", 1:3), col = 1:3, lty = 1:3)


# Create a list to store the time series with 10 data points each
time_series_10 <- list()

# Iterate over each time series in the original list
for (i in 1:length(time_series)) {
  # Extract the current time series
  current_series <- time_series[[i]]

  # Split the current time series into chunks of 10 data points each
  split_series <- split(current_series, rep(1:3, each = 20, length.out = length(current_series)))

  # Add the split time series to the new list
  time_series_10 <- c(time_series_10, split_series)
}

FPCAobj <- FPCA(Ly = time_series_10, Lt = lapply(time_series_10, function(x) 1:20))
plot(FPCAobj)

pred_list <- list()

pred_list <- lapply(time_series, function(time_serie){
  # Extract the current time series
  current_series <- time_serie

  # Split the current time series into chunks of 10 data points each
  split_series <- split(current_series, rep(1:3, each = 20, length.out = length(current_series)))

  # Add the split time series to the new list
  pred <- predict(
        object = FPCAobj,
        newLy = split_series,
        newLt = lapply(split_series,function(x) {1:20}),
        )

  return(pred[["scores"]])

})

scores_1 <- lapply(pred_list, function(x) x[,1])

# Plot all the time series in the list with legend
plot(x = NULL, y = NULL, xlim = c(1, 5), ylim = range(unlist(scores_1)), xlab = "Time", ylab = "Value")
for (i in 1:3) {
  lines(scores_1[[i]], col = i, lty = i)
}

# Add legend
legend("topright", legend = paste("Series", 1:3), col = 1:3, lty = 1:3)


l2d <- get_matrix_dist(scores_1)
sbd <- proxy::dist(scores_1, method="sbd")

l2d
sbd

lines(scores_1[[8]])

kNNdistplot(l2d, minPts=3)
dbscan(l2d, eps=0.5, minPts=3)

kNNdistplot(sbd, minPts=3)
dbscan(sbd, eps=0.55, minPts=3)

fitl2 <- cmdscale(l2d, k=2)
fitsbd <- cmdscale(sbd, k=2)

x <- fitl2[,1]
y <- fitl2[,2]
plot(x,y)


x <- fitsbd[,1]
y <- fitsbd[,2]
plot(x,y)
