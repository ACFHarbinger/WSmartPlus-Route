
setwd("C:/Users/Utilizador/OneDriveUL/Desktop/Masters/Wsmart+Route/Initial_studies")


library("fdapace")
library("data.table")
library("lubridate")
library("dplyr")
library("forecast")
library("dtwclust")
library("dbscan")
library("matrixStats")


#DATA 4H
df_crude <- read.csv("out_INTERVAL_RATE_12H.csv", header = TRUE)
time <- as.POSIXct(colnames(df_crude), format = "%Y-%m-%d %H")
names <- rownames(df_crude)
df_crude <- transpose(df_crude)
colnames(df_crude) <- names
df_crude$Date <- time

rm(time)
rm(names)

df_crude <- df_crude[order(df_crude$"Date"), ]



df_crude <- read.csv("out_CRUDE_RATE.csv", header = TRUE)
names(df_crude)[names(df_crude) == 'X'] <- 'Date'
colnames(df_crude) <- sub("^X", "", colnames(df_crude))
df_crude$Date <- as.Date(as.POSIXct(df_crude$Date, format = "%Y-%m-%d"))
df_crude <- df_crude[order(df_crude$Date), ]



#INFO
df_info <- read.csv("out_INFO.csv", row.names = 'ID')
rownames(df_info) <- as.integer(rownames(df_info))

df_crude$period <- cut(as.Date(df_crude$"Date"), 
                    breaks = seq(as.Date("2021-01-01"),
                    by = "4 weeks", length.out = nrow(df_crude)), 
                    include.lowest = TRUE)

##Remove first and last days <- RUN EVERYTHING AGAIN 
df_crude <- df_crude[-c(1:12),]
df_crude <- df_crude[1:(nrow(df_crude) - 11), ]

df_crude$period <- format(as.Date(df_crude$period), "%Y-%m-%d %H:")

ldfs <- split(df_crude, df_crude$period)

ldfs <- lapply(ldfs, function(df) df[, !(names(df) %in% c("Date", "period"))])

##To save for later
bins_fpc <- lapply(names(ldfs[[1]]), function(col_name) {
  lapply(ldfs, function(df) as.vector(df[[col_name]]))
})
names(bins_fpc) <- names(ldfs[[1]])

yy <- Reduce(c, lapply(ldfs, function(x) {lapply(x, as.vector)}))
yy <- yy[sapply(yy, function(x) !any(is.na(x)) && length(x) == 28 )]

FPCAobj <- FPCA(Ly = yy, 
                Lt = lapply(yy, function(x) {1:length(x)}))

plot(FPCAobj)

rm(yy)

#Now we predict fpc scores for each bin

fpc_temp <- lapply(bins_fpc, function(ts) {
  mask <- sapply(ts, function(x) !any(is.na(x)) && length(x) == 28)
  ts_temp <- ts[mask]
  ts_temp <-predict(
    object = FPCAobj,
    newLy = ts_temp,
    newLt = lapply(ts_temp, function(x) 1:28),
  )
})

shifted_copies <- function(vector, r) {
  shifted_copies <- numeric(length(vector) * r)
  for (i in 1:length(vector)) {
    for (j in 1:r) {
      shifted_copies[(j-1)*length(vector) + i] <- vector[i] + 7 * (j-1)
    }
  }
  return(shifted_copies)
}

predx(t(fpc_temp[["192"]][["predGrid"]]))

plot(
  shifted_copies(fpc_temp[["192"]][["predGrid"]], 
                 nrow(fpc_temp[["192"]][["predCurves"]])),
  as.vector(t(fpc_temp[["192"]][["predCurves"]])),
  type='l', col = "blue"
  )
ts <- bins_fpc[['192']]
mask <- sapply(ts, function(x) !any(is.na(x)) && length(x) == 7)
ts_temp <- ts[mask]

lines(Reduce(c, ts_temp), col = "darkgrey")

plot(as.vector(t(fpc_temp[["1858"]][["predCurves"]])), type='l', col = "blue")
ts <- bins_fpc[['1858']]
mask <- sapply(ts, function(x) !any(is.na(x)) && length(x) == 7)
ts_temp <- ts[mask]

lines(Reduce(c, ts_temp), col = "darkgrey")

mean(Reduce(c, ts_temp))
mean(as.vector(t(fpc_temp[["1858"]][["predCurves"]])))


##Check for dbscan
kNNdistplot(plastic_predict$score, minPts = 4)


## Create list of series of score 1 for each bin
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

##create average scores for each component
avg_scores <- function(fpc){
  lapply(fpc , function(bin){
    colMeans(bin[["scores"]])
  })
}

compute_argmax <- function(matrix) {
  # Compute argmax for each row
  argmax <- Reduce(c,lapply(matrix, function(row) which.max(row)))
  
  
  # Compute frequencies of argmax values
  frequencies <- tabulate(argmax, nbins=6)
  
  # Return frequencies
  return(frequencies)
}

compute_argmax(fpc_temp[[1]][["scores"]])

arg_max_for_each <- function(fpc){
  Reduce(rbind, lapply(fpc , function(bin){
    compute_argmax(bin$scores)
  }))
}

matrix_max_frequencies <- arg_max_for_each(fpc_temp)

matrix_score <- to_pad_matrix(fpc_temp, 1)
matrix_score_avg <- Reduce(rbind, avg_scores(fpc_temp))

kNNdistplot(matrix_max_frequencies,minPts = 12)
db <- dbscan(matrix_max_frequencies, eps=30 ,minPts = 12)

clusters <- db$"cluster"

data <- lapply(matrix_score_avg, function(x) x[[1]])

plot(rep(2,length(data)), data)

mscore <- Reduce(rbind, matrix_score)


mydtw <- proxy::dist(lapply(fpc_temp, function(x) x[["scores"]][,1]), method = "SBD")

kNNdistplot(mydtw, minPts = 10)

db <- dbscan(mydtw, minPts = 10, eps=1)

cluster <- db$cluster

## Create the dtw matrix.

library(stats)


fit <- cmdscale(mydtw, eig=FALSE, k=2)

x <- fit[,1]
y <- fit[,2]

colors <- c("red", "blue", "black")
cluster_labels <- c("Noise", "Cluster 1", "Cluster 2")
point_colors <- colors[cluster + 1]

plot(x, y, col = point_colors, pch=16, main="Score 1")

legend("topright", legend = cluster_labels, col = colors, pch = 16)

?knndistplot

