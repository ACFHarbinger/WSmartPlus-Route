
source("/home/unix/SmartBinAnalysis/R_predictors/Arima_predictor_and_simple_ones.R")
source("/home/unix/SmartBinAnalysis/R_predictors/FPCA_Predict.R")
source("/home/unix/SmartBinAnalysis/R_predictors/acf_l2_pam_cluster_predictor.R")

setwd("C:/Users/Utilizador/OneDriveUL/Desktop/Masters/Wsmart+Route/Initial_studies")

### IMPORT AND TREAT DATAFRAMES
cut_matrix <- function(matrix, cutoff) {
  date <- matrix$Date
  index <- which(matrix$Date == cutoff)
  if (length(index) == 0) {
    stop("Cutoff value not found in the matrix.")
  }
  matrix1 <- matrix[1:(index - 1), ]
  matrix2 <- matrix[index:nrow(matrix), ]
  matrix1$Date <- NULL
  matrix2$Date <- NULL
  res <- list(matrix1, matrix2, date)
  names(res) <- c("Train", "Test", "date")
  return(res)
}

split_type <- function(matrix, info_matrix, types)
{
  filtered_ids <- lapply(types, function(type)
  {
    rownames(info_matrix)[which(info_matrix[, "Tipo.de.Residuos"] == type)]
  })
  l <- lapply(filtered_ids, function(ids) {
    m <- matrix[, ids]
    m$Date <- matrix$Date
    m
  })

  names(l) <- types

  return(l)
}

#DATA DAY
df_lin <- read.csv("out_CRUDE_RATE.csv", header = TRUE)
names(df_lin)[names(df_lin) == 'X'] <- "Date"
colnames(df_lin) <- sub("^X", "", colnames(df_lin))
df_lin$Date <- as.Date(as.POSIXct(df_lin$Date, format = "%Y-%m-%d"))
df_lin <- df_lin[order(df_lin$Date), ]

##Remove first and last days <- RUN EVERYTHING AGAIN
df_lin <- df_lin[-c(1:12),]
df_lin <- df_lin[1:(nrow(df_lin) - 11), ]

df_info <- read.csv("out_INFO.csv", row.names = 'ID')
rownames(df_info) <- as.integer(rownames(df_info))

unique(df_info$Tipo.de.Residuos)

dfs <- split_type(df_lin, df_info, unique(df_info$Tipo.de.Residuos))

df_lin <- NULL

dfs <- lapply(dfs, function(x) {cut_matrix( x, as.Date(as.POSIXct("2023-02-01", format = "%Y-%m-%d")))})


### SIMPLE PREDICTORS

predictions_plastic <- list()
predictions_paper <- list()

#Naive Predictor
predictions_plastic[["naive"]] <- est_naive(dfs[[1]]["Train"][[1]], dfs[[1]]["Test"][[1]])
predictions_paper[["naive"]] <- est_naive(dfs[[2]]["Train"][[1]], dfs[[2]]["Test"][[1]])

#Allways the mean
predictions_plastic[["mean"]] <- est_mean(dfs[[1]]["Train"][[1]], dfs[[1]]["Test"][[1]])
predictions_paper[["mean"]] <- est_mean(dfs[[2]]["Train"][[1]], dfs[[2]]["Test"][[1]])

#40 day past mean
predictions_plastic[["40mean"]] <- est_40_day_mean(dfs[[1]]["Train"][[1]], dfs[[1]]["Test"][[1]])
predictions_paper[["40mean"]] <- est_40_day_mean(dfs[[2]]["Train"][[1]], dfs[[2]]["Test"][[1]])

#auto.arima prediction (non-seasonal)
predictions_plastic[["arima_ns"]] <- est_indiv_arima(dfs[[1]]["Train"][[1]], dfs[[1]]["Test"][[1]])
predictions_paper[["arima_ns"]] <- est_indiv_arima(dfs[[2]]["Train"][[1]], dfs[[2]]["Test"][[1]])

#auto.arima prediction wiht moving average as regressor
predictions_plastic[["arima_w"]] <- est_indiv_arima_w(dfs[[1]]["Train"][[1]], dfs[[1]]["Test"][[1]])
predictions_paper[["arima_w"]] <- est_indiv_arima_w(dfs[[2]]["Train"][[1]], dfs[[2]]["Test"][[1]])

#auto.arima with 7 day seasonality
predictions_plastic[["arima_s_7"]] <- est_indiv_arima_s(dfs[[1]]["Train"][[1]], dfs[[1]]["Test"][[1]])
predictions_paper[["arima_s_7"]] <- est_indiv_arima_s(dfs[[2]]["Train"][[1]], dfs[[2]]["Test"][[1]])

#predictor follow up -> ouput a matrix of predicitons using dfs as input. Use only test set data

### MORE DIfICULT ONES

#l2 pam cluster using arima as xreg

# ARIMA of the average of the cluster
predictions[["xreg_l2_pam"]] <- acf_l2_pam_cluster(dfs[[1]]["Train"][[1]], dfs[[1]]["Test"][[1]], trace=FALSE)

predictions_plastic[["fpc_simple"]] <- FPCA_predict(dfs[[1]]["Train"][[1]], dfs[[1]]["Test"][[1]], dfs[[1]]["date"], "2 week", trace=TRUE)
predictions_plastic[["fpc_simple"]] <- NULL
# Vectorized ARIMA to extract rules from fuzzy clusters (shape based distance) -> use average
# Vectorized ARIMA to extract rules from fuzzy clusters -> use centroid
# Vectorized ARIMA to extract rules from weekly fpc clusters -> use centroid

# dbscan localização

# Vectorized ARIMA based on FPC scores


### STATISTICS -> MSE histogram and grphic of a predictor;
                 # Average and variance MSE of the predictor
                 # Winner for each bin;
                 # Ordering of least error for each bin
                 # Average weekly error

get_MSE <- function(pred, real){
  MSE <- matrix(NaN, 1, ncol = ncol(pred))

  names(MSE) <- colnames(pred)

  for (i in 1:ncol(pred)){
    index <- !is.na(pred[,i]) & !is.na(real[,i])

    MSE[[i]] <- sqrt(mean( (pred[index,i] - real[index,i])^2))
  }
  MSE
}

##GET MEAN OF PREDICTOR
mean_of_each <- function(pred){
  lapply(pred, function(x){mean(x, na.rm=TRUE)})
}

##GET ORDERED PLOT FOR EACH PREDICTOR

##GET HISTOGRAM OF PREDICTOR
hist_predictors <- function(pred){
  # Plot histograms
  base_colors <- c("steelblue", "indianred", "seagreen", "darkorange", "mediumpurple",
                   "sienna", "slateblue", "tomato", "goldenrod")

  trans_colors <- sapply(base_colors, function(color) {
    rgb(col2rgb(color)[[1]], col2rgb(color)[[2]], col2rgb(color)[[3]], alpha = 0.5, maxColorValue = 255)
  })

  num_bins <- 20
  hist(pred[[1]], breaks = num_bins, col = base_colors[[1]], xlim = range(1:60), ylim = c(0, 70), main = "Histograms of Predictors", xlab = "Value", ylab = "Frequency")

  for (i in 2:length(pred)) {
    hist(pred[[i]], breaks = num_bins, col = base_colors[[i]], add = TRUE)
  }

  legend("topright", legend=names(pred), fill=base_colors[1:length(pred)])
}

order_line_predictor <- function(pred){
  base_colors <- c("steelblue", "indianred", "seagreen", "darkorange", "mediumpurple",
                   "sienna", "slateblue", "tomato", "goldenrod")

  y <- sort(pred[[1]])
  plot(1:length(y), y, type ="l", ylim = range(1:55), col= base_colors[[1]], main = "Histograms of Predictors", xlab = "order", ylab = "Error")

  for (i in 2:length(pred)) {
    y<-sort(pred[[i]])
    lines(1:length(y),y , col = base_colors[[i]])
  }


  legend("topleft", legend=names(pred), fill=base_colors[1:length(pred)])
}

predictor_comp <- function(pred, kind){
  base_colors <- c("steelblue", "indianred", "seagreen", "darkorange", "mediumpurple",
                                        "sienna", "slateblue", "tomato", "goldenrod")
  y <- pred[[1]]
  plot(1:length(y), y, type ="l", ylim = range(1:55), col= base_colors[[1]], main = paste(kind, "Error for Each Bin"), xlab = "bin index", ylab = "MSE")

  for (i in 2:length(pred)) {
    y<-pred[[i]]
    lines(1:length(y),y , col = base_colors[[i]])
  }
  grid()

  legend("topleft", legend=names(pred), fill=base_colors[1:length(pred)])
}

best_predictor <- function(df, kind){

  base_colors <- c("steelblue", "indianred", "seagreen", "darkorange", "mediumpurple",
                   "sienna", "slateblue", "tomato", "goldenrod")

  best_predictor_indices <- apply(df[, -1], 2, which.min)

  best_predictors <- rownames(df)[unlist(best_predictor_indices)]

  counts <- table(best_predictors)

  barplot(counts, col= base_colors[1:length(counts)], main = paste(kind,"Best Predictor Frequency"))
  grid()

  ppp <- award_points(df)

  barplot(ppp, col= base_colors[1:length(counts)], main = paste(kind, "Formula 1 points\n Rule: 25, 18, 15, 12, 10, 8, 6, 4, 2, 1"))
  grid()
}

award_points <- function(df) {

  points <- c(25, 18, 15, 12, 10, 8, 6, 4, 2, 1)

  a_points <- rep(0, nrow(df))

  names(a_points) <- rownames(df)

  ac <- 0
  for (j in 1:ncol(df)){
    event <- df[,j]

    names(event) <- rownames(df)
    sorted <- sort(event, decreasing = FALSE)

    for (i in 1:length(a_points)){
      idx <- which(names(sorted) == names(a_points)[[i]])

      if(length(idx) >0){
        a_points[[i]] <- a_points[[i]] + points[[idx]]
      }
      else{
        ac <- ac + 1/nrow(df)
      }
    }
  }

  if (length(a_points) <= length(points)){
    a_points <- a_points - points[[length(a_points)]]*(ncol(df)-ac)
  }

  a_points
}

pred_mse_plastic <- lapply(predictions_plastic, function(pred){ get_MSE(pred , dfs[[1]]["Test"][[1]])})
pred_mse_paper <- lapply(predictions_paper, function(pred){ get_MSE(pred , dfs[[2]]["Test"][[1]])})

df_mse_plastic <- data.frame(Reduce(rbind, pred_mse_plastic))
df_mse_paper <- data.frame(Reduce(rbind, pred_mse_paper))
row.names(df_mse_plastic) <- names(pred_mse_plastic)
row.names(df_mse_paper) <- names(pred_mse_paper)
colnames(df_mse_plastic) <- names(pred_mse_plastic[[1]])
colnames(df_mse_paper) <- names(pred_mse_paper[[1]])

mean_of_each(pred_mse_plastic)
mean_of_each(pred_mse_paper)
predictor_comp(pred_mse_plastic, "Plastic")
predictor_comp(pred_mse_paper, "Paper")
best_predictor(df_mse_plastic, "Plastic")
best_predictor(df_mse_paper, "Paper")

#order_line_predictor(pred_mse)
