library(forecast)
library(data.table)


write_to_file <- function(model, file){

  # Extract model information
  has_model <- length(model$arma) > 1

  if(has_model) {
    p <- length(model$arma[1:model$arma[1]])  # AR order
    q <- length(model$arma[(model$arma[1]+1):(model$arma[1]+model$arma[2])])  # MA order
  } else {
    p <- 0  # AR order
    q <- 0  # MA order
  }


  # Check if seasonal components exist (arma vector length > 4 for seasonal models)
  has_seasonal <- length(model$arma) > 4
  if(has_seasonal) {
    P <- model$arma[3]  # Seasonal AR order
    Q <- model$arma[4]  # Seasonal MA order
    seasonal_period <- model$arma[5]  # Seasonal period
  } else {
    P <- 0
    Q <- 0
    seasonal_period <- NA
  }

  # Get coefficients
  coefs <- coef(model)

  # Initialize parameter vectors with NA
  ar_coefs <- rep(NA, 5)
  ma_coefs <- rep(NA, 5)

  # Fill AR coefficients
  if(p > 0) {
    ar_names <- paste0("ar", 1:p)
    ar_coefs[1:p] <- coefs[ar_names]
  }

  # Fill MA coefficients
  if(q > 0) {
    ma_names <- paste0("ma", 1:q)
    ma_coefs[1:q] <- coefs[ma_names]
  }

  # Get seasonal coefficients
  sar1 <- ifelse("sar1" %in% names(coefs), coefs["sar1"], NA)
  sma1 <- ifelse("sma1" %in% names(coefs), coefs["sma1"], NA)

  # Get mean and sqrt(sigma2)
  model_mean <- ifelse("intercept" %in% names(coefs), coefs["intercept"],
                      ifelse("drift" %in% names(coefs), coefs["drift"], NA))
  sqrt_sigma2 <- sqrt(as.numeric(model$sigma2))

  if(P > 0 || Q > 0) {
    model_name <- paste0("SARIMA(", p, ";0;", q, ")(", P, ";1;", Q, ")_", seasonal_period)
  } else {
    model_name <- paste0("ARIMA(", p, ";0;", q, ")")
  }

  # Append to CSV
  cat(paste(c(model_name, p, q, ar_coefs, ma_coefs, sar1, sma1, model_mean, sqrt_sigma2),
            collapse = ","), "\n",
      file = file, append = TRUE)

}


fit_arima_model <- function(data, xreg) {

  model1 <- auto.arima(
                       data,
                       xreg = xreg,
                       seasonal = FALSE,
                       allowdrift = FALSE)

  model2 <- tryCatch(
    Arima(
          data,
          xreg = xreg,
          order = arimaorder(model1),
          seasonal = list(order = c(1, 0, 1), period = 7)),
    error = function(cond) {
      model1
    }
  )

  if (model2$aic < model1$aic) {
    return(model2)
  } else {
    return(model1)
  }

}

arima_null_model <- function(train_data, test_data) {

  temp <- NULL

  temp$fitted <- rep(mean(train_data),  length(test_data))
  temp$sigma2 <- var(train_data)
  temp$residuals <- abs(temp$fitted - test_data)

  return(temp)
}

## MAIN

args <- commandArgs(trailingOnly = TRUE)

train_name <- args[1]
test_name  <- args[2]
res_name   <- args[3]
pred_error_name <- args[4]
real_error_name <- args[5]
file_name      <- args[6]

if (!file.exists(file_name)) {
    file.create(file_name)
  }
cat("Model,p,q,ar1,ar2,ar3,ar4,ar5,ma1,ma2,ma3,ma4,ma5,sar7,sma7,mean,sqrt_sigma2\n",
    file = file_name)

train <- read.csv(train_name, header = TRUE)
test  <- read.csv(test_name, header = TRUE)

predicted  <- data.frame(matrix(NaN, nrow = nrow(test), ncol = ncol(test)))
fit        <- data.frame(matrix(NaN, nrow = nrow(train), ncol = ncol(train)))
residuals  <- data.frame(matrix(NaN, nrow = nrow(train), ncol = ncol(train)))
pred_error <- data.frame(matrix(NaN, nrow = nrow(test), ncol = ncol(test)))
real_error <- data.frame(matrix(NaN, nrow = nrow(test), ncol = ncol(test)))

# save_fit        <- data.frame(matrix(NaN, nrow = nrow(test) + nrow(train), ncol = ncol(test)))
# save_residuals  <- data.frame(matrix(NaN, nrow = nrow(test) + nrow(train), ncol = ncol(test)))

colnames(train) <- as.integer(sub("^X", "", colnames(train)))

colnames(predicted)  <- colnames(train)
colnames(real_error) <- colnames(train)
colnames(fit)        <- colnames(train)
colnames(residuals)  <- colnames(train)

curr_thresh <- 0
dth <- ncol(train) %/% 20

for (i in 1:ncol(train)) {

  #Get rid of NANs -> which are sequential if they exist

  index_t <- which(!is.na(train[, i]))

  train_data <- train[index_t, i]

  four_day_avg <- shift(
                        x = frollmean(
                                      train_data,
                                      n = 39,
                                      align = "right",
                                      fill = mean(train_data),
                                      algo = "fast"),
                        n = 1,
                        type = "cyclic")

  four_day_avg[0] <- four_day_avg[1]


  if(!all(is.na(train_data))) {
    model <- fit_arima_model(train_data, four_day_avg)
  }
  else{
    model <- NULL
  }

  shortage = nrow(fit) - length(model$fitted)

  fit[ ,i]        <- c(rep(NA, shortage), model$fitted)
  residuals[ ,i]  <- c(rep(NA, shortage), model$residuals)

  index <- which(!is.na(test[, i]))

  test_data <- test[index, i]

  mean_data <- c(tail(train_data, 39), test_data)

  four_day_avg <- shift(
                      x = frollmean(
                                    mean_data,
                                    n = 39,
                                    align = "right",
                                    fill = mean(train_data),
                                    algo = "fast"),
                      n = 1,
                      type = "cyclic")

  if (!is.null(model)) {
    temp <- Arima(test_data, model = model, xreg = tail(four_day_avg, length(test_data)))
  } else {
    temp <- arima_null_model(train_data, test_data)
  }


  if(length(index) > 0){

    predicted[index,  i] <- temp$fitted

    if(!is.null(model) & length(index) > 0) pred_error[index, i] <- rep(sqrt(model$sigma2), length(index))

    real_error[index, i] <- temp$residuals

  }
  if (i >= curr_thresh) {
    print(sprintf("Fit and Predicted % d out of % d bins", i, ncol(train)))
    curr_thresh <- curr_thresh + dth
  }

  if(length(args) == 6){
    write_to_file(model, file_name)
  }
}


write.csv(predicted,  res_name,        row.names = FALSE)
write.csv(pred_error, pred_error_name, row.names = FALSE)
write.csv(real_error, real_error_name, row.names = FALSE)

if(length(args) == 6){
  write.csv(predicted,   "Out_of_sample_2023-2024_Predictions.csv" , quote = FALSE, row.names = FALSE)
  write.csv(real_error,  "Out_of_sample_2023-2024_residuals.csv"   , quote = FALSE, row.names = FALSE)

  write.csv(fit,         "In_sample_2021-2023_Predictions.csv"     , quote = FALSE, row.names = FALSE)
  write.csv(residuals,   "In_sample_2021-2023_Residuals.csv"       , quote = FALSE, row.names = FALSE)
}
