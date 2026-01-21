

library(forecast)
library(data.table)

##INDIVIDUAL ARIMA without seasonality
est_indiv_arima <- function(train, test){

  predicted <- data.frame(matrix(NaN, nrow = nrow(test), ncol = ncol(test)))

  colnames(predicted) <- colnames(test)

  for (i in 1:ncol(train)){

    index_t <- is.na(train[,i])

    if(sum(index_t) < (length(index_t)-30)){

      index_t <- which(!index_t)
      model <- auto.arima(train[index_t,i], seasonal = FALSE, allowdrift = FALSE)

      index <- is.na(test[,i])

      if (sum(index) < (length(index)-10)){
        index <- which(!index)
        temp <- Arima(test[index,i], model=model)
        predicted[index,i] <- temp$fitted
      }
    }

  }
  predicted
}

est_delivered_arima <- function(train, test)
{
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

  temp$fitted <- rep(mean(train_data),   length(test_data))
  temp$residuals <- rep(var(train_data), length(test_data))

  return(temp)
  }

  predicted  <- data.frame(matrix(NaN, nrow = nrow(test), ncol = ncol(test)))

  for (i in 1:ncol(train)) {

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

    model <- fit_arima_model(train_data, four_day_avg)

    index <- which(!is.na(test[, i]))

    test_data <- test[index, i]

    if (!is.null(model)) {
      temp <- tryCatch(
        Arima(test_data, model = model),
        error = function(cond) {
          arima_null_model(train_data, test_data)
        }
      )
    } else {
      temp <- arima_null_model(train_data, test_data)
    }

  predicted[index,  i] <- temp$fitted

}

}


##INDIVIDUAL ARIMA with seasonality
est_indiv_arima_s <- function(train, test){

  predicted <- data.frame(matrix(NaN, nrow = nrow(test), ncol = ncol(test)))

  colnames(predicted) <- colnames(test)

  for (i in 1:ncol(train)){
    index_t <- is.na(train[,i])

    if (sum(index_t) < (length(index_t)-30)){

      index_t <- which(!index_t)

      model <- auto.arima(train[index_t,i], seasonal = FALSE, allowdrift = FALSE)
      model <- tryCatch(
        Arima(train[index_t,i], order = arimaorder(model), seasonal = list(order = c(1,0,1), period=7)),
        error = function(cond) {
          message(paste("Error bin: ", i))
          model
        }
      )

      index <- is.na(test[,i])

      if (sum(index) < (length(index)-10)){
        index <- which(!index)

        temp <- Arima(test[index ,i], model=model)

        predicted[index,i] <- temp$fitted
      }
    }

  }

  predicted
}

##NAIVE PREDICTOR
est_naive <- function(train, test){
  predicted <- data.frame(matrix(NaN, nrow = nrow(test), ncol = ncol(test)))

  colnames(predicted) <- colnames(test)

  for (i in 1:ncol(test)){
    predicted[, i] <- shift(test[ , i], n=1, fill=NA, type="lag")
    predicted[1, i] <- tail(train[,i],1)
  }
  predicted
}

##JUST THE MEAN
est_mean <- function(train, test){
  predicted <- data.frame(matrix(NaN, nrow = nrow(test), ncol = ncol(test)))

  colnames(predicted) <- colnames(test)

  m <- lapply(train,function(x) mean(na.omit(x)))
  for (i in 1:ncol(test)){
    predicted[,i] <- rep(m[[i]], nrow(test))
  }
  predicted
}

##40 day mean
est_40_day_mean <- function(train, test){
  predicted <- data.frame(matrix(NaN, nrow = nrow(test), ncol = ncol(test)))

  colnames(predicted) <- colnames(test)

  for (i in 1:ncol(test)){
    m <- mean(train[,i])
    data <- test[,i]
    data[is.na(data)] <- m
    data <- c(tail(train[,i], 39), data)

    predicted[,i] <- four_day_avg <- shift(
                                      x = frollmean(
                                          train_data,
                                          n = 39,
                                          align = "right",
                                          fill = mean(train_data),
                                          algo = "fast"
                                      ),
                                      n = 1,
                                      type = "cyclic")
  }
  predicted
}
