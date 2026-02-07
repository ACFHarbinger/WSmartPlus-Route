

libray("fdapace")

period_parse <- function(string){
  p <- as.numeric(substr(string, 1, 1))
  if(p == 1){
    return(7)
  }
  else if(p==2){
    return(14)
  }
  else if(p==3){
    return(21)
  }
  else if(p==4){
    return(28)
  }
}

get_FPCA <- function(ldfs, period){

  p <- period_parse(period)

  yy <- Reduce(c, lapply(ldfs, function(x) {lapply(x, as.vector)}))
  yy <- yy[sapply(yy, function(x) !any(is.na(x)) && length(x) == p )]

  FPCA(Ly = yy,
      Lt = lapply(yy, function(x) {1:length(x)}))
}


week_split <- function(train, date, period, date_split){

  train$Date <- date

  train$period <- date_split

  ldfs <- split(train, train$period)

  ldfs <- lapply(ldfs, function(l){
    l[["period"]] <- NULL
    l[["Date"]] <- NULL
    l
  })

  ldfs
}

week_reorganise <- function(week_data, period, test){
  p <- period_parse(period)

  ##To save for later
  bins_fpc <- lapply(names(week_data[[1]]), function(col_name) {
    lapply(week_data, function(df) {
      v <- as.vector(df[[col_name]])
      if (length(v) < p){
        if (!test){
          for (i in (length(v)+1):p){
            v[[i]] <- NaN
          }
        }
        else{
          v <- c( rep(NaN, p-length(v)), v)
        }
      }
      else if (length(v) > p){
        v <- v[1:p]
      }
      v
    })
  })

  bins_fpc <- lapply(bins_fpc, function(l) Filter(function(x) !all(is.na(x)), l))

  names(bins_fpc) <- names(week_data[[1]])

  bins_fpc

}

join_tail <- function(week_train, week_test, week, pl){
  nlist <-  names(week_test)

  week_test <- lapply(nlist, function(name){
    start <- week_train[[name]][[week]]
    end <- week_test[[name]][[week]]

    if(sum(c(!is.na(start),!is.na(end))) == pl){
      start[is.na(start)] <- end[!is.na(end)]
      week_test[[name]][[week]] <- start
    }

    week_test[[name]]
  })

  names(week_test) <- nlist

  week_test
}


FPCA_predict <- function(train, test, date, trace, period){

  ##Pre split the date for the weeks to match well

  date <- date[[1]]

  date_split <- cut(date, breaks = seq(date[[1]], by = period,
                    length.out = nrow(train)+nrow(test)), include.lowest = TRUE)

  date_split <- format(as.Date(date_split), "%Y-%m-%d")

  p <- period_parse(period)

  ##get weeked dates for training

  week_data <- week_split(train, date[1:nrow(train)], period, date_split[1:nrow(train)])

  #forecast Functional Principal Components

  FPCA_obj <- get_FPCA(week_data, period)

  if(trace){
    plot(FPCA_obj)
  }

  #re-group data and predict FPC weights

  week_data <- week_reorganise(week_data, period, test=FALSE)

  train_predict <- lapply(week_data, function(ts){
    predict(object = FPCAobj, newLy = ts, newLt = lapply(ts, function(x) 1:p))
  })

  # split test_data by weeks too

  week_test <- week_split(test, date[(nrow(train)+1):(nrow(train)+nrow(test))],
                          period, date_split[(nrow(train)+1):(nrow(train)+nrow(test))])
  week_test <- week_reorganise(week_test, period, test=TRUE)

  week_test <- join_tail(week_data, week_test, date_split[[nrow(train)]], p)

  ## predicted scores for the test

  test_predict <- lapply(week_test, function(ts){
    pred <- NULL
    if(length(ts)>0){
      pred <- predict(object = FPCAobj, newLy = ts, newLt = lapply(ts, function(x) 1:p))
    }
    pred
  })

  # Start Predicting

  predicted <- data.frame(matrix(NaN, nrow = nrow(test), ncol = ncol(test)))

  colnames(predicted) <- colnames(test)

  for (i in 1:ncol(train)){
  #for (i in 3:4){

    # Estimate ARIMA models for the weights

    scores_t <- train_predict[[i]][["scores"]]
    scores_t <- scores_t[,1:ncol(FPCA_obj$phi)]

    if(!is.null(test_predict[[i]])){
      scores_test <- test_predict[[i]][["scores"]]
      scores_test <- scores_test[,1:ncol(FPCA_obj$phi)]

      w_models <- list()

      for (j in 1:ncol(scores_t)){
        w_models[[j]] <- auto.arima(
          scores_t[,j],
          #xreg = cbind(scores_t[,1:(j-1)], scores_t[,(j+1):ncol(scores_t)])
        )}


      # Estimate the arima score for each fpc_score
      pred_score <-  data.frame(matrix(NaN, nrow = nrow(scores_test), ncol = ncol(scores_test)))

      for (j in 1:ncol(scores_test)){

        temp <- Arima(scores_test[,j],
          #xreg = cbind(scores_test[,1:(j-1)], scores_test[,(j+1):ncol(scores_test)]),
          model = w_models[[j]])

        pred_score[,j] <- temp[["fitted"]]
      }

      # Estimate curves using the estimated
      fpcs <- FPCA_obj$phi%*%t(pred_score)

      dim(fpcs) <- NULL

      trajectory <- FPCA_obj$mu + fpcs*sd(train[,i])

      trajectory <- trajectory - (mean(trajectory) - mean(train[,i]))
      predicted[,i] <- trajectory[1:nrow(predicted)]

    }
  }

  return(predicted)
}
