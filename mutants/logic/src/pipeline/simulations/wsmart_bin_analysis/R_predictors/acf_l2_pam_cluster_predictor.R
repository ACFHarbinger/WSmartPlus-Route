

library("dtwclust")
library("forecast")
library("dbscan")

acf_fun <- function(series) {
  lapply(series, function(x) {
    d <- as.numeric(acf(na.omit(x), lag.max = 16, plot = FALSE)$acf)
    r <- d[3:8]*mean(x,na.rm=TRUE)
    r[is.na(r)] <- 0
    r
  })
}

clustering <- function(train, trace){
  cfgs <- compare_clusterings_configs(
    types = c("p", "h", "f", "t"),
    k = 4L:8L,
    controls = list(
      partitional = partitional_control(
        iter.max = 100L,
        nrep = 20L
      ),
      hierarchical = hierarchical_control(
        method = "all"
      ),
      fuzzy = fuzzy_control(
        # notice the vector
        fuzziness = c(2, 2.5),
        iter.max = 30L
      ),
      tadpole = tadpole_control(
        # notice the vectors
        dc = c(1.5, 2),
        window.size = 19L:20L
      )
    ),
    preprocs = pdc_configs(
      type = "preproc",
      # shared
      none = list(),
      zscore = list(center = c(FALSE)),
      # only for fuzzy
      fuzzy = list(
        acf_fun = list()
      ),
      # only for tadpole
      tadpole = list(
        reinterpolate = list(new.length = 205L)
      ),
      # specify which should consider the shared ones
      share.config = c("p", "h")
    ),
    distances = pdc_configs(
      type = "distance",
      L2 = list(),
      fuzzy = list(
        L2 = list()
      ),
      share.config = c("p", "h")
    ),
    centroids = pdc_configs(
      type = "centroid",
      partitional = list(
        pam = list()
      ),
      # special name 'default'
      hierarchical = list(
        default = list()
      ),
      fuzzy = list(
        fcmdd = list()
      ),
      tadpole = list(
        default = list(),
        shape_extraction = list(znorm = TRUE)
      )
    )
  )

  eval <- cvi_evaluators(type="Sil")
  score_fun <- eval$score
  pick_fun <- eval$pick

  comp <- compare_clusterings(as.list.data.frame(acf_fun(train)), types=c("p"),
                              configs = cfgs,
                              score.clus = score_fun,
                              pick.clus = pick_fun,
                              trace=FALSE,
                              return.objects = TRUE
                              )
  if(trace){
    print("CLUSTERS FOUND")
    print(comp$pick)
    plot(comp$pick$object, series = acf_fun(train),type = "series")
  }

  comp$pick$object@cluster
}

avg_by_cluster <- function(df, mask){
  # Initialize a list to store the means for each cluster
  cluster_means_list <- list()

  for (cl in unique(mask)) {

    cluster_columns <- df[, mask == cl]

    if (class(cluster_columns) == 'numeric'){
      cluster_means_list[[cl]] <- cluster_columns
    }else{
      cluster_means_list[[cl]] <- rowMeans(cluster_columns, na.rm = TRUE)
    }
  }

  cluster_means_list
}

acf_l2_pam_cluster <- function(train, test, trace){

  mask <- clustering(train, trace)

  cluster_series <- avg_by_cluster(train, mask)
  test_series <- avg_by_cluster(test,mask)

  #extract cluster models
  cl_models <- lapply(cluster_series, function(serie){
    model <- auto.arima(serie, seasonal = FALSE, allowdrift = FALSE)
    model <- Arima(serie, order = arimaorder(model), seasonal = list(order = c(1,0,1), period=7))
    if(trace){
      print(model)
      print(summary(model))
    }
    model
  })

  train_xreg <- lapply(cl_models, function(model) {model$fitted})
  test_xreg <- list()

  #make a prediction for the cluster
  for (cl in unique(mask)){
   test_xreg[[cl]] <- Arima(test_series[[cl]], model= cl_models[[cl]])$fitted
  }

  predicted <- data.frame(matrix(NaN, nrow = nrow(test), ncol = ncol(test)))

  ##estimate a simple arima with external regressors
  for (i in 1:ncol(train)){
    index_t <- is.na(train[,i])

    if (sum(index_t) < (length(index_t)-30)){

      index_t <- which(!index_t)

      model <- auto.arima(train[index_t,i], xreg= train_xreg[[mask[[i]]]][index_t],seasonal = FALSE, allowdrift = FALSE)
      if(trace){
        print(model)
      }

      index <- is.na(test[,i])

      if (sum(index) < (length(index)-10)){
        index <- which(!index)

        # print("blaaaaaaaaaaaaaaaaa\n\n\n\n\n\n")
        # print(test[index,i])
        # print("\n\n")
        # print(test_xreg[[mask[[i]]]])

        temp <- Arima(test[index ,i], xreg = test_xreg[[mask[[i]]]][index], model=model)

        predicted[index,i] <- temp$fitted
      }
    }
  }

  predicted
}
