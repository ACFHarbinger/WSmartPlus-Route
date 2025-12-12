setwd("C:/Users/Utilizador/OneDriveUL/Desktop/Masters/Wsmart+Route/Initial_studies")

library(tseries)
library(data.table)
library(forecast)
library(xts)

#### IMPORT DATA HEADER #####

#DATA DAY
df_lin <- read.csv("out_CRUDE_RATE.csv")
names(df_lin)[names(df_lin) == 'X'] <- 'Date'
colnames(df_lin) <- sub("^X", "", colnames(df_lin))
rownames(df_lin) <- as.POSIXct(df_lin$Date, format = "%Y-%m-%d ")

#INFO
df_info <- read.csv("out_INFO.csv", row.names = 'ID')
rownames(df_info) <- as.integer(rownames(df_info))

### SPLITING PAPEL PLASTICO ####

tipo = unique(df_info$Tipo.de.Residuos)

filtered_ids_c <- rownames(df_info)[which(df_info[, "Tipo.de.Residuos"] == tipo[[1]])]
filtered_ids_p <- rownames(df_info)[which(df_info[, "Tipo.de.Residuos"] == tipo[[2]])]

df_lin_p <- df_lin[,filtered_ids_p]
df_lin_c <- df_lin[,filtered_ids_c]

### CLUSTERING ###

library("dtwclust")


acf_fun <- function(series) {
  lapply(series, function(x) {
    d <- as.numeric(acf(na.omit(x), lag.max = 16, plot = FALSE)$acf)
    r <- d[3:9]*mean(x,na.rm=TRUE)
    r[is.na(r)] <- 0
    r
  })
}

cfgs_ <- compare_clusterings_configs(
  types = c("p"),
  k = 4L:8L,
  distances = pdc_configs(
    type = "distance",
    pam = list(
      L2 = list()
    )
  ),
  controls = list(
    partitional = partitional_control(
      iter.max = 30L,
      nrep = 1L
    )
  ),
  centroids = pdc_configs(
    type = "centroid",
    partitional = list(
      pam = list()
    )
  )
)
cfgs <- compare_clusterings_configs(
  types = c("p", "h", "f", "t"),
  k = 4L:8L,
  controls = list(
    partitional = partitional_control(
      iter.max = 100L,
      nrep = 10L
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

score_fun

comp <- compare_clusterings(as.list.data.frame(acf_fun(df_lin[,2:ncol(df_lin)])), types=c("p"),
                    configs = cfgs,
                    score.clus = score_fun,
                    pick.clus = pick_fun,
                    trace=FALSE,
                    return.objects = TRUE)


fc <- tsclust(as.list.data.frame(acf_fun(df_lin[,2:ncol(df_lin)])), type = "partitional",centroid="pam", k = 5, distance = "l2")

fc@centroids

plot(comp$pick$object, series = acf_fun(df_lin[,2:ncol(df_lin)]), type = "series")

names(fc) <- paste0("k_", 2L:20L)


r <- sapply(fc, cvi, type = "internal")

r



normalized_matrix <- t(apply(r, 1, function(x) x / sum(x)))
normalized_matrix
# Plot each row as a line
matplot(t(normalized_matrix), type = "l", xlab = "Columns", ylab = "Normalized Values", main = "Line Plot for Each Row")

# Add legend with column names
legend("topright", legend = colnames(t(r)), col = 1:ncol(t(r)), lty = 1, cex = 0.8)

acf_fun(df_lin[,2:ncol(df_lin)])
