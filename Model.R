# Libraries ####
library(data.table)
library(xgboost)
library(mltools)
library(Matrix)
library(foreach)
library(iterators)
library(parallel)
library(doParallel)
library(ModelMetrics)

# Miscellaneous ####
options(scipen = 999)
registerDoParallel(16)
p <- c('data.table', 'mltools', 'xgboost', 'Matrix')

# User Defined Functions ####
custom_folds <- function(label) {
  
  cut <- folds(label, stratified = TRUE, seed = 42)
  fld <- list()
  for (i in 1:5) fld[[i]] <- which(cut == i)
  return(list(vector = cut, list = fld))
  
}

cv_best <- function(train, folds, hyper) {
  
  set.seed(42)
  cross <- xgb.cv(params = hyper,
                  data = train,
                  nrounds = 1e6,
                  prediction = TRUE,
                  folds = folds,
                  verbose = FALSE,
                  early_stopping_rounds = 10)
  
  hold <- data.table(predicted = cross$pred, target = getinfo(train, 'label'))
  
  loss <- as.numeric(cross$evaluation_log[cross$best_iteration, 2])
  stop <- cross$best_iteration * 2
  list <- list(train = train)
  
  set.seed(42)
  best <- xgb.train(hyper, train, stop, list, verbose = 0)$evaluation_log
  setnames(best, names(best), c('iter', 'error'))
  best <- min(best[error < loss]$iter)
  
  set.seed(42)
  model <- xgb.train(hyper, train, best)
  
  return(list(holdout = hold, fitted = model))
  
}

inv_logit <- function(x) exp(x) / (1 + exp(x))

# Import Data ####
data(agaricus.train)
data(agaricus.test)

# Identify Elements ####
train.data <- as.matrix(agaricus.train$data)
train.label <- agaricus.train$label
test.data <- as.matrix(agaricus.test$data)
test.label <- agaricus.test$label

# Create Tables ####
train <- data.table(label = train.label, train.data)
test <- data.table(label = test.label, test.data)

# Combine Tables ####
raw <- rbind(train, test)

# Columns ####
cols <- names(raw)
cols <- setdiff(cols, 'label')
cols <- gsub('=.*', '', cols)
cols <- gsub('\\?', '', cols)
cols <- unique(cols)
cols <- sort(cols)

# Cleaning Loop ####
for (i in cols) {
  
  keep <- names(raw)[substring(names(raw), 1, nchar(i)) == i]
  table <- raw[, ..keep]
  for (j in keep) table[[j]] <- ifelse(table[[j]] == 1, j, '')
  table$new <- NA_character_
  table[, new := do.call(paste0, .SD), .SDcols = keep]
  table[, new := substring(new, nchar(i) + 2)]
  table[, new := gsub('=', '', new)]
  table <- table[, .(new)]
  setnames(table, names(table), i)
  raw[, (keep) := NULL]
  raw <- cbind(raw, table)
  
}

# Removal ####
rm(agaricus.train, agaricus.test, table, test, test.data,
   train, train.data, i, j, keep, test.label, train.label)

# Fix Column Names ####
names(raw) <- make.names(names(raw))

# Copy Data ####
data <- copy(raw)

# Pre-Processing ####
data[, bruises := fifelse(bruises == 'bruises', 1, 0)]
data[, gill.attachment := fifelse(gill.attachment == 'attached', 1, 0)]
data[, gill.size := fifelse(gill.size == 'narrow', 1, 0)]
data[, gill.spacing := fifelse(gill.spacing == 'crowded', 1, 0)]

data[, ring.number := fcase(ring.number == 'none', 0,
                            ring.number == 'one', 1,
                            ring.number == 'two', 2)]

data[, stalk.shape := fifelse(stalk.shape == 'enlarging', 1, 0)]
data[, veil.type := NULL]

# Modeling Variables ####
vars <- setdiff(names(data), 'label')

# Categorical to Factor ####
cats <- names(data[, sapply(data, is.character), with = FALSE])
data[, (cats) := lapply(.SD, as.factor), .SDcols = cats]

# Stratified Folds ####
folds <- custom_folds(data$label)
data[, folds := folds$vector]

# Hyperparameters ####
hyper <- list(objective = 'binary:logistic',
              tree_method = 'hist',
              grow_policy = 'lossguide',
              max_depth = 0,
              eta = .01,
              max_leaves = 25,
              subsample = .5,
              colsample_bytree = .5)

# Cross-Validation ####
cross <- foreach(i = 1:5, .combine = rbind, .packages = p) %dopar% {
  
  vars <- vars
  train <- data[folds != i]
  folds <- custom_folds(train$label)$list
  label <- train$label
  train <- Matrix(as.matrix(one_hot(train[, ..vars])), sparse = TRUE)
  train <- xgb.DMatrix(data = train, label = label)
  model <- cv_best(train, folds, hyper)$fitted
  valid <- data[folds == i]
  matrix <- Matrix(as.matrix(one_hot(valid[, ..vars])), sparse = TRUE)
  matrix <- xgb.DMatrix(data = matrix)
  valid[, predicted := predict(model, matrix)]
  valid <- valid[, .(folds, label, predicted)]
  valid
  
}

# CV Fold Error ####
cross[, .(value = brier(label, predicted)), folds]

# Bootstrapped Error #####
error <- foreach(i = 1:1000, .combine = c) %dopar% {
  
  set.seed(i)
  x <- sample(nrow(cross), nrow(cross), replace = TRUE)
  ModelMetrics::brier(cross[-x]$label, cross[-x]$predicted)
  
}
mean(error)
sd(error)

# Target ####
target <- data$label

# Training Matrix ####
matrix <- Matrix(as.matrix(one_hot(data[, ..vars])), sparse = TRUE)
matrix <- xgb.DMatrix(data = matrix, label = target)

# Model Fitting ####
model <- cv_best(matrix, folds$list, hyper)$fitted

# Save ####
save.image('Model.RData')

# SHAP ####
n <- sample(nrow(data), 1)
shap <- data.table(predict(model, slice(matrix, n:n), predcontrib = TRUE))
setnames(shap, names(shap), c(model$feature_names, 'bias'))
score <- data.table(Variable = c('bias', vars))
score[, Value := NA_real_]
score[Variable == 'bias', Value := unique(shap$bias)]
for (i in vars) {
  
  x <- names(shap)[substring(names(shap), 1, nchar(i)) == i]
  y <- shap[, Reduce(`+`, .SD), .SDcols = x]
  score[Variable == i, Value := y]
  
}
score[, Cumulative := cumsum(Value)]
score[, Transformed := inv_logit(Cumulative)]
score[, Difference := Transformed - shift(Transformed)]
score[Variable == 'bias', Difference := Transformed]

predict(model, slice(matrix, n:n))
score[nrow(score), Transformed]

# Shiny ####
lookup <- list()
for (i in cats) {
  
  tbl <- data.table(lookup = data[[i]])
  tbl <- unique(tbl)
  lookup[[i]] <- tbl
  
}
rm(list = setdiff(ls(), c('lookup', 'model', 'cats', 'vars', 'inv_logit')))
save.image('Shiny.RData')
