## ######################################################################### ##
## data preparation
## ------------------------------------------------------------------------- ##
## for data science assignment "employee attrition"
## ######################################################################### ##

# rm(list = ls())

## ========================================================================= ##
## source other script parts
## ========================================================================= ##

source("./r-scripts/02-data-prep.R")

## ========================================================================= ##
## load additional packages
## ========================================================================= ##

library(mlr)
library(tictoc)

## ========================================================================= ##
## define additional global variables
## ========================================================================= ##

## parallel computing:
mlr::configureMlr(on.learner.error = "warn")
n_cpus <- 3

## ========================================================================= ##
## ML models ####
## ========================================================================= ##

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## define features and target
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## select target variable:
varnames_target <- "Attrition"

## select features:
varnames_features <- setdiff(
  names(dat_all),
  varnames_target)

## combine:
varnames_model <- union(varnames_target, varnames_features)

## select subset of data:
dat_model <- dat_train[varnames_model]

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## create model matrix (one-hot-encoding)
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## function to do one-hot encoding of categorical variables:
## (based on global variables only for exploration)
create_mm_data <- function(dat, interaction = NA) {
  ## define formulas (model matrix / design matrix):
  if (is.na(interaction)) {
    formula_this <- paste0(
      varnames_target, " ~ ",
      paste(varnames_features, collapse = " + ")
    )
  } 
  else {
    formula_this <- paste0(
      varnames_target, " ~ ", 
      "(", 
      paste(varnames_features, collapse = " + "),
      ") ^ ", interaction
    )
  }

  ## make model frame with interactions for regression type models:
  dat_model_mm <- model.matrix(
    as.formula(formula_this), 
    dat) %>% as.data.frame()
  ## add target variable:
  dat_model_mm[varnames_target] <- dat[varnames_target]
  ## sanitize names:
  names(dat_model_mm) <- names(dat_model_mm) %>% stringr::str_replace_all(":", "_x_") %>%
    make.names() %>%
    stringr::str_replace_all("\\.+", "_")
  
  return(dat_model_mm)
}

dat_model_mm <- create_mm_data(dat_model)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## define tasks
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## create a task: (= data + meta-information)
task_attrition_basic_mm <- makeClassifTask(id = "task_attrition_basic",
                                        data = dat_model_mm,
                                        target = varnames_target
                                        #fixup.data = "no", check.data = FALSE, ## for createDummyFeatures to work.
)


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## create tuned single learners with random parameter search and CV
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## (models to be refitted later on training data with tuned parameters)

## enable parallel execution
library(parallelMap)
parallelGetRegisteredLevels()
parallelStartMulticore(cpus = n_cpus, level = "mlr.resample")

## set random seed, also valid for parallel execution:
set.seed(4271, "L'Ecuyer")

## choose resampling strategy for parameter tuning:
rdesc <- makeResampleDesc(predict = "both", 
                          method = "CV", iters = 6) #10)
                          #method = "RepCV", reps = 1, folds = 10)


## parameters for parameter tuning:
n_maxit <- 10  ## use more in final estimation.
ctrl <- makeTuneControlRandom(maxit = n_maxit)  
tune_measures <- list(auc, f1, acc, mmce, timetrain, timepredict)


## classif.logreg (later when refitting)

## classif.glmnet
tic("time: classif.glmnet")
tune_results_glmnet <- tuneParams(
  makeLearner("classif.glmnet", predict.type = "prob"),
  task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeNumericParam("alpha", lower = 0, upper = 1),
    makeIntegerParam("s", lower = 0, upper = 1000),
    makeIntegerParam("nlambda", lower = 50, upper = 500),
    makeLogicalParam("standardize")
  )
)
toc()
getParamSet("classif.glmnet")


## faster random forest implementation:
tic("time: tuning ranger")
tune_results_ranger <- tuneParams(
  makeLearner("classif.ranger", predict.type = "prob"),
  task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("mtry", lower = 2, upper = length(varnames_features)),
    makeIntegerParam("min.node.size", lower = 20, upper = 100),
    makeIntegerParam("num.trees", lower = 100, upper = 1000)
  )
)
toc()

## classif.glmboost
tic("time: tuning classif.glmboost")
tune_results_glmboost <- tuneParams(
  makeLearner("classif.glmboost", predict.type = "prob"),
  task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("mstop", lower = 50, upper = 1000),
    makeNumericParam("nu", lower = .01, upper = .8),
    makeLogicalParam("center")
  )
)
toc()
getParamSet("classif.glmboost")

## gradient boosting using xgboost:
tic("time: tuning xgboost")
tune_results_xgboost <- tuneParams(
  makeLearner("classif.xgboost", predict.type = "prob"),
  task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("max_depth", lower = 1, upper = 6),
    makeIntegerParam("nrounds", lower = 100, upper = 1000)
  )
)
toc()

## adaboost:
tic("time: ada")
tune_results_ada <- tuneParams(
  makeLearner("classif.ada", predict.type = "prob"),
  task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeDiscreteParam("type", c("discrete", "real", "gentle")),
    makeIntegerParam("iter", lower = 40, upper = 200),
    makeNumericParam("nu", lower = 0.1, upper = 1),
    makeNumericParam("bag.frac", lower = .2, upper = .8),
    makeLogicalParam("model.coef"),
    makeIntegerParam("max.iter", lower = 20, upper = 100),
    makeIntegerParam("minbucket", lower = 30, upper = 200),
    makeNumericParam("cp", lower = .005, upper = .1)
    #makeIntegerParam("maxdepth", lower = 30, upper = 100)
  )
)
toc()
getParamSet("classif.ada")

# ## extraTrees:
# ## (much too slow)
# tic("time: extraTrees")
# tune_results_xgboost <- tuneParams(
#   makeLearner("classif.extraTrees", predict.type = "prob"),
#   task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
#   par.set = makeParamSet(
#     #makeIntegerParam("ntree", lower = 100, upper = 1000),
#     makeIntegerParam("mtry", lower = 2, upper = length(varnames_features)),
#     makeIntegerParam("nodesize", lower = 20, upper = 100),
#     makeIntegerParam("numRandomCuts", lower = 1, upper = 10),
#     makeLogicalParam("evenCuts")
#   )
# )
# toc()
# getParamSet("classif.extraTrees")

# # classif.evtree 
# tic("time: evtree")
# tune_results_evtree <- tuneParams(
#   makeLearner("classif.evtree", predict.type = "prob"),
#   task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
#   par.set = makeParamSet(
#     makeIntegerParam("minbucket", lower = 30, upper = 200),
#     makeIntegerParam("maxdepth", lower = 4, upper = 30),
#     makeIntegerParam("niterations", lower = 40, upper = 200),
#     makeIntegerParam("ntrees", lower = 50, upper = 1000)
#   )
# )
# toc()
# getParamSet("classif.evtree")

## dbnDNN:
tic("time: dbnDNN")
tune_results_dbndnn <- tuneParams(
  makeLearner("classif.dbnDNN", predict.type = "prob"),
  task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("hidden", lower = 3, upper = 20),
    makeDiscreteParam("activationfun", c("sigm", "linear", "tanh")),
    makeNumericParam("learningrate", lower = .01, upper = .9),
    makeNumericParam("momentum", lower = .1, upper = .9),
    makeIntegerParam("numepochs", lower = 2, upper = 20),
    makeIntegerParam("batchsize", lower = 64, upper = 256),
    makeNumericParam("hidden_dropout", lower = .2, upper = .6),
    makeNumericParam("visible_dropout", lower = .2, upper = .6)
  )
)
toc()
getParamSet("classif.dbnDNN")

# classif.featureless 
tic("time: featureless")
tune_results_featureless <- tuneParams(
  makeLearner("classif.featureless", predict.type = "prob"),
  task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeDiscreteParam("method", c("majority", "sample-prior"))
  )
)
toc()
getParamSet("classif.featureless")

# ## support vector machine
# ## (too slow)
# tic("time: tuning svm")
# tune_results_svm <- tuneParams(
#   makeLearner("classif.svm", predict.type = "prob"),
#   task = task_attrition_basic, resampling = rdesc, measures = tune_measures, control = ctrl,
#   par.set = makeParamSet(
#     makeDiscreteParam("kernel", c("linear", "polynomial", "radial")),
#     #makeIntegerParam("degree", lower = 3, upper = 3),
#     makeNumericParam("gamma", lower = 0.1, upper = 10)
#   )
# )
# toc()
# getParamSet("regr.svm")


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## refit all learners with their tuned parameters (with CV)
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## set random seed, also valid for parallel execution:
set.seed(427121, "L'Ecuyer")

lrns_tuned <- list(
  makeLearner("classif.logreg", predict.type = "prob", 
              id = "Logistic\nRegression"),
  makeLearner("classif.glmnet", predict.type = "prob", 
              id = "Elastic Net\nRegression", par.vals = tune_results_glmnet$x),
  makeLearner("classif.ranger", predict.type = "prob", 
              id = "Random Forest", par.vals = tune_results_ranger$x),
  makeLearner("classif.glmboost", predict.type = "prob", 
              id = "Model-Based Boosting\n(glmboost)", par.vals = tune_results_glmboost$x),
  makeLearner("classif.xgboost", predict.type = "prob", 
              id = "Gradient Boosting\n(XGBoost)", par.vals = tune_results_xgboost$x),
  makeLearner("classif.ada", predict.type = "prob", 
              id = "Adaboost", par.vals = tune_results_ada$x),
  makeLearner("classif.dbnDNN", predict.type = "prob", 
              id = "Deep Neural\nNetwork", par.vals = tune_results_dbndnn$x),
  makeLearner("classif.featureless", predict.type = "prob", 
              id = "Featureless\nClassifier", par.vals = tune_results_featureless$x)
)

## create training aggregation measures:
auc.train.mean <- setAggregation(auc, train.mean)
f1.train.mean <- setAggregation(f1, train.mean)
acc.train.mean <- setAggregation(acc, train.mean)
mmce.train.mean <- setAggregation(mmce, train.mean)

n_reps <- 3
n_folds <- 5
## set resampling strategy for benchmarking:
rdesc_bm <- makeResampleDesc(predict = "both", 
                             method = "RepCV", reps = n_reps, folds = n_folds)

## refit tuned models on complete training data:
tic("time: refit tuned models on training data")
bmr_train <- benchmark(
  lrns_tuned, task_attrition_basic_mm, rdesc_bm,
  measures = list(auc, #rmse.train.mean,
                  f1, #mae.train.mean,
                  acc, #rsq.train.mean,
                  timetrain, timepredict)
)
toc()

parallelMap::parallelStop()


bmr_train
plotBMRBoxplots(bmr_train)
plotBMRBoxplots(bmr_train, 
                measure = auc, 
                style = "violin",
                pretty.names = FALSE) +
  aes(fill = learner.id) + geom_point(alpha = .5) +
  labs(
    title = paste0("Area under the ROC curve (AUC) of ", n_reps, "x repeated ",
                   n_folds, "-fold cross validation"),
    subtitle = paste0("with hyperparameters from ", n_maxit, " iterations of random search cross validation"),
    y = "AUC",
    x = ""
  ) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## refit all learners on full training set and evaluate on eval set
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

dat_modeleval <- bind_rows(
  dat_train[varnames_model],
  dat_eval[varnames_model]
)
dat_modeleval_mm <- create_mm_data(dat_modeleval)

idx_train <- 1:nrow(dat_train)
idx_eval <- nrow(dat_train) + (1:nrow(dat_eval))

## create a task: (= data + meta-information)
task_attrition_basic_mm_eval <- makeClassifTask(
  id = "task_attrition_basic_eval",
  data = dat_modeleval_mm,
  target = varnames_target
  #fixup.data = "no", check.data = FALSE, ## for createDummyFeatures to work.
)

## and estimate performance on an identical test set:
rdesc_bmf <- makeFixedHoldoutInstance(train.inds = idx_train,
                                      test.inds = idx_eval,
                                      size = length(c(idx_train, idx_eval)))
rdesc_bmf

## refit models on complete training data, validate on test data:
tic("time: refit models on complete training data, validate on eval data")
bmr_traineval <- benchmark(
  lrns_tuned, task_attrition_basic_mm_eval, rdesc_bmf,
  # measures = list(rmse, mae, rsq)
  measures = list(auc, #rmse.train.mean,
                  f1, #mae.train.mean,
                  acc, #rsq.train.mean,
                  timetrain, timepredict)
)
toc()

plotBMRBoxplots(bmr_traineval)

# parallelMap::parallelStop()



## next steps:
## * (done) re-estimate on complete training set
## * (done) benchmark on validation set
## * try refitting linear models (only) with interactions and feature selection
## * choose best model of each class, quantify performance
## * take best model and inspect it:
##   * variable importance
##   * ice plots or similar
