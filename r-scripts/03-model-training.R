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
library(scales)

## ========================================================================= ##
## define additional global variables
## ========================================================================= ##

## parallel computing:
mlr::configureMlr(on.learner.error = "warn")
n_cpus <- 3

## ========================================================================= ##
## function definitions
## ========================================================================= ##

#' customized ggsave function to avoid retyping all parameters:
#'
#' @param fname filename to save in \code{path_img} (defined globally)
#' @param ... 
ggsave_cust <- function(fname, p = last_plot(), width = 8, height = 4, dpi = 300, ...) {
  ggsave(filename = file.path(path_img, fname), 
         plot = p,
         width = width, height = height, dpi = dpi, ...)
}

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
dat_model_imbal <- dat_train[varnames_model]

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## SMOTE upsampling
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

library(DMwR)

set.seed(9560)
formula_smote <- paste0(varnames_target, " ~ .") %>% as.formula()
dat_model <- SMOTE(formula_smote, 
                   data  = dat_model_imbal %>% as.data.frame())                         
dim(dat_model_imbal)
dim(dat_model)
dat_model[[varnames_target]] %>% table() %>% prop.table()

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## explore target variable distribution 
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

dat_train[[varnames_target]] %>% table() %>% prop.table() %>% data.frame("dataset" = "train")
dat_eval[[varnames_target]] %>% table() %>% prop.table() %>% data.frame("dataset" = "eval")
dat_test[[varnames_target]] %>% table() %>% prop.table() %>% data.frame("dataset" = "test")

bind_rows(
  data.frame(dat_train, "dataset" = "train", stringsAsFactors = FALSE),
  data.frame(dat_eval, "dataset" = "eval", stringsAsFactors = FALSE),
  data.frame(dat_test, "dataset" = "test", stringsAsFactors = FALSE)
) %>% ggplot(aes(Attrition)) +
  geom_bar() +
  #geom_bar(aes(y = (..count..)/sum(..count..))) +
  facet_wrap(vars(forcats::fct_relevel(dataset, "train", "eval")), nrow = 1) #+
  #scale_y_continuous(labels = percent_format())
ggsave_cust("train-eval-test-no-upsampling.jpg", height = 4, width = 4)

bind_rows(
  data.frame(dat_model, "dataset" = "train", stringsAsFactors = FALSE),
  data.frame(dat_eval, "dataset" = "eval", stringsAsFactors = FALSE),
  data.frame(dat_test, "dataset" = "test", stringsAsFactors = FALSE)
) %>% ggplot(aes(Attrition)) +
  geom_bar() +
  facet_wrap(vars(forcats::fct_relevel(dataset, "train", "eval")), nrow = 1)
ggsave_cust("train-eval-test-smote-upsampling.jpg", height = 4, width = 4)



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
      paste(varnames_features, collapse = " + "),
      " - 1"
    )
  } 
  else {
    formula_this <- paste0(
      varnames_target, " ~ ", 
      "(", 
      paste(varnames_features, collapse = " + "),
      ") ^ ", interaction, " - 1"
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
n_maxit <- 50 ## 10 currently; use more (about 50) in final estimation.
ctrl <- makeTuneControlRandom(maxit = n_maxit)  
tune_measures <- list(mcc, auc, f1, bac, acc, mmce, timetrain, timepredict)


## classif.logreg (later when refitting)

## classif.glmnet
tic("time: classif.glmnet")
tune_results_glmnet <- tuneParams(
  makeLearner("classif.glmnet", predict.type = "prob"),
  task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    #makeIntegerParam("nfolds", lower = 3, upper = 6),
    makeNumericParam("alpha", lower = 0, upper = 1),
    #makeIntegerParam("s", lower = 0, upper = 1000),
    #makeIntegerParam("nlambda", lower = 1, upper = 500),
    #makeNumericParam("lambda", lower = 0, upper = 10),
    makeLogicalParam("standardize", default = TRUE, tunable = FALSE),
    makeLogicalParam("intercept")
  )
)
toc()
# getParamSet("classif.glmnet")
# getParamSet("classif.cvglmnet")

## decision tree:
tic("time: tuning classif.rpart")
tune_results_rpart <- tuneParams(
  makeLearner("classif.rpart", predict.type = "prob"),
  task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("minsplit", lower = 20, upper = 200),
    makeIntegerParam("minbucket", lower = 20, upper = 100),
    makeNumericParam("cp", lower = .01, upper = 1)
  )
)
toc()
# getParamSet("classif.rpart")



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
    makeIntegerParam("mstop", lower = 20, upper = 1000),
    makeNumericParam("nu", lower = .001, upper = .8),
    makeLogicalParam("center")
  )
)
toc()
# getParamSet("classif.glmboost")

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
    makeNumericParam("nu", lower = 0.01, upper = 1),
    makeNumericParam("bag.frac", lower = .2, upper = .8),
    makeLogicalParam("model.coef"),
    makeIntegerParam("max.iter", lower = 20, upper = 100),
    makeIntegerParam("minbucket", lower = 30, upper = 200),
    makeNumericParam("cp", lower = .005, upper = .1)
    #makeIntegerParam("maxdepth", lower = 30, upper = 100)
  )
)
toc()
# getParamSet("classif.ada")

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

## neural net:
tic("time: nnet")
tune_results_nnet <- tuneParams(
  makeLearner("classif.nnet", predict.type = "prob"),
  task = task_attrition_basic_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeIntegerParam("size", lower = 3, upper = 12),
    makeIntegerParam("maxit", lower = 50, upper = 800),
    makeLogicalParam("skip"),
    makeNumericParam("decay", lower = -1, upper = 1)
  )
)
toc()
# getParamSet("classif.nnet")


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
# getParamSet("classif.dbnDNN")

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
# getParamSet("classif.featureless")

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
  makeLearner("classif.logreg", predict.type = "prob"),
  makeLearner("classif.glmnet", predict.type = "prob", par.vals = tune_results_glmnet$x),
  #makeLearner("classif.rpart", predict.type = "prob", par.vals = tune_results_rpart$x),
  makeLearner("classif.ranger", predict.type = "prob", par.vals = tune_results_ranger$x),
  makeLearner("classif.glmboost", predict.type = "prob", par.vals = tune_results_glmboost$x),
  makeLearner("classif.xgboost", predict.type = "prob", par.vals = tune_results_xgboost$x),
  makeLearner("classif.ada", predict.type = "prob", par.vals = tune_results_ada$x),
  makeLearner("classif.nnet", predict.type = "prob", par.vals = tune_results_nnet$x)
  # makeLearner("classif.dbnDNN", predict.type = "prob", 
  #             id = "Deep Neural\nNetwork", par.vals = tune_results_dbndnn$x),
  # makeLearner("classif.featureless", predict.type = "prob", 
  #             id = "Featureless\nClassifier", par.vals = tune_results_featureless$x)
)

## create training aggregation measures:
mcc_train <- setAggregation(mcc, train.mean)
mcc_train[["id"]] <- "mcc_train"
auc_train <- setAggregation(auc, train.mean)
auc_train[["id"]] <- "auc_train"
f1_train <- setAggregation(f1, train.mean)
f1_train[["id"]] <- "f1_train"
bac_train <- setAggregation(bac, train.mean)
bac_train[["id"]] <- "bac_train"
acc_train <- setAggregation(acc, train.mean)
acc_train[["id"]] <- "acc_train"
mmce_train <- setAggregation(mmce, train.mean)
mmce_train[["id"]] <- "mmce_train"

n_reps <- 3
n_folds <- 5 

## set resampling strategy for benchmarking:
rdesc_bm <- makeResampleDesc(predict = "both", 
                             method = "RepCV", reps = n_reps, folds = n_folds)

## refit tuned models on complete training data:
tic("time: refit tuned models on training data")
bmr_train <- benchmark(
  lrns_tuned, task_attrition_basic_mm, rdesc_bm,
  measures = list(mcc_train, mcc,
                  auc_train, auc,
                  f1_train, f1,
                  bac_train, bac,
                  acc_train, acc,
                  timetrain, timepredict)
)
toc()

parallelMap::parallelStop()

## tabluar results:
bmr_train
bmr_train_summary <- print(bmr_train)
bmr_train_summary %>% select(matches("learner|mcc"))
bmr_train_summary %>% select(matches("learner|bac"))
bmr_train_summary %>% select(matches("learner|auc"))
bmr_train_summary %>% select(matches("learner|acc"))

bmr_train_summary %>% select(matches("learner|mcc|bac|auc"))


plotBMRBoxplots(bmr_train)

plotBMRBoxplots_cust <- function(bmr, measure_mlr, measure_name, measure_longname) {
  plotBMRBoxplots(bmr, 
                  measure = measure_mlr, 
                  style = "violin",
                  pretty.names = TRUE) +
    aes(fill = learner.id) + geom_point(alpha = .5) +
    labs(
      title = paste0(measure_longname, " (", measure_name, ") of ", n_reps, "x repeated ",
                     n_folds, "-fold CV"),
      subtitle = paste0("with hyperparameters from ", n_maxit, " iterations of random search cross validation"),
      y = measure_name,
      x = ""
    ) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    theme(
      plot.title = element_text(size = 11),
      plot.subtitle = element_text(size = 9))
}

plotBMRBoxplots_cust(bmr_train, mcc, "MCC", "Matthew's Correlation Coefficient")
ggsave_cust("model-fit-mcc-train-cv.jpg", width = 5, height = 5)
plotBMRBoxplots_cust(bmr_train, auc, "AUC", "Area under the ROC curve")
ggsave_cust("model-fit-auc-train-cv.jpg", width = 5, height = 5)

## ========================================================================= ##
## refit all learners on full training set and evaluate on eval set
## ========================================================================= ##

dat_modeleval <- bind_rows(
  dat_train[varnames_model],
  dat_eval[varnames_model]
)
dat_modeleval_mm <- create_mm_data(dat_modeleval)

idx_train <- 1:nrow(dat_train)
idx_eval <- nrow(dat_train) + (1:nrow(dat_eval))

## set random seed, also valid for parallel execution:
set.seed(427121, "L'Ecuyer")

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
  measures = list(mcc,
                  auc,
                  f1,
                  bac,
                  acc,
                  timetrain, timepredict)
)
toc()

## tabluar results:
bmr_traineval_summary <- print(bmr_traineval)
bmr_traineval_summary %>% select(matches("learner|mcc"))
bmr_traineval_summary %>% select(matches("learner|bac"))
bmr_traineval_summary %>% select(matches("learner|auc"))
bmr_traineval_summary %>% select(matches("learner|acc"))

bmr_traineval_summary_rnd <- bmr_traineval_summary %>% 
  select(matches("learner|mcc|bac|auc|acc"))
bmr_traineval_summary_rnd[2:5] <- round(bmr_traineval_summary_rnd[2:5], 3)
names(bmr_traineval_summary_rnd) <- names(bmr_traineval_summary_rnd) %>% 
  stringr::str_replace("\\.mean", "")
bmr_traineval_summary_rnd

plotBMRBoxplots(bmr_traineval, measure = mcc)
ggsave_cust("model-fit-mcc-eval.jpg", width = 5, height = 5)

plotBMRBoxplots(bmr_traineval, measure = auc)
ggsave_cust("model-fit-auc-eval.jpg", width = 5, height = 5)

# save.image(file = file.path(path_tmp, "03-model-training___dump01.Rdata"))
# load(file = file.path(path_tmp, "03-model-training___dump01.Rdata"))
# load(file = file.path(path_tmp, "03-model-training___dump02a_smote.Rdata"))

## ========================================================================= ##
## refit subset of learners on data with factors
## ========================================================================= ##

## set random seed, also valid for parallel execution:
set.seed(427121, "L'Ecuyer")

## create a task: (= data + meta-information)
task_attrition_basic_fact_eval <- makeClassifTask(
  id = "task_attrition_basic_eval",
  data = dat_modeleval %>% as.data.frame(),
  target = varnames_target
  #fixup.data = "no", check.data = FALSE, ## for createDummyFeatures to work.
)

## and estimate performance on an identical test set:
rdesc_bmf <- makeFixedHoldoutInstance(train.inds = idx_train,
                                      test.inds = idx_eval,
                                      size = length(c(idx_train, idx_eval)))
rdesc_bmf

lrns_tuned_fact <- list(
  makeLearner("classif.logreg", predict.type = "prob"),
  makeLearner("classif.glmnet", predict.type = "prob", par.vals = tune_results_glmnet$x),
  makeLearner("classif.glmboost", predict.type = "prob", par.vals = tune_results_glmboost$x),
  makeLearner("classif.nnet", predict.type = "prob", par.vals = tune_results_nnet$x)
)

## refit factor models on complete training data, validate on test data:
tic("time: refit factor models on complete training data, validate on eval data")
bmr_traineval_fact <- benchmark(
  lrns_tuned_fact, task_attrition_basic_fact_eval, rdesc_bmf,
  measures = list(mcc,
                  auc,
                  f1,
                  bac,
                  acc,
                  timetrain, timepredict)
)
toc()

bmr_traineval_fact

plotBMRBoxplots(bmr_traineval_fact, measure = mcc)
ggsave_cust("model-fit-mcc-eval-factor.jpg", width = 5, height = 5)

plotBMRBoxplots(bmr_traineval_fact, measure = auc)
ggsave_cust("model-fit-auc-eval-factor.jpg", width = 5, height = 5)



## ========================================================================= ##
## inspect best model using iml package
## ========================================================================= ##

library(iml)

## take sample for quicker model exploration:
set.seed(442)
dat_iml <- dat_model_mm # %>% sample_n(500)
set.seed(442)
dat_iml_fact <- dat_model # %>% sample_n(500)

## create a predictor container(s):
create_predictor <- function(classif_name, bmr_obj = bmr_traineval, task = 1, data = dat_iml) {
  ret <- Predictor$new(
    model = getBMRModels(bmr_obj)[[task]][[classif_name]][[1]],
    data = data %>% select(-varnames_target),  y = dat_iml[varnames_target]
  )
  return(ret)
}
create_predictor_fact <- function(classif_name, bmr_obj = bmr_traineval_fact, task = 1, data = dat_iml_fact) {
  create_predictor(classif_name, bmr_obj, task, data)
}
predictor_logreg <- create_predictor("classif.logreg")
predictor_glmnet <- create_predictor("classif.glmnet")
# predictor_ranger <- create_predictor("classif.ranger")
predictor_glmboost <- create_predictor("classif.glmboost")
predictor_xgboost <- create_predictor("classif.xgboost")
predictor_nnet <- create_predictor("classif.nnet")

predictor_logreg_fact <- create_predictor_fact("classif.logreg")
predictor_glmnet_fact <- create_predictor_fact("classif.glmnet")
predictor_glmboost_fact <- create_predictor_fact("classif.glmboost")
predictor_nnet_fact <- create_predictor_fact("classif.nnet")

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## feature importance: main effects
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


n_reps <- 50

## most important features:
# set.seed(44126)
# imp_logreg <- FeatureImp$new(predictor_logreg, loss = "ce")
# plot(imp_logreg)
set.seed(44126)
imp_logreg_fact <- FeatureImp$new(predictor_logreg_fact, loss = "ce", n.repetition = n_reps)
plot(imp_logreg_fact)

# set.seed(44126)
# imp_glmnet <- FeatureImp$new(predictor_glmnet, loss = "ce")
# plot(imp_glmnet)
set.seed(44126)
imp_glmnet_fact <- FeatureImp$new(predictor_glmnet_fact, loss = "ce", n.repetition = n_reps)
plot(imp_glmnet_fact)
ggsave_cust("varimp-glmnet-fact.jpg", width = 4.5, height = 4.5)

# set.seed(44126)
# imp_glmboost <- FeatureImp$new(predictor_glmboost, loss = "ce")
# plot(imp_glmboost)
set.seed(44126)
imp_glmboost_fact <- FeatureImp$new(predictor_glmboost_fact, loss = "ce", n.repetition = n_reps)
plot(imp_glmboost_fact)
ggsave_cust("varimp-glmboost-fact.jpg", width = 4.5, height = 4.5)

# set.seed(44126)
# imp_nnet <- FeatureImp$new(predictor_nnet, loss = "ce")
# plot(imp_nnet)
set.seed(44126)
imp_nnet_fact <- FeatureImp$new(predictor_nnet_fact, loss = "ce", n.repetition = n_reps)
plot(imp_nnet_fact)


# imp_xgboost <- FeatureImp$new(predictor_xgboost, loss = "ce")
# plot(imp_xgboost)
# imp_ranger <- FeatureImp$new(predictor_ranger, loss = "ce")
# plot(imp_ranger)

## extract top-n most important features:
get_imp_topn <- function(imp, n_top = 15) {
  ret <- imp$clone()
  ret$results <- arrange(imp$results, desc(importance)) %>% head(n = n_top)
  return(ret)
}
get_imp_topn(imp_logreg_fact, 10) %>% plot()
get_imp_topn(imp_glmboost_fact, 10) %>% plot()
get_imp_topn(imp_glmnet_fact, 10) %>% plot()


n_intersect <- 9
get_imp_topn(imp_logreg_fact, n_intersect)$results$feature %>% 
  intersect(get_imp_topn(imp_glmboost_fact, n_intersect)$results$feature) %>%
  intersect(get_imp_topn(imp_glmnet_fact, n_intersect)$results$feature)

## overlap between different models of variable importance with 50 reps
## top-5 features: Only OverTime, JobRole
## top-8 features: Only OverTime, JobRole
## top-9 features: OverTime, JobRole, YearsInCurrentRole
## top-10 features: OverTime, JobRole, YearsInCurrentRole, NumCompaniesWorked, StockOptionLevel

imp_glmboost_fact$results %>% arrange(desc(importance.05))

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## feature importance: interactions
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


interact_glmboost <- Interaction$new(predictor_glmboost)
interact_glmboost %>% str()

interact_glmboost_fact <- Interaction$new(predictor_glmboost_fact)
interact_glmboost_fact %>% str()
plot(interact_glmboost_fact)

interact_ranger <- Interaction$new(predictor_ranger)
plot(interact_ranger)
## [[?]][[better plot]]

interact_nnet <- Interaction$new(predictor_nnet)
plot(interact_nnet)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## glmboost coefficients
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## model summary:
model_glmboost_fact <- getBMRModels(bmr_traineval_fact)[[1]][["classif.glmboost"]][[1]] %>% 
  getLearnerModel()
model_glmboost_fact
summary(model_glmboost_fact)

## model summary of logistic regression, for comparision:
getBMRModels(bmr_traineval_fact)[[1]][["classif.logreg"]][[1]] %>% 
  getLearnerModel() %>% summary()
  
## get coefficients:
# model_glmboost_fact$coef() %>% unlist()
coefficients(model_glmboost_fact, off2int = TRUE) %>% 
  round(3) %>% as.matrix()
## seem to be reversed; obviously, seems to predict the first level 
## (which is "no")
levels(dat_model[[varnames_target]])

# ?coefficients.mboost
# methods("coefficients")
# ?coef.glmboost

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## feature effects (with iml)
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

# # ## accumulated local effects (ALE) for specific feature:
# # # (similar to partial dependence plots):
# effs <- FeatureEffect$new(predictor, feature = "Age")
# plot(effs)

## "choose" a standard predictor to be used below:
#predictor <- predictor_ranger
#predictor <- predictor_xgboost
predictor <- predictor_glmboost


## ------------------------------------------------------------------------- ##
## partial dependence plots with ice plots for glmnet_fact
## ------------------------------------------------------------------------- ##

effs <- FeatureEffect$new(predictor_glmnet_fact, feature = "OverTime", method = "pdp+ice")
plot(effs)
ggsave_cust("feat-eff-overtime-glmnet.jpg", width = 4.5, height = 4.5)

effs <- FeatureEffect$new(predictor_glmnet_fact, feature = "JobRole", method = "pdp+ice")
plot(effs) + theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave_cust("feat-eff-jobrole-glmnet.jpg", width = 8, height = 3)

effs <- FeatureEffect$new(predictor_glmnet_fact, feature = "WorkLifeBalance", method = "pdp+ice")
plot(effs)
ggsave_cust("feat-eff-worklifebalance-glmnet.jpg", width = 8, height = 2.5)


StockOptionLevel
JobInvolvement

effs <- FeatureEffect$new(predictor_glmnet_fact, feature = "TotalWorkingYears", method = "pdp+ice")
plot(effs)
ggsave_cust("feat-eff-totalworkingyears-glmnet.jpg", width = 4.5, height = 4.5)

effs <- FeatureEffect$new(predictor_glmnet_fact, feature = "NumCompaniesWorked", method = "pdp+ice")
plot(effs)
ggsave_cust("feat-eff-numcompaniesworked-glmnet.jpg", width = 4.5, height = 4.5)

effs <- FeatureEffect$new(predictor_glmnet_fact, feature = "YearsInCurrentRole", method = "pdp+ice")
plot(effs)
#ggsave_cust("feat-eff--glmnet.jpg", width = 4.5, height = 4.5)

effs <- FeatureEffect$new(predictor_glmnet_fact, feature = "EducationField", method = "pdp+ice")
plot(effs)
#ggsave_cust("feat-eff--glmnet.jpg", width = 4.5, height = 4.5)



## ------------------------------------------------------------------------- ##
## partial dependence plots with ice plots for glmboost_fact
## ------------------------------------------------------------------------- ##

effs <- FeatureEffect$new(predictor_glmboost_fact, feature = "OverTime", method = "pdp+ice")
plot(effs)
ggsave_cust("feat-eff-overtime-glmboost.jpg", width = 4.5, height = 4.5)

effs <- FeatureEffect$new(predictor_glmboost_fact, feature = "JobRole", method = "pdp+ice")
plot(effs) + theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave_cust("feat-eff-jobrole-glmboost.jpg", width = 8, height = 3)

effs <- FeatureEffect$new(predictor_glmboost_fact, feature = "TotalWorkingYears", method = "pdp+ice")
plot(effs)
ggsave_cust("feat-eff-totalworkingyears-glmboost.jpg", width = 4.5, height = 4.5)

effs <- FeatureEffect$new(predictor_glmboost_fact, feature = "NumCompaniesWorked", method = "pdp+ice")
plot(effs)
ggsave_cust("feat-eff-numcompaniesworked-glmboost.jpg", width = 4.5, height = 4.5)

effs <- FeatureEffect$new(predictor_glmboost_fact, feature = "YearsInCurrentRole", method = "pdp+ice")
plot(effs)
#ggsave_cust("feat-eff--glmboost.jpg", width = 4.5, height = 4.5)

effs <- FeatureEffect$new(predictor_glmboost_fact, feature = "EducationField", method = "pdp+ice")
plot(effs)
#ggsave_cust("feat-eff--glmboost.jpg", width = 4.5, height = 4.5)

effs <- FeatureEffect$new(predictor_glmboost_fact, feature = "WorkLifeBalance", method = "pdp+ice")
plot(effs)
ggsave_cust("feat-eff-worklifebalance-glmboost.jpg", width = 8, height = 2.5)

## * Working overtime
## * Job role (but with low confidence)
## * Total working years
## * Number of companies worked for
## * Years in current role
## * Education field
## * Work life balance




## partial dependence plot with ice plot, comparisons:
effs <- FeatureEffect$new(predictor_glmboost, feature = "OverTimeYes", method = "pdp+ice")
plot(effs)

effs <- FeatureEffect$new(predictor_glmboost_fact, feature = "OverTime", method = "pdp+ice")
plot(effs)
# ?plot.FeatureEffect
# iml:::plot.FeatureEffect
# effs$generatePlot

effs <- FeatureEffect$new(predictor, feature = "MonthlyIncome", method = "pdp+ice")
plot(effs)

effs <- FeatureEffect$new(predictor_glmboost_fact, feature = "OverTime", method = "pdp+ice")
plot(effs)
effs <- FeatureEffect$new(predictor_glmboost_fact, feature = "Age", method = "pdp+ice")
plot(effs)

# save.image(file = file.path(path_tmp, "03-model-training___dump02a.Rdata"))


dat_iml %>% names()

## next steps:
## * (done) re-estimate on complete training set
## * (done) benchmark on validation set
## * (done) try refitting linear models (only) with interactions and feature selection
## * (done) try stepwise glm with interactions with forward selection, starting at full model
## * cleanup and move interaction stuff to different file
## * choose best model of each class, quantify performance
## * take best model and inspect it:
##   * variable importance
##   * ice plots or similar
