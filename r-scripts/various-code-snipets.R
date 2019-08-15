## ######################################################################### ##
## various code snippets
## ------------------------------------------------------------------------- ##
## for data science assignment "employee attrition"
## ######################################################################### ##

## ######################################################################### ##
## some mlr learners that weren't used in the analysis ####
## ######################################################################### ##

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


## ######################################################################### ##
## Refit linear ML models with interactions #### 
## ######################################################################### ##

## ========================================================================= ##
## Refit linear ML models with interactions
## ========================================================================= ##

dat_model_interact_mm <- create_mm_data(dat_model, interaction = 2)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## define tasks
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## create a task: (= data + meta-information)
task_attrition_interact_mm <- makeClassifTask(id = "task_attrition_interact",
                                           data = dat_model_interact_mm,
                                           target = varnames_target
                                           #fixup.data = "no", check.data = FALSE, ## for createDummyFeatures to work.
)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## create tuned single learners with random parameter search and CV
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## (models to be refitted later on training data with tuned parameters)

## classif.glmnet
tic("time: classif.glmnet")
tune_results_glmnet_interact <- tuneParams(
  makeLearner("classif.glmnet", predict.type = "prob"),
  task = task_attrition_interact_mm, resampling = rdesc, measures = tune_measures, control = ctrl,
  par.set = makeParamSet(
    makeNumericParam("alpha", lower = 0, upper = 1),
    makeLogicalParam("standardize", default = TRUE, tunable = FALSE),
    makeLogicalParam("intercept")
  )
)
toc()

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## refit with tuned parameters (with CV)
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## set random seed, also valid for parallel execution:
set.seed(427121, "L'Ecuyer")

lrns_tuned_interact <- list(
  makeLearner("classif.glmnet", predict.type = "prob", 
              id = "Elastic Net\nRegression", par.vals = tune_results_glmnet_interact$x)
)

## refit tuned models on complete training data:
tic("time: refit tuned models on training data")
bmr_train_interact <- benchmark(
  lrns_tuned_interact, task_attrition_interact_mm, rdesc_bm,
  measures = list(mcc,
                  auc, #rmse.train.mean,
                  f1, #mae.train.mean,
                  acc, #rsq.train.mean,
                  timetrain, timepredict)
)
toc()

plotBMRBoxplots(bmr_train_interact)
plotBMRBoxplots_cust(bmr_train_interact, mcc, "MCC", "Matthew's Correlation Coefficient")
plotBMRBoxplots_cust(bmr_train_interact, auc, "AUC", "Area under the ROC curve")








## ######################################################################### ##
## Refit linear ML models with interactions using StepAIC #### 
## ######################################################################### ##

## doesn't work; immensley overfitting the training set, resulting model does
## not at all generalize to eval set!!!


## ========================================================================= ##
## Refit linear ML models with more feature selection and interactions ####
## ========================================================================= ##

#varnames_model <- union(varnames_target, varnames_features)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## more feature selection
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

## check variance inflation factor (VIF) in linear model:
formula_glm <- paste0(varnames_target, " ~ .") %>% as.formula()
fit_glm <- glm(formula_glm, family = "binomial", data = dat_model)
summary(fit_glm)
car::vif(fit_glm)

## check high correlations:

## correlations (heterogenous correlation matrix):
cormat_hetcor <- polycor::hetcor(as.data.frame(dat_all), std.err = FALSE, use = "pairwise")
cormat <- cormat_hetcor[["correlations"]]
cormat

cormat_long <- reshape2::melt(cormat, na.rm = TRUE)
cormat_long %>% filter(Var1 != Var2, abs(value) > .8) %>% arrange(value)

## automatically exclude high correlations:
varnames_cor_caret <- caret::findCorrelation(cormat, cutoff = 0.9, names = TRUE)
varnames_cor_caret

## caret identifies Joblevel, and with lower cutoff also JobRole;
## Jobrole is highly correlated with Department, but only Department shows high VIF
## hence, exclude Department and not JobLevel
varnames_cor_exclude <- c("JobLevel", "Department")

## redefine model variables 
## (overwriting existing variable):
varnames_features <- setdiff(varnames_features, varnames_cor_exclude)
varnames_model <- setdiff(varnames_model, varnames_cor_exclude)

## select subset of data:
dat_model <- dat_train[varnames_model]

## (currently unnecessary):
dim(dat_model_mm)
dat_model_mm <- create_mm_data(dat_model, interaction = 2)


## [[todo]] refit linear models using model-matrix with interactions

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## stepwise regression
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

scopedef <- list(
  lower = ~ 1,
  upper = as.formula(
    paste(" ~ ", 
          "(", 
          paste(varnames_features, collapse = " + "),
          ") ^ 2")
  )
)

formula_glm <- paste0(varnames_target, " ~ .") %>% as.formula()
fit_glm <- glm(formula_glm, family = "binomial", data = dat_model)

## (do just forward first, then backward?) [[?]]

fit_glmstep_forward <- MASS::stepAIC(fit_glm, 
                                     scope = scopedef, 
                                     direction = "forward", 
                                     scale = log(nrow(dat_train)), 
                                     trace = TRUE)
summary(fit_glmstep_forward)

fit_glmstep_back <- MASS::stepAIC(fit_glmstep_forward, 
                                  scope = scopedef, 
                                  direction = "backward", 
                                  scale = log(nrow(dat_train)), 
                                  trace = TRUE)
summary(fit_glmstep_back)
AIC(fit_glmstep_back)
BIC(fit_glmstep_back)

save.image(file = file.path(path_tmp, "03-model-training___dump02.Rdata"))


pR2(fit_glmstep_back)            ## pseudo R^2
NagelkerkeR2(fit_glmstep_back)   ## Nagelkerke R^2


dat_this <- dat_eval
pred_this <- predict(fit_glm, newdata = dat_this, type = "response")
resp_this <- factor(as.numeric(pred_this > .5), levels=c(0,1), labels=c("No", "Yes"))
true_this <- dat_this[[varnames_target]]
mltools::mcc(preds = resp_this, actuals = true_this)

dat_this <- dat_eval
#pred_this <- predict(fit_glmstep_back, newdata = dat_this, type = "response")
pred_this <- predict(fit_glmcaret, newdata = dat_this, type = "response")
resp_this <- factor(as.numeric(pred_this > .5), levels=c(0,1), labels=c("No", "Yes"))
true_this <- dat_this[[varnames_target]]
mltools::mcc(preds = resp_this, actuals = true_this)


# parallelMap::parallelStop()

# Stepwise regression using Caret package and RMSE (https://github.com/tirthajyoti/R-stats-machine-learning/blob/master/Stepwise%20regression%2C%20LASSO%2C%20Elastic%20Net.R)
# ==================================================
library(caret)
# Set up repeated k-fold cross-validation

# formula_interact <- paste0(
#   varnames_target, " ~ ", 
#   "(", 
#   paste(varnames_features, collapse = " + "),
#   ") ^ 2"
# ) %>% as.formula()

train_control <- trainControl(method = "cv", number = 3)
# Train the model
fit_glmcaret <- train(formula_gom, data = dat_train,
                     method = "glmStepAIC", 
                     trControl = train_control)

# Result of the stepwise regression
print(step.model2$results)

# Best model
print(step.model2$bestTune)

# Coefficients of best model
print(coef(step.model2$finalModel, 6))




