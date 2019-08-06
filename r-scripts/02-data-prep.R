## ######################################################################### ##
## data preparation
## ------------------------------------------------------------------------- ##
## for data science assignment "employee attrition"
## ######################################################################### ##

# rm(list = ls())

## ========================================================================= ##
## define global variables and set options
## ========================================================================= ##

path_raw <- "."  ## current path of project
path_dat <- file.path(path_raw, "data")
path_r <- file.path(path_raw, "r-scripts")

options(tibble.width = Inf)

## ========================================================================= ##
## load packages
## ========================================================================= ##

library(dplyr)
library(tibble)
library(readr)
library(magrittr)
library(purrr)
library(ggplot2)

## ========================================================================= ##
## load data
## ========================================================================= ##

dat_raw <- read_csv(
  file = file.path(path_dat, "employee-attrition.csv"),
  col_types = cols(
    .default = col_double(),
    Attrition = col_character(),
    BusinessTravel = col_character(),
    Department = col_character(),
    EducationField = col_character(),
    Gender = col_character(),
    JobRole = col_character(),
    MaritalStatus = col_character(),
    Over18 = col_character(),
    OverTime = col_character()
  ))
head(dat_raw)

## ========================================================================= ##
## prepare data
## ========================================================================= ##

dat_all <- dat_raw

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ## 
## drop features
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ## 

## features to exclude:
## EmployeeCount: constant, always 1
## Over18: constant, always Y
## StandardHours: constant at 80
varnames_raw_exclude <- c(
  "EmployeeCount",
  "Over18",
  "StandardHours"
)

## drop constant variables:
dat_all <- dat_all %>% select(-one_of(varnames_raw_exclude))

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ## 
## convert features to categorical
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ## 

## store categorical variables names:
varnames_categorical <- union(
  names(dat_all)[sapply(dat_all, function(i) is.character(i) | is.factor(i))],
  c(
    "WorkLifeBalance",
    "StockOptionLevel",
    "RelationshipSatisfaction",
    "JobSatisfaction",
    "JobLevel",
    "JobInvolvement",
    "EnvironmentSatisfaction",
    "Education"
  ))

## convert:
dat_all <- dat_all %>% mutate_at(varnames_categorical,
                                 as.factor)

## ========================================================================= ##
## split data: train, eval and test set
## ========================================================================= ##

set.seed(448)

## random sampling? or stratified sampling based on target variable?
idx <- sample(c("train", "eval", "test"), 
              prob = c(.8, .1, .1),
              size = nrow(dat_all),
              replace = TRUE)

dat_train <- dat_all[idx == "train", ]
dat_eval <- dat_all[idx == "eval", ]
dat_test <- dat_all[idx == "test", ]

dim(dat_train)
dim(dat_eval)
dim(dat_test)


