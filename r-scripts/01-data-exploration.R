## ######################################################################### ##
## data exploration
## ------------------------------------------------------------------------- ##
## for data science assignment "employee attrition"
## ######################################################################### ##

# rm(list = ls())

## ========================================================================= ##
## define global variables and set options
## ========================================================================= ##

path_raw <- "."  ## current path of project
path_dat <- file.path(path_raw, "data")

options(tibble.width = Inf)

## ========================================================================= ##
## load packages
## ========================================================================= ##

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

## create a copy to be modified for modeling:
dat_all <- dat_raw

## ========================================================================= ##
## raw data exploration 
## ========================================================================= ##

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ## 
## check missing values
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ## 

dat_raw %>% map_int(~sum(is.na(.x)))
## no missing values

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ## 
## univariate feature exploration
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ## 

## define variable types of raw data for easy selection:
varnames_raw_numeric <- names(dat_raw)[sapply(dat_raw, is.numeric)]
varnames_raw_categorical <- names(dat_raw)[sapply(dat_raw, function(i) is.character(i) | is.factor(i))]

## ------------------------------------------------------------------------ ##
## check target variable
## ------------------------------------------------------------------------ ##

## check distribution of target variable:
table(dat_raw[["Attrition"]]) %>% prop.table()

## ------------------------------------------------------------------------ ##
## check categorical variables
## ------------------------------------------------------------------------ ##

## barplots:
plot_uni_cat <- map(varnames_raw_categorical, ~ ggplot(dat_raw, aes_string(x = .x)) + geom_bar() + labs(x = .x))
names(plot_uni_cat) <- varnames_raw_categorical
plot_uni_cat

## further inspection of categorical variables:
ggplot(dat_raw, aes_string(x = "EducationField")) + geom_bar()
ggplot(dat_raw, aes_string(x = "Department")) + geom_bar()
## human ressources rather infrequent

## check column EmployeeCount:
table(dat_raw[["EmployeeCount"]])

## ------------------------------------------------------------------------ ##
## check numeric variables
## ------------------------------------------------------------------------ ##

## histograms:
plot_uni_num <- map(varnames_raw_numeric, ~ ggplot(dat_raw, aes_string(x = .x)) + geom_histogram(bins = 50) + labs(x = .x))
names(plot_uni_num) <- varnames_raw_numeric
plot_uni_num

## further inspection:
table(dat_raw[["StandardHours"]])


## features to exclude:
## EmployeeCount: constant, always 1
## Over18: constant, always Y
## StandardHours: constant at 80
varnames_raw_exclude <- c(
  "EmployeeCount",
  "Over18",
  "StandardHours"
)
dat_all <- dat_all %>% select(-one_of(varnames_raw_exclude))

## features to convert to categorical:
## WorkLifeBalance
## StockOptionLevel
## RelationshipSatisfaction? 1-4, might be ordinal or likert-type...
## JobSatisfaction: 1-4, might be ordinal or likert-type...
## JobLevel: 1-5, might be ordinal or likert-type...
## JobInvolvement: 1-4, might be ordinal or likert-type...
## EnvironmentSatisfaction: 1-4, might be ordinal or likert-type...
## Education: 1-5, might be ordinal or likert-type...





