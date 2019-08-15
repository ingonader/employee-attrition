## ######################################################################### ##
## setup file to be used by other scripts
## ------------------------------------------------------------------------- ##
## for data science assignment "employee attrition"
## ######################################################################### ##

## ========================================================================= ##
## define global variables and set options
## ========================================================================= ##

path_raw <- "."  ## current path of project
path_dat <- file.path(path_raw, "data")
path_r <- file.path(path_raw, "r-scripts")
path_out <- file.path(path_raw, "output")
path_img <- file.path(path_raw, "img")
path_tmp <- file.path(path_raw, "tmp")

options(tibble.width = Inf)

