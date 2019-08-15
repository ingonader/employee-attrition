## ######################################################################### ##
## function library
## ------------------------------------------------------------------------- ##
## for data science assignment "employee attrition"
## ######################################################################### ##

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

#' function to do one-hot encoding of categorical variables:
#' (based on global variables only for exploration)
#'
#' @param dat Data that should be one-hot encoded
#' @param interaction Either \code{NA} for no interactions, or the
#'   level of interactions that should be generated (e.g., 2 for
#'   main effects and all two-way interactions, 3 for these and all
#'   three-way interactions -- which is probably overkill).
#' 
#' @return A data frame without factors, as all factors have been
#'   dummy-coded using \code{model.matrix}. Data does not contain
#'   an intercept to avoid rank-deficient matrices when fitting 
#'   regression models (will add intercept by default)
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
