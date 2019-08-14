---
title: "Employee Attrition"
subtitle: "Most important factors and prediction quality"
author: "Ingo Nader"
date: "Aug 2019"
#output: html_document
output: 
  ioslides_presentation:
    css: styles-edx-mod.css
    #logo: img/logo.png
    widescreen: true
    keep_md: true
    #smaller: true  ## only works without "---" slide breaks (use ##)
    slide_level: 2
#csl: plos-one.csl
#link-citations: true
#bibliography: references.yaml
## References: See 
## http://pandoc.org/MANUAL.html#citation-rendering
## https://github.com/jgm/pandoc-citeproc
##
## Comments and Instructions
##
## ## ------------------------------------------- ##
## ## Controlling presentation (best use chrome):
## ## ------------------------------------------- ##
    # 'f' enable fullscreen mode
    # 'w' toggle widescreen mode
    # 'o' enable overview mode
    # 'h' enable code highlight mode
    # 'p' show presenter notes
##
## ## ------------------------------------------- ##
## ## Images
## ## ------------------------------------------- ##
##
## Replace markdown images "![]()" with R's include_graphics()
## (in order for them to scale to slide width properly):
## Search:
## !\[\]\((.*)\)
## Replace with:
## ```{r, eval = TRUE, echo = FALSE, out.width = "100%", fig.align = "left"}\nknitr::include_graphics("\1")\n```
##
##
## ## ------------------------------------------- ##
## ## Font size in slides, and other layout stuff
## ## ------------------------------------------- ##
##
## use {.smaller} after title for single slides
## use {.flexbox .vcenter} for centering of text
## 
## ## ------------------------------------------- ##
## ## color:
## ## ------------------------------------------- ##
##
##   <div class="red2"></div>
## or:
##   <font color="red"> </font>
##
## ## ------------------------------------------- ##
## ## two-column layout:
## ## ------------------------------------------- ##
## 
## <div></div><!-- ------------------------------- needed as is before cols - -->
## <div style="float: left; width: 48%;"><!-- ---- start of first column ---- -->
## Put col 1 markdown here
## </div><!-- ------------------------------------ end of first column ------ -->
## <div style="float: left; width: 4%"><br></div><!-- spacing column -------- -->
## <div style="float: left; width: 48%;"><!-- ---- start of second column --- --> 
## Put col 2 markdown here
## </div><!-- ------------------------------------ end of second column ----- -->
## <div style="clear: both"></div><!-- end cols for text over both cols below -->
##
## additionally, if one column needs to start higher (for right columns and 
## short slide titles, mostly):
## <div style="float: left; width: 30%; margin-top: -15%"><!-- ---- start of second column              --> 
## 
## other possibilities (not as good):
## * In slide title line, use:
##   ## title {.columns-2}
## * put md into this div:
##   <div class="columns-2">  </div>
##
---
[//]: # (
http://www.w3schools.com/css/css_font.asp
http://www.cssfontstack.com/Helvetica
)

<style>  <!-- put CSS here to test quickly -->
</style>

<script type="text/x-mathjax-config">  <!-- LaTeX formula config -->
MathJax.Hub.Config({
  jax: ["input/TeX", "output/HTML-CSS"],
  "HTML-CSS": { 
      preferredFont: "Arial", 
      availableFonts: [],
      scale: 85
      // styles: {".MathJax": {color: "#CCCCCC"}} 
      }
});
</script>




## Problem Statement

Attrition: 

* problem that impacts all businesses 
* leads to significant costs for a business
* including the cost of business disruption, hiring new staff and training new staff. 

<br>

* underderstanding the drivers is crucial
* classification models to predict if an employee is likely to quit could greatly increase HR’s ability to intervene on time and remedy the situation to prevent attrition.

## Dataset

* $n = 1470$ employees and some of their attributes (selection below):

| Name | Description |
|------|-------------|
|AGE| Numerical Value |
|GENDER|(1=FEMALE, 2=MALE)|
|EDUCATION|Numerical Value|
|BUSINESS TRAVEL|(1=No Travel, 2=Travel Frequently, 3=Tavel Rarely)|
|DISTANCE FROM HOME|Numerical Value - THE DISTANCE FROM WORK TO HOME|
|JOB SATISFACTION|Numerical Value - SATISFACTION WITH THE JOB|
|MONTHLY INCOME|Numerical Value - MONTHLY SALARY|
|NUMCOMPANIES WORKED|Numerical Value - NO. OF COMPANIES WORKED AT|
|OVERTIME|(1=NO, 2=YES)|
|PERCENT SALARY HIKE|Numerical Value - PERCENTAGE INCREASE IN SALARY|
|PERFORMANCE RATING|Numerical Value - ERFORMANCE RATING|
|TOTAL WORKING YEARS|Numerical Value - TOTAL YEARS WORKED|
|TRAINING TIMES LAST YEAR|Numerical Value - HOURS SPENT TRAINING|
|WORK LIFE BALANCE|Numerical Value - TIME SPENT BEWTWEEN WORK AND OUTSIDE|
|YEARS AT COMPANY|Numerical Value - TOTAL NUMBER OF YEARS AT THE COMPNAY|
|YEARS SINCE LAST PROMOTION|Numerical Value - LAST PROMOTION|
|ATTRITION|Employee leaving the company (0=no, 1=yes) |

* Target variable `Attrition`: Whether or not an employee has quit
    
## Data Exploration 

<div></div><!-- ------------------------------- needed as is before cols - -->
<div style="float: left; width: 48%;"><!-- ---- start of first column ---- -->

* Target variable `Attrition` imbalanced
* No missing values in the data
* Some features with no variation: 
    * `EmployeeCount`: constant, always 1
    * `Over18`: constant, always Y
    * `StandardHours`: constant at 80

</div><!-- ------------------------------------ end of first column ------ -->
<div style="float: left; width: 4%"><br></div><!-- spacing column -------- -->
<div style="float: left; width: 48%;"><!-- ---- start of second column --- --> 

<img src="../img/attrition-barplot.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />

</div><!-- ------------------------------------ end of second column ----- -->
<div style="clear: both"></div><!-- end cols for text over both cols below -->


## Data Exploration


<div></div><!-- ------------------------------- needed as is before cols - -->
<div style="float: left; width: 38%;"><!-- ---- start of first column ---- -->

* Feature correlations:
```r
cormat_long %>% 
  filter(Var1 != Var2, abs(value) > .8) %>% 
  arrange(value)
```
```
           Var1              Var2      value
1 MaritalStatus  StockOptionLevel -0.8131347
2    Department           JobRole  0.8508444
3      JobLevel TotalWorkingYears  0.8523883
4      JobLevel     MonthlyIncome  0.9675631
```

Pearson correlations,  
polyserial correlations, and  
polychoric correlations

</div><!-- ------------------------------------ end of first column ------ -->
<div style="float: left; width: 4%"><br></div><!-- spacing column -------- -->
<div style="float: left; width: 58%;"><!-- ---- start of second column --- --> 

<img src="../img/cormat.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />

</div><!-- ------------------------------------ end of second column ----- -->
<div style="clear: both"></div><!-- end cols for text over both cols below -->


## Features used in the model

* Features used in the model:

```r
varnames_features
```
```
 [1] "Age"                      "BusinessTravel"           "DailyRate"               
 [4] "DistanceFromHome"         "Education"                "EducationField"          
 [7] "EnvironmentSatisfaction"  "Gender"                   "HourlyRate"              
[10] "JobInvolvement"           "JobRole"                  "JobSatisfaction"         
[13] "MaritalStatus"            "MonthlyIncome"            "MonthlyRate"             
[16] "NumCompaniesWorked"       "OverTime"                 "PercentSalaryHike"       
[19] "PerformanceRating"        "RelationshipSatisfaction" "StockOptionLevel"        
[22] "TotalWorkingYears"        "TrainingTimesLastYear"    "WorkLifeBalance"         
[25] "YearsAtCompany"           "YearsInCurrentRole"       "YearsSinceLastPromotion" 
[28] "YearsWithCurrManager"    
```

* Features excluded:
    * features with no relevant information: `EmployeeNumber`
    * constant features: `EmployeeCount`, `Over18`, `StandardHours`
    * highly correlated features: `JobLevel`, `Department`

## Categorical variables

* Variable description did not provide scale level
* hence, some features were assumed to be categorical and dummy-coded:

```r
varnames_convert_to_cat
```
```
[1] "WorkLifeBalance"          "StockOptionLevel"         "RelationshipSatisfaction"
[4] "JobSatisfaction"          "JobLevel"                 "JobInvolvement"          
[7] "EnvironmentSatisfaction"  "Education" 
```

* all of them are in a range of `[1, 4]` or `[1, 5]`
* No information about how the information was collected
* Might be a Likert-Scale (ordinal or even interval scale)
* But might also be totally unrelated options (nominal scale)
* Assumed to be ordinal scale, just to be on the save side


## Assumptions: Summary

* Data on individual level and not aggregated  
  (even though there is a variable `EmployeeCount`)
* Some variables were assumed to be nominal scale,  
  even though this might not be the case

* [[?]] some assumption about monthlyrate and monthlyincome and their non-relation?


## Train-/Eval-/Test-Split

<div></div><!-- ------------------------------- needed as is before cols - -->
<div style="float: left; width: 48%;"><!-- ---- start of first column ---- -->

* Data was split into 3 parts using random sampling:
    * Training set: $80\%$, $n = 1180$
    * Evaluation set: $10\%$, $n = 135$
    * Test set: $10\%$, $n = 155$

* Distribution of target variable in each part remained essentially unchanged:

</div><!-- ------------------------------------ end of first column ------ -->
<div style="float: left; width: 4%"><br></div><!-- spacing column -------- -->
<div style="float: left; width: 48%;"><!-- ---- start of second column --- --> 

<img src="../img/train-eval-test-no-upsampling.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />

</div><!-- ------------------------------------ end of second column ----- -->
<div style="clear: both"></div><!-- end cols for text over both cols below -->


## Upsampling the Training Set

<div></div><!-- ------------------------------- needed as is before cols - -->
<div style="float: left; width: 48%;"><!-- ---- start of first column ---- -->

* To balance out the classes, SMOTE sampling was used for the training set
    * Synthetic Minority Oversampling Technique
    * Undersamples the majority class
    * Creates synthetic examples of the minority class
    * by randomly varying the features of $k$ nearest neighbours ($k = 5$ in this case)
* Validation and test set remain untouched

</div><!-- ------------------------------------ end of first column ------ -->
<div style="float: left; width: 4%"><br></div><!-- spacing column -------- -->
<div style="float: left; width: 48%;"><!-- ---- start of second column --- --> 

<img src="../img/train-eval-test-smote-upsampling.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />

</div><!-- ------------------------------------ end of second column ----- -->
<div style="clear: both"></div><!-- end cols for text over both cols below -->


## Machine Learning Models

* A number of models were fitted to the training data:
    * Logistic regression (R's `base` package)
    * Elastic net regression (`glmnet` package) 
    * Random forest (`ranger` package)
    * Boosted GLMs (`glmboost` from the model-based boosting package `mboost`)
    * XGBoost (`xgboost` package)
    * AdaBoost (`adaboost` package) 
    * Neural net with 1 hidden layer (`nnet` package)
    

## Model Fitting

* **Data preparation**: nominal variables were manually dummy-coded for model tuning and (initial) model fitting (necessary for applying xgboost, unfortunately)
* All model tuning and fitting was performed using the `mlr` package

<br>

* **Parameter tuning**: 50 iterations of random search with 6-fold CV within the training set
* 





## todo

<div></div><!-- ------------------------------- needed as is before cols - -->
<div style="float: left; width: 48%;"><!-- ---- start of first column ---- -->


</div><!-- ------------------------------------ end of first column ------ -->
<div style="float: left; width: 4%"><br></div><!-- spacing column -------- -->
<div style="float: left; width: 48%;"><!-- ---- start of second column --- --> 

<img src="../img/model-fit-mcc-train-cv.jpg" width="100%" style="display: block; margin: auto auto auto 0;" />

</div><!-- ------------------------------------ end of second column ----- -->
<div style="clear: both"></div><!-- end cols for text over both cols below -->


