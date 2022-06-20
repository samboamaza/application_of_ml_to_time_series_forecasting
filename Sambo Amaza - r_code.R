#installing the necessary packages
install.packages("dplyr")                       # Install dplyr
install.packages("fastDummies")
install.packages("readr")
install.packages("caret")
install.packages("repr")
install.packages("FNN")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("readr")
install.packages("caTools")
install.packages("party")
install.packages("partykit")
install.packages("randomForest")
library(randomForest)
library("dplyr")                                
library(tidyverse)
library(lubridate)
library(fastDummies)
library(plyr)
library(readr)
library(caret)
library(ggplot2)
library(repr)
library(glmnet)
library(plotmo) # for plot_glmnet
library(FNN)
library(rpart)
library(readr)
library(caTools)
library(party)
library(partykit)
library(rpart.plot)


#set working directory
setwd("/Users/samboamaza/OneDrive - UWM/Courses/ECON 790 - Research Seminar for M.A. Students/Data/Realtor.com")

#import and clean data
df_realtor = read.csv("RDC_Inventory_Core_Metrics_Metro_History.csv")
typeof(df_realtor)
attributes(df_realtor)
str(df_realtor)

#renaming column names
names(df_realtor)
names(df_realtor)[1] <- "date"
names(df_realtor)[3] <- "region"

#converting the "date" column to a datetime dataframe

#group by reigion
df_realtor_grouped = select(df_realtor, region, median_listing_price)#selecting cols

#grouping by region
df_realtor_grouped = df_realtor_grouped %>%
  group_by(region) %>%
  summarize(median_listing_price = mean(median_listing_price, na.rm = TRUE))

#sorting by median listing price in decending order
df_realtor_grouped  <-df_realtor_grouped[order(-df_realtor_grouped$median_listing_price),]

#top 10 metropolitan areas
df_realtor_grouped[0:10,]
reg_list = c('vineyard haven, ma','summit park, ut','santa maria-santa barbara, ca',
            'san jose-sunnyvale-santa clara, ca','jackson, wy-id','napa, ca',
            'santa cruz-watsonville, ca','salinas, ca','san francisco-oakland-hayward, 
            ca','hailey, id')

df = subset(df_realtor, select = -c(cbsa_code,HouseholdRank,pending_ratio_mm,pending_ratio_yy,total_listing_count_mm, total_listing_count_yy,average_listing_price_mm,average_listing_price_yy,median_square_feet_mm,median_square_feet_yy,pending_listing_count_mm,pending_listing_count_yy,price_reduced_count_mm,price_reduced_count_yy,price_increased_count_mm,price_increased_count_yy,new_listing_count_mm,new_listing_count_yy,median_days_on_market_mm,median_days_on_market_yy,active_listing_count_mm,active_listing_count_yy,median_listing_price_mm,median_listing_price_yy,median_listing_price_per_square_foot,median_listing_price_per_square_foot_mm,median_listing_price_per_square_foot_yy))

df = df %>%
  group_by(date,region) %>%
  summarize(median_listing_price = mean(median_listing_price, na.rm = TRUE),
            active_listing_count = mean(active_listing_count),
            median_days_on_market = mean(median_days_on_market),
            active_listing_count = mean(active_listing_count),
            median_days_on_market = mean(median_days_on_market),
            new_listing_count = mean(new_listing_count),
            price_increased_count = mean(price_increased_count),
            price_reduced_count = mean(price_reduced_count),
            pending_listing_count = mean(pending_listing_count),
            median_square_feet = mean(median_square_feet),
            average_listing_price = mean(average_listing_price),
            total_listing_count = mean(total_listing_count),
            pending_ratio = mean(pending_ratio))

df = df[order(-df$median_listing_price),]

df = df[df$region %in% reg_list, ]

#######################
df = read_csv("df.csv")
df = subset(df, select = -c(X1))

# Specifying dependent and independent variables
X = subset(df, select = -c(date))
y = subset(df, select = c(median_listing_price))
X$median_listing_price = log(X$median_listing_price)

# Labeling the categorical and numerical variables
Xcat <- fastDummies::dummy_cols(X, select_columns = "region", remove_first_dummy = TRUE)
df <- subset(Xcat, select = -c(region,new_listing_count,price_increased_count,price_reduced_count,pending_listing_count,total_listing_count,pending_ratio))

# Standardizing (scaling) the independent variables
df$active_listing_count = scale(df$active_listing_count)
df$median_days_on_market = scale(df$median_days_on_market)
df$median_square_feet = scale(df$median_square_feet)
df$average_listing_price = scale(df$average_listing_price)

#train and test split
train = df[0:306,]
test = df[306:510,]

# evaluation metrics
eval_metrics = function(model, df, predictions, target){
  resids = df[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  r2 = as.character(round(summary(model)$r.squared, 2))
  adj_r2 = as.character(round(summary(model)$adj.r.squared, 2))
  print(adj_r2) #Adjusted R-squared
  print(as.character(round(sqrt(sum(resids2)/N), 2))) #RMSE
}

# 1. MULTILINEAR REGRESSION MODEL
LinReg <- lm(train$median_listing_price ~ ., data = train)
summary(LinReg)
plot.histogram(LinReg)

# Residual analysis of LinReg
plot.histogram(LinReg) #QQ plot

LinReg.res = resid(LinReg)
myhist <- hist(LinReg.res) 
mydensity <- density(LinReg.res)
multiplier <- myhist$counts / myhist$density
mydensity$y <- mydensity$y * multiplier[1]
plot(myhist)#histogram of residuals
lines(mydensity)

#predicting and evaluating model on train data 
predictions = predict(LinReg, newdata = train)
eval_metrics(LinReg, train, predictions, target = 'median_listing_price')

#predicting and evaluating on test data
predictions = predict(LinReg, newdata = test)
ols_rmse = eval_metrics(LinReg, test, predictions, target = 'median_listing_price')

# 2. LASSO AND RIDGE REGRESSIONS
X_train = subset(df, select = -c(median_listing_price))
X_train = as.matrix(X_train[0:306,])
y_train = train$median_listing_price

X_test = subset(df, select = -c(median_listing_price))
X_test = as.matrix(X_test[306:510,])
y_test = test$median_listing_price

lambdas <- 10^seq(2, -3, by = -.1)
ridge_reg = glmnet(X_train, y_train, nlambda = 25, alpha = 0, family = 'gaussian', lambda = lambdas)
summary(ridge_reg)
plot(ridge_reg.sp,xvar="lambda",label=TRUE)

cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min
optimal_lambda

# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  
#Model performance metrics
data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# Prediction and evaluation on train data
predictions_train <- predict(ridge_reg, s = optimal_lambda, newx = X_train)
ridge_rmse = eval_results(y_train, predictions_train, train)

# Prediction and evaluation on test data
predictions_test <- predict(ridge_reg, s = optimal_lambda, newx = X_test)
eval_results(y_test, predictions_test, X_test)

# LASSO Regression
lambdas <- 10^seq(2, -3, by = -.1)
# Setting alpha = 1 implements lasso regression
lasso_reg <- cv.glmnet(X_train, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)
# Best 
lambda_best <- lasso_reg$lambda.min 
lambda_best

#fitting the best lasso model
lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)
predictions_train <- predict(lasso_model, s = lambda_best, newx = X_train)
eval_results(y_train, predictions_train, train)

predictions_test <- predict(lasso_model, s = lambda_best, newx = X_test)
lasso_rmse = eval_results(y_test, predictions_test, X_test)

#KNN
pred_caret <- train(X_train, y_train, method = "knn", preProcess = c("center","scale"))
pred_caret
plot(pred_caret)

#knn regression model
knn_reg <- knn.reg(train, test, train$median_listing_price, k = 5)
print(knn_reg)

#eval results
knn_rmse = eval_results(test$median_listing_price,train$median_listing_price,X_test)

#Decision Tree
rtree <- rpart(median_listing_price ~ ., train)
rpart.plot(rtree)

# random forest
rf <- randomForest(train$median_listing_price ~ train$active_listing_count+train$median_days_on_market+train$median_square_feet+train$average_listing_price, data=train)
print(rf)
pred = predict(rf, newdata=test$median_listing_price)
pred
