import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
#! pip install pmdarima #installing the pmdarima package
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import matplotlib.dates as mdates
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics 
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from math import sqrt
from sklearn import svm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
#!pip3 install pydotplus --no-cache-dir --no-binary :all:
#pip install six
#pip install --upgrade scikit-learn==0.20.3
from six import StringIO
#from sklearn.externals.six import StringIO 
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.ensemble import RandomForestRegressor


os.chdir("/Users/samboamaza/OneDrive - UWM/Courses/ECON 790 - Research Seminar for M.A. Students/Data/Realtor.com")

#impoer and clean data
df_realtor = pd.read_csv("RDC_Inventory_Core_Metrics_Metro_History.csv")
df_realtor = df_realtor.rename(columns = {"month_date_yyyymm" : "date", "cbsa_title":"region"})
df_realtor.date = pd.to_datetime(df_realtor['date'], format = '%Y%m')
df_realtor['date'] = df_realtor['date'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m'))
df_realtor['date'] = df_realtor['date'].astype('datetime64')

#EDA
df_realtor_grouped = df_realtor.groupby('region', as_index=False)['median_listing_price'] #groupby region and only return the MLP variable
df_realtor_grouped = df_realtor_grouped.mean().dropna().sort_values('median_listing_price', ascending = False) #drop na's and sort in ascending order


#Top 10 metropolitan areas
df_realtor_grouped.head(10)
reg_list = ['vineyard haven, ma','summit park, ut','santa maria-santa barbara, ca','san jose-sunnyvale-santa clara, ca','jackson, wy-id','napa, ca','santa cruz-watsonville, ca','salinas, ca','san francisco-oakland-hayward, ca','hailey, id']

#selecting the data with only the top 10 metrpolitan areas
df = df_realtor[df_realtor['region'].isin(reg_list)]

#dropoing irrelevant variables
df = df.drop(columns = ['cbsa_code','HouseholdRank','pending_ratio_mm','pending_ratio_yy','total_listing_count_mm', 'total_listing_count_yy','average_listing_price_mm','average_listing_price_yy','median_square_feet_mm','median_square_feet_yy','pending_listing_count_mm','pending_listing_count_yy','price_reduced_count_mm','price_reduced_count_yy','price_increased_count_mm','price_increased_count_yy','new_listing_count_mm','new_listing_count_yy','median_days_on_market_mm','median_days_on_market_yy','active_listing_count_mm','active_listing_count_yy','median_listing_price_mm','median_listing_price_yy','median_listing_price_per_square_foot','median_listing_price_per_square_foot_mm','median_listing_price_per_square_foot_yy'], axis = 1)

#Summary statistics
df.describe()
df.groupby('region')['median_listing_price'].describe(include = 'all')


# Checking if the sample is balanced;
df.groupby('region').size()

# Trend in time series for the top regions  
reg_list = ['vineyard haven, ma','summit park, ut','santa maria-santa barbara, ca','san jose-sunnyvale-santa clara, ca','jackson, wy-id','napa, ca','santa cruz-watsonville, ca','salinas, ca','san francisco-oakland-hayward, ca','hailey, id']
reg_series = pd.DataFrame(df_realtor[(df_realtor['region'].\
    isin(reg_list))][['date','region','median_listing_price']].\
    dropna().\
    groupby(['date', 'region'])['region','median_listing_price'].mean().unstack())
reg_series.plot(figsize=(15,8), linewidth=3)
plt.xlabel('Year')
plt.ylabel('Median Listing Prices \n (in million USD)')
L = plt.legend()
L.get_texts()[0].set_text('Hailey, ID')
L.get_texts()[1].set_text('Jackson, WY-ID')
L.get_texts()[2].set_text('Napa, CA')
L.get_texts()[3].set_text('Salinas, CA')
L.get_texts()[4].set_text('San Francisco-Oakland-Hayward, CA')
L.get_texts()[5].set_text('San Jose-Sunnyvale-Santa Clara, CA')
L.get_texts()[6].set_text('Santa Cruz=Watsonville, CA')
L.get_texts()[7].set_text('Santa Maria-Santa Barbara, CA')
L.get_texts()[8].set_text('Summit Park, UT')
L.get_texts()[9].set_text('Vineyard Haven, MA')
plt.show()


# The average prices by regions
plt.figure(figsize=(10,11))
plt.title("Median Listing Price of Homes by Region")
Av= sns.barplot(x="median_listing_price",y="region",data= df)

# DATA PROCESSING 

# Determining the regressor variables using pearson correlation matrix
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Draw the heatmap with the mask and correct aspect ratio
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True,mask = mask)

# Standardizing (scaling) the independent variables
scaler = StandardScaler()
df.loc[:,'active_listing_count':'pending_ratio'] = scaler.fit_transform(df.loc[:,'active_listing_count':'pending_ratio'])
df.head()

# Specifying dependent and independent variables
df = df.dropna().sort_values('date', ascending = True) 
X = df.drop(['median_listing_price'], axis = 1)
y = df['median_listing_price']
y = np.log1p(y)


# Labeling the categorical and numerical variables
Xcat = pd.get_dummies(X["region"], drop_first = True)
Xnum = X[['active_listing_count','median_days_on_market','median_square_feet','average_listing_price']]

#cocanating the categorical and numerical variables
X= pd.concat([Xcat, Xnum], axis = 1) 
X.shape

#new data frame with the y variable and modified x variables
F_DF = pd.concat([y,X],axis=1)
F_DF.tail(2)

#train and test split
X_train = X[0:306]
y_train = y[0:306]
X_test = X[306:]
y_test = y[306:]

# 1. MULTILINEAR REGRESSION MODEL
LinReg = LinearRegression()
LinReg.fit(X_train,y_train)
print("R-squared of Linear Regression:",LinReg.score(X_train,y_train))
print('MAE: ',metrics.mean_absolute_error(y_test, LinReg.predict(X_test)))
print('MSE: ',metrics.mean_squared_error(y_test, LinReg.predict(X_test)))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, LinReg.predict(X_test))))

# Residual analysis of the linear regression model 
# Creating a Histogram of Residuals
plt.figure(figsize=(6,4))
sns.distplot(y_test - LinReg.predict(X_test))
plt.title('Distribution of residuals')

plt.figure(figsize=(6,4))
plt.scatter(y_test,LinReg.predict(X_test))


# we can confirm the R2 value (moreover, get the R2 Adj.value)
# of the model by statsmodels library of python
X_train = sm.add_constant(X_train) # adding a constant
model = sm.OLS(y_train, X_train).fit()
print(model.summary())

#export to LaTex
# beginningtex = """\\documentclass{report}
# \\usepackage{booktabs}
# \\begin{document}"""
# endtex = "\end{document}"

# f = open('lin.txt', 'w')
# f.write(beginningtex)
# f.write(model.summary().as_latex())
# f.write(endtex)
# f.close()

# LASSO AND RIDGE REGRESSIONS
X_train = np.array(X[0:306])
y_train = np.array(y[0:306])
X_test = np.array(X[306:])
y_test = np.array(y[306:])


alphas = np.logspace(-5,3,20)

clf = GridSearchCV(estimator=linear_model.Ridge(), param_grid=dict(alpha=alphas), cv=10)
clf.fit(X_train, y_train)
optlamGSCV_R = clf.best_estimator_.alpha
print('Optimum regularization parameter (Ridge):', optlamGSCV_R)

clf = GridSearchCV(estimator=linear_model.Lasso(), param_grid=dict(alpha=alphas), cv=10)
clf.fit(X_train, y_train)
optlamGSCV_L= clf.best_estimator_.alpha
print('Optimum regularization parameter (Lasso):', optlamGSCV_L)

# RIDGE REGRESSION
ridge = linear_model.Ridge(alpha = optlamGSCV_R)
ridge.fit(X_train, y_train)
print('RMSE value of the Ridge Model is: ',np.sqrt(metrics.mean_squared_error(y_test, ridge.predict(X_test))))
print("R-squared of Ridge Regression: ",ridge.score(X_train, y_train))
# Residual analysis of the ridge regession
plt.figure(figsize=(6,4))
sns.distplot(y_test - ridge.predict(X_test))
plt.title('Distribution of residuals')

# LASSO REGRESSION
lasso = linear_model.Lasso(alpha = optlamGSCV_L)
lasso.fit(X_train, y_train)
print('RMSE value of the Lasso Model is: ',np.sqrt(metrics.mean_squared_error(y_test, lasso.predict(X_test))))
print("R-squared of Lasso Regression: ",lasso.score(X_train, y_train))
# Residual analysis of the lasso regession
plt.figure(figsize=(6,4))
sns.distplot(y_test - lasso.predict(X_test))
plt.title('Distribution of residuals')

#plot of the residuals of the ridge and the lasso side by side 
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
sns.distplot(y_test - ridge.predict(X_test), ax = ax1)
sns.distplot(y_test - lasso.predict(X_test), ax = ax2)
ax1.set_title('Ridge Residuals')
ax2.set_title('Lasso Residuals')

# KNN MODEL
# Defining the mape function
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

# Determining the optimal k parameter for the KNN regression
for n in range(1,10):
    Knn = neighbors.KNeighborsRegressor(n_neighbors=n,weights='distance')
    Knn.fit(X_train, y_train)  
    error = sqrt(metrics.mean_squared_error( y_test, Knn.predict(X_test)))
    mae = metrics.mean_absolute_error(y_test,Knn.predict(X_test))
    mape = MAPE(y_test,Knn.predict(X_test))
    print(n,error, mae, mape) 


Knn = neighbors.KNeighborsRegressor(n_neighbors=3,weights='distance')
Knn.fit(X_train, y_train)  
error = sqrt(metrics.mean_squared_error( y_test, Knn.predict(X_test))) 
print('RMSE value of the KNN Model is:', error)
print("R-squared of KNN: ",Knn.score(X_train, y_train))

# SVR MODEL
# First, let's choose which kernel is the best for our data
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_train, y_train)
    print(k,confidence)

for k in ['linear', 'poly','rbf', 'sigmoid']:
    for c in [0.01,0.1,1,10]:
        for e in [0.01,0.1,1]:
            clf = svm.SVR(kernel=k, C=c, epsilon=e)
            clf.fit(X_train, y_train)
            conf = clf.score(X_train, y_train)
            pred = clf.predict(X_test)
            mse = metrics.mean_squared_error(y_test,pred)
            rmse = np.sqrt(mse)
            print(k, c, e, rmse)

for p in [2,3,4,5]:
    for c in [0.01,0.1,1,10]:
        for e in [0.01,0.1,1]:
            clf2 = SVR(kernel= 'poly', C=c, epsilon=e, degree=p)
            clf2.fit(X_train,y_train)
            conf2 = clf.score(X_train, y_train)
            pred2 = clf.predict(X_test)
            mse2 = metrics.mean_squared_error(y_test,pred)
            rmse2 = np.sqrt(mse2)
            print(p, c, rmse2)            

Svr=SVR(kernel='linear', C=1 ,epsilon=0.1, gamma= 0.5)  
Svr.fit(X_train,y_train)
print(Svr.score(X_train,y_train))
error = sqrt(metrics.mean_squared_error(y_test,Svr.predict(X_test))) #calculate rmse
print('RMSE value of the SVR Model is:', error)
#accuracy check
Svr.predict(X_test)[0:5] 
y_test[0:5]

# DECISION TREE REGRESSOR
minDepth = 100
minRMSE = 100000

for depth in range(2,10):
  tree_reg = DecisionTreeRegressor(max_depth=depth)
  tree_reg.fit(X_train, y_train)
  y_pred = tree_reg.predict(X_test)
  mse = metrics.mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  print("Depth:",depth,", MSE:", mse)
  print("Depth:",depth, ",RMSE:", rmse)
  
  if rmse < minRMSE:
    minRMSE = rmse
    minDepth = depth
     
print("MinDepth:", minDepth)
print("MinRMSE:", minRMSE)

DTree=DecisionTreeRegressor(max_depth=minDepth)
DTree.fit(X_train,y_train)
print(DTree.score(X_train,y_train))  
print('MAE:', metrics.mean_absolute_error(y_test, DTree.predict(X_test)))
print('MSE:', metrics.mean_squared_error(y_test, DTree.predict(X_test)))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, DTree.predict(X_test))))

#RANDOM FOREST
RForest = RandomForestRegressor()
RForest.fit(X_train,y_train)
print(RForest.score(X_train,y_train)) 

print('MAE:', metrics.mean_absolute_error(y_test, RForest.predict(X_test)))
print('MSE:', metrics.mean_squared_error(y_test, RForest.predict(X_test)))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, RForest.predict(X_test))))

# CONCLUSION 
# Comparing The RMSE Values Of The Models
# Linear Regression RMSE : 
print('RMSE value of the Linear Regr : ',round(np.sqrt(metrics.mean_squared_error(y_test, LinReg.predict(X_test))),4))

# Ridge RMSE             : 
print('RMSE value of the Ridge Model : ',round(np.sqrt(metrics.mean_squared_error(y_test, ridge.predict(X_test))),4))

# Lasso RMSE             : 
print('RMSE value of the Lasso Model : ',round(np.sqrt(metrics.mean_squared_error(y_test, lasso.predict(X_test))),4))

# KNN RMSE               : 
print('RMSE value of the KNN Model   : ',round(np.sqrt(metrics.mean_squared_error(y_test, Knn.predict(X_test))),4))

# SVR RMSE               : 
print('RMSE value of the SVR Model   : ',round(np.sqrt(metrics.mean_squared_error(y_test, Svr.predict(X_test))),4))

# Decision Tree RMSE     : 
print('RMSE value of the Decis Tree  : ',round(np.sqrt(metrics.mean_squared_error(y_test, DTree.predict(X_test))),4))

# Random Forest RMSE     : 
print('RMSE value of the Rnd Forest  : ',round(np.sqrt(metrics.mean_squared_error(y_test, RForest.predict(X_test))),4))