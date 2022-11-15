import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\mypc\Videos\Data Science\Machine Learning\Files\7th\MULTIPLE LINEAR REGRESSION\50_Startups.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]

X=pd.get_dummies(X)

#spliting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#import linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm

X = np.append(arr = np.ones((50,1)).astype(int),values = X ,axis = 1)


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog=y , exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog = y, exog =X_opt).fit()

X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog = y,exog =X_opt).fit()
regressor_OLS.summary()


X_opt = X[:,[0,1,3]]
regressor_OLS = sm.OLS(endog = y,exog =X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog =y,exog =X_opt).fit()
regressor_OLS.summary()




