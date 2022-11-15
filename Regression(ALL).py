import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\mypc\Videos\Data Science\Machine Learning\Files\10th\1.SVR\Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(X,y)

from sklearn.neighbors import KNeighborsRegressor
regressor1=KNeighborsRegressor()
regressor1.fit(X,y)

from sklearn.tree import DecisionTreeRegressor
regressor2=DecisionTreeRegressor()
regressor2.fit(X,y)

from sklearn.svm import SVR
regressor3=SVR(kernel='poly', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1)
regressor3.fit(X,y)

y_pred=regressor.predict([[6.5]])

y_pred=regressor1.predict([[6.5]])

y_pred=regressor2.predict([[6.5]])

y_pred=regressor3.predict([[6.5]])

X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Decission TRee Regressor')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


