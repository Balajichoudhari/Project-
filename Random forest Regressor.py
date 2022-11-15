import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\mypc\Videos\Data Science\Machine Learning\Files\10th\1.SVR\Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100, criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
regressor.fit(x,y)


y_pred=regressor.predict([[6.5]])

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or bulff(RandomForest Regressor')
plt.xlabel('position salary')
plt.ylabel('salary')