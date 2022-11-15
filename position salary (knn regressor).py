import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\mypc\Downloads\10th\1.SVR\Position_Salaries.csv')
x=dataset.iloc[:,1:2]
y=dataset.iloc[:,2]

from sklearn.neighbors import KNeighborsRegressor
regressor=KNeighborsRegressor()
regressor.fit(x,y)

y_pred=regressor.predict([[6.5]])

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or bluff(KNN Regressor)')
plt.xlabel('Position salary')
plt.ylabel('Salary')
plt.show()