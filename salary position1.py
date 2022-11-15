import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\mypc\Videos\Data Science\Machine Learning\Files\1.SVR\Position_Salaries.csv')
x=dataset.iloc[:, 1:2]
y=dataset.iloc[:,2]


from sklearn.svm import SVR
regressor=SVR()
regressor.fit(x,y)

y_pred=regressor.predict([[6.5]])

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or bluff(SVR)')
plt.xlabel('Position Salary')
plt.ylabel('Salary')
plt.show()