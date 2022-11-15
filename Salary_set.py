import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

salary=pd.read_csv(r'C:\Users\mypc\Videos\Data Science\Machine Learning\Files\4th\SIMPLE LINEAR REGRESSION\slr\Salary_Data.csv')
x=salary.iloc[:,:-1]
y=salary.iloc[:,1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Year OF Exprience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Year of experience')
plt.ylabel('Salary')
plt.show()