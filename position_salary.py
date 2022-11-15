import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\mypc\Videos\Data Science\Machine Learning\Files\8th\1.POLYNOMIAL REGRESSION\Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=0)
x_poly=poly_reg.fit_transform(x)
poly_reg.fit(x_poly,x)

line_reg2=LinearRegression()
line_reg2.fit(x_poly,y)

plt.scatter(x, y, color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('truth or BLUFF(Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(x,y,color='red')
plt.plot(x,line_reg2.predict(x),color='blue')
plt.title('truth or buff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

lin_reg.predict([[5.5]])
line_reg2.predict(poly_reg.fit_transform([[5.5]]))
