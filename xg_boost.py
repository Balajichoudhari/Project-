import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\mypc\Videos\Data Science\Machine Learning\Files\4th,7th\7.XGBOOST\Churn_Modelling.csv')
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

print(x)
print(y)

#Encoding categorical data
#label Encoding the "gender" column

from sklearn.preprocessing import LabelEncoder
le=