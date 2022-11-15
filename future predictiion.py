import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\mypc\Videos\Data Science\Machine Learning\Files\18th\Social_Network_Ads.csv")
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values


future_dataset=pd.read_excel(r"C:\Users\mypc\Videos\Data Science\Machine Learning\Files\18th\future prediction.xlsx")
future_dataset.to_csv('future_prediction.csv')
x1=future_dataset.iloc[:,[2,3]].values




from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
x1=sc.fit_transform(x1)

#fit the dataset into logistic regression model
from sklearn.linear_model import LogisticRegression
classifier1=LogisticRegression()
classifier1.fit(x_train,y_train)
y_future_pred1=classifier1.predict(x1)

#fit the dataset into KNN model
from sklearn.neighbors import KNeighborsClassifier
classifier2=KNeighborsClassifier()
classifier2.fit(x_train,y_train)
y_pred=classifier2.predict(x_test)
y_future_pred2=classifier2.predict(x1)

#fit the model in Decision tree model
from sklearn.tree import DecisionTreeClassifier
classifier3=DecisionTreeClassifier()
classifier3.fit(x_train,y_train)
y_future_pred3=classifier3.predict(x1) 

#fit the model into GaussianNB
from sklearn.naive_bayes import GaussianNB
classifier4=GaussianNB()
classifier4.fit(x_train,y_train)
y_pred=classifier4.predict(x_test)
y_future_pred4=classifier4.predict(x1)

#fit the model into SVC
from sklearn.svm import SVC
classifier5=SVC()
classifier5.fit(x_train,y_train)
y_pred=classifier5.predict(x_test) 
y_future_pred=classifier5.predict(x1)

#fit the model in Bernoli
from sklearn.naive_bayes import BernoulliNB
classifier6=BernoulliNB()
classifier6.fit(x_train,y_train)
y_pred=classifier6.predict(x_test)
y_future_pred=classifier6.predict(x1)

