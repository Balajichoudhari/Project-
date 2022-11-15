import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\mypc\Videos\Data Science\Machine Learning\Files\4th,7th\7.XGBOOST\Churn_Modelling.csv')
x=dataset.iloc[:,3:-1]
y=dataset.iloc[:,-1]
# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x.iloc[:, 2]=le.fit_transform(x.iloc[:, 2])

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x=np.array(ct.fit_transform(x))

#Splitting dataset into Training set and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.15,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#importing Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
logistic_Classifier=LogisticRegression()
logistic_Classifier.fit(X_train, Y_train)

#Predicting test set values
logistic_Y_pred=logistic_Classifier.predict(X_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
logistic_cm=confusion_matrix(Y_test,logistic_Y_pred)

#Logistic regression accuracy score
from sklearn.metrics import accuracy_score
logistic_ac=accuracy_score(Y_test, logistic_Y_pred)

#importing  KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier=KNeighborsClassifier()
knn_classifier.fit(X_train,Y_train)

#Predicting test set values
knn_Y_pred=knn_classifier.predict(X_test)

#KNN accuracy score
knn_ac=accuracy_score(Y_test,knn_Y_pred)

#importing svm classifier
from sklearn.svm import SVC
svc_classifier=SVC()
svc_classifier.fit(X_train,Y_train)

#svc - predicting test set values
svc_Y_pred=svc_classifier.predict(X_test)

#svc accuracy score
svc_ac=accuracy_score(Y_test,svc_Y_pred)

#importing Gaussian NB
from sklearn.naive_bayes import GaussianNB
gaussian_classifier=GaussianNB()
gaussian_classifier.fit(X_train,Y_train)

gaussian_Y_pred=gaussian_classifier.predict(X_test)

gaussian_ac=accuracy_score(Y_test,gaussian_Y_pred)

#importing Bernoulli NB
from sklearn.naive_bayes import BernoulliNB
bernoulli_classifier=BernoulliNB()
bernoulli_classifier.fit(X_train,Y_train)

bernoulli_Y_pred=bernoulli_classifier.predict(X_test)

bernoulli_ac=accuracy_score(Y_test,bernoulli_Y_pred)

#importing random forest classifier
from sklearn.ensemble import RandomForestClassifier
random_classifier=RandomForestClassifier()
random_classifier.fit(X_train,Y_train)

random_Y_pred=random_classifier.predict(X_test)

random_ac=accuracy_score(Y_test,random_Y_pred)

#importing Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_classifier=DecisionTreeClassifier()
dt_classifier.fit(X_train,Y_train)

dt_Y_pred=dt_classifier.predict(X_test)

dt_ac=accuracy_score(Y_test,dt_Y_pred)

print(logistic_ac)
print(knn_ac)
print(svc_ac)
print(gaussian_ac)
print(bernoulli_ac)
print(dt_ac)














