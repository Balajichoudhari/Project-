# Natural Language Processing

#Import The libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset=pd.read_csv(r'C:\Users\mypc\Videos\Data Science\Machine Learning\Files\24th\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv',delimiter='\t',quoting= 3 )

#cleaning the text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#creatingthe Bag of Words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=0)


#logistic classifier
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train,y_train)

logistic_y_pred = logistic_classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
logistic_cm = confusion_matrix(y_test,logistic_y_pred)
print(logistic_cm)

#accuracy matrix
from sklearn.metrics import accuracy_score
logistic_ac = accuracy_score(y_test,logistic_y_pred)
print(logistic_ac) 

#Naive bayes
from sklearn.naive_bayes import BernoulliNB
Bernoli_classifier = BernoulliNB()
Bernoli_classifier.fit(X_train,y_train)

y_pred_bernoli = Bernoli_classifier.predict(X_test)

bernoli_ac = accuracy_score(y_test,y_pred_bernoli)
print(bernoli_ac)

#multinomial
from sklearn.naive_bayes import MultinomialNB
multinomial_classifier = MultinomialNB()
multinomial_classifier.fit(X_train,y_train)

y_pred_multinomial = multinomial_classifier.predict(X_test)

multinomial_ac = accuracy_score(y_test,y_pred_multinomial)
print(multinomial_ac)

#guassian 
from sklearn.naive_bayes import GaussianNB
guassian_classifier = GaussianNB()
guassian_classifier.fit(X_train,y_train)

y_pred_guassian = guassian_classifier.predict(X_test)

guassian_ac = accuracy_score(y_test,y_pred_guassian)
print(guassian_ac)

#training the SVC  Bayes model
from sklearn.svm import SVC
classifier =SVC( C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
classifier.fit(X_train,y_train)

#predicting the test set result
y_pred_SVC= classifier.predict(X_test)

#accuracy the SVC
SVC_ac = accuracy_score(y_test,y_pred_SVC)
print(SVC_ac)

#Randomforest Classifier
from sklearn.ensemble import RandomForestClassifier
random_classifier = RandomForestClassifier()
random_classifier.fit(X_train,y_train)

y_pred_random = random_classifier.predict(X_test)

random_ac = accuracy_score(y_test, y_pred_random)
print(random_ac)

#decision tree
from sklearn.tree import DecisionTreeClassifier
decision_classifier = DecisionTreeClassifier()
decision_classifier.fit(X_train,y_train)

y_pred_decision = decision_classifier.predict(X_test)

decision_ac = accuracy_score(y_test, y_pred_decision)
print(decision_ac)

#KNN model
from sklearn.neighbors import KNeighborsClassifier
KNe_classifier = KNeighborsClassifier()
KNe_classifier.fit(X_train,y_train)

y_pred_KNe = KNe_classifier.predict(X_test)

KNe_accuracy = accuracy_score(y_test, y_pred_KNe)
print(KNe_accuracy)


#xgboost model 
from xgboost import XGBClassifier
xg_classifier = XGBClassifier()
xg_classifier.fit(X_train,y_train)

y_pred_xg = xg_classifier.predict(X_test)

xg_accuracy = accuracy_score(y_test,y_pred_xg)
print(xg_accuracy)






















