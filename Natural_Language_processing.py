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


#training the Naive Bayes model
from sklearn.naive_bayes import BernoulliNB
classifier =BernoulliNB()
classifier.fit(X_train,y_train)

#predicting the test set result
y_pred= classifier.predict(X_test)

#confusion martrix
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test, y_pred)
print(cm)

#accuracy matrix
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)

































