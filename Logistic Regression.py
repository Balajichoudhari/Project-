import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:\Users\mypc\Videos\Data Science\Machine Learning\Files\16th,17th\2.LOGISTIC REGRESSION CODE\Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score 
ac= accuracy_score(y_test,y_pred)
print(ac)

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)


#visualising the training set result
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train 
x1,x2=np.meshgrid(np.arange(start =x_set[:,0].min() -1,stop =x_set[:,0].max() +1,step=0.01),
                  np.arange(start= x_set[:,1].min() -1,stop =x_set[0,1].max() +1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set ==j,0], x_set[y_set== j,1],
                c=ListedColormap(('red','green')) (i),label= j)
    
plt.title('Logistic Regression(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
    

#Visualising the Test set result
from matplotlib.colors import ListedColormap
x_set,y_set=x_set,y_set
x1,x2=np.meshgrid(np.arange(start= x_set[:,0].min() -1,stop=x_set[:,0].max() +1,step=0.01),
                  np.arange(start=x_set[:,1].min() -1,stop=x_set[:,1].max() +1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set== j,0],x_set[y_set ==j,1],
                c= ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression(test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary') 
plt.legend()
plt.show()   
































    
    
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
