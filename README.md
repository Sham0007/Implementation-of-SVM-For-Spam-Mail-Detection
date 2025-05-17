# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: saravanan sham prakash
RegisterNumber: 212224230254
*/
```
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![image](https://github.com/user-attachments/assets/9083490b-9b35-4730-87b7-df5ec3d68b5d)


![image](https://github.com/user-attachments/assets/5138dcac-a7b3-49c7-9a0b-708b3c6d78a6)


![image](https://github.com/user-attachments/assets/0152f718-2048-48c0-a5b1-0b508e2e606f)


![image](https://github.com/user-attachments/assets/f70c94b2-3877-402b-b3a1-59d425a45416)


![image](https://github.com/user-attachments/assets/de59a868-3db9-4b80-8751-0bfc5bd6d920)


![image](https://github.com/user-attachments/assets/5d41fe5c-af2d-4e7c-a23d-ad4f0844eb3e)











## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
