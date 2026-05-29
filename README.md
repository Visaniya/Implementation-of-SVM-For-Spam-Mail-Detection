# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Detect File Encoding: Use chardet to determine the dataset's encoding.

Load Data: Read the dataset with pandas.read_csv using the detected encoding.

Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().

Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.

Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.

Train SVM Model: Fit an SVC model on the training data.

Predict Labels: Predict test labels using the trained SVM model.

Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S.VISANIYA
RegisterNumber:  212225040492
*/


import pandas as pd

from google.colab import files
uploaded = files.upload()

data = pd.read_csv("spam.csv", encoding='latin-1')

print(data.head())

print("Dataset Shape:", data.shape)

x = data['v2'].values
y = data['v1'].values

print("x shape:", x.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC

svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

print("Predictions:")
print(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:")
print(acc)

con = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(con)

cl = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(cl)
```

## Output:
<img width="1320" height="631" alt="ML 12 0P 1" src="https://github.com/user-attachments/assets/f4a48b6d-38ec-4cf8-a7d1-7f49b8d0b829" />

<img width="1265" height="220" alt="ML 12 OP 2" src="https://github.com/user-attachments/assets/8b1e67db-a0c8-4429-b99f-d9fdeed29b0d" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
