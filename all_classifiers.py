import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, classification_report
warnings.filterwarnings('ignore')

df=pd.read_csv("/kaggle/input/datasets/ranaghulamnabi/eda-unlocking-the-story-behind-the-numbers/Titanic.csv")
df.head()

#Data preprocessing 

#EDA
df.info()
df.columns

df.drop(['deck', 'embark_town','alive','class', 'who', 'adult_male'],axis=1,inplace=True)

#age and embarked have missing values
df["age"].fillna(df["age"].mean(),inplace=True)
#for embarked , remove the entire two rows because we cant replace this value by mean
df.dropna(subset=["embarked"],inplace=True)
df.info()

#Label Encoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["embarked"] = le.fit_transform(df["embarked"]) # S=2,C=0,Q=1

df=df.astype(int)
df.head()

X=df.drop("survived",axis=1)
Y=df["survived"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

models={
    "Logistic Regression":LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes":GaussianNB(),
    "Decision Tree":DecisionTreeClassifier(),
    "SVM":SVC(probability=True)
}

results=[]

for name,model in models.items():
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    results.append({
        'Model':name,
        'Accuracy':round(acc,4),
        'F1 score':round(f1,4)
    })

results

#saving model in pickle model(Serialization - converting python into bytestream and then stored in a file, Deserilization is opposite)

import joblib 
joblib.dump(models["SVM"],"SVM_titanic.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(X.columns.tolist,"columns.pkl")
