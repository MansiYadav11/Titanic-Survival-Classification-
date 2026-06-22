import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
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



from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)
pred=model.predict(X_test)
pred

#Confusion matrix for validation or evaluation
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,pred)
 
confusion_matrix(y_test,pred)
print(classification_report(y_test,pred))
