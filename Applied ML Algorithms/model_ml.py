import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR

data= pd.read_csv("salary_data_cleaned.csv")
data.head()

df_model = data[['avg_salary','Rating','Size','Type of ownership','Industry' ,
            'job_state','same_state','age','python_yn','spark','aws','excel','desc_len']]
df_model

data_ml = pd.get_dummies(df_model , drop_first=True)   #convert categorical data to numerical data
data_ml

from sklearn.model_selection import train_test_split
X = data_ml.drop('avg_salary' , axis = 1)
y = data_ml['avg_salary']
X
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_test
X_train
y_test
y_train

lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_train,y_train)    #0.4980593459532999
y_pred_lr = lr.predict(X_test)
y_pred_lr
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test , y_pred_lr)            #1269.2267679736963



ds = DecisionTreeRegressor()
ds.fit(X_train,y_train)
ds.score(X_train,y_train)                         #1.0
y_pred_ds = ds.predict(X_test)
mean_squared_error(y_pred_ds,y_test)                #1366.9513422818793



svr = SVR()
svr.fit(X_train,y_train)
SVR()
svr.score(X_train,y_train)             #-0.0005660307457742153
y_pred_svr = svr.predict(X_test)
mean_squared_error(y_pred_svr,y_test)            #1800.049245017374



ada = AdaBoostRegressor()
ada.fit(X_train,y_train)
ada.score(X_train,y_train)             #0.3690979974347802
y_pred_ada = ada.predict(X_test)
mean_squared_error(y_pred_ada,y_test)            #1190.0755086073814

