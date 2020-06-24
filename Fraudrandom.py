

import pandas as pd
import numpy as np
# Reading the Fraud Data #################
Fraud = pd.read_csv("D:\\excelR\\Data science notes\\Random forest\\asgmnt\\Fraud_check.csv")
Fraud.head()
Fraud.columns
colnames = list(Fraud.columns)

colnames = ['Undergrad',  'Marital.Status', 'Urban']
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for i in colnames:
    Fraud[i] = le.fit_transform(Fraud[i])



labels=["risky","good"]
bins=[0,30000,1000000]
Fraud1=pd.cut(Fraud.iloc[:,2],labels=labels,bins=bins)
Fraud.drop(["Taxable.Income",],axis=1,inplace=True)
Fraud=pd.concat([Fraud,Fraud1],axis=1)


columnnames = list(Fraud.columns)
columnnames



predictors = columnnames[:5]
target = columnnames[5]

X = Fraud[predictors]
Y = Fraud[target]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

np.shape(Fraud) # 600,7 => Shape 

#### Attributes that comes along with RandomForest function
rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  # 0.72916
rf.predict(X)
##############################

Fraud['rf_pred'] = rf.predict(X)
cols = ['rf_pred','Taxable.Income']
Fraud[cols].head()
Fraud["Taxable.Income"]


from sklearn.metrics import confusion_matrix
confusion_matrix(Fraud['Taxable.Income'],Fraud['rf_pred']) # Confusion matrix

pd.crosstab(Fraud['Taxable.Income'],Fraud['rf_pred'])



print("Accuracy",(476+115)/(476+115+9)*100)

# Accuracy is 98.5
Fraud["rf_pred"]







