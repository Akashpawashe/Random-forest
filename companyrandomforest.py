import pandas as pd
import numpy as np
# Reading the data Data #################
data = pd.read_csv("D:\\excelR\\Data science notes\\Random forest\\asgmnt\\Company_Data.csv")
data.head()
data.columns
data=data[['CompPrice', 'Income', 'Advertising', 'Population', 'Price','Age', 'Education', 'ShelveLoc', 'Urban', 'US','Sales']]
data.head()
colnames = list(data.columns)
predictors = colnames[:10]
target = colnames[10]

X = data[predictors]
Y = data[target]

x=data.iloc[:,7] #ShelveLoc
y=data.iloc[:,8] #Urban
z=data.iloc[:,9]#US

data.shape
from sklearn.preprocessing import LabelEncoder
from numpy import array


#Converting strings from shelveLoC to integer
values = array(x)
print(values)

label_encoder=LabelEncoder()
integerEncoded= label_encoder.fit_transform(values)
print(integerEncoded)

#Converting strings from urban to integer
values1 = array(y)
print(values1)

labelEncoder1=LabelEncoder()
integerEncoded1= labelEncoder1.fit_transform(values1)
print(integerEncoded1)

values2 = array(z)
print(values2)

labelEncoder2=LabelEncoder()
integerEncoded2= labelEncoder2.fit_transform(values2)
print(integerEncoded2)

#Dropping columns with strings 
data.drop(["ShelveLoc","Urban","US"],axis =1,inplace=True)


#Adding string converted to integer columns to the dataset
df=pd.DataFrame(data)
df['ShelveLoc']=integerEncoded
df['Urban']=integerEncoded1
df['US']=integerEncoded2

data=df
data.columns
data.corr()

data=data[['CompPrice', 'Income', 'Advertising', 'Population', 'Price','Age', 'Education', 'ShelveLoc', 'Urban', 'US','Sales']]
data.head()

labels = ["bad","good","best"]
bins = [0,7,12,19]
data_1 = pd.cut(data.iloc[:,10],labels =labels,bins = bins)
data.drop(["Sales",],axis=1,inplace=True)
data = pd.concat([data,data_1],axis =1)

X = data[predictors]
Y = data[target]

X.isnull().sum()
Y.isnull().sum()

#to find null values in index
Y.columns=["Sales"]
Y = pd.DataFrame(Y)
Y.loc[pd.isna(Y["Sales"]),:].index
#to change nan values with other values
Y.mode()#good
Y.loc[174]=("good")

Y.isnull().sum()


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

np.shape(data) # 400,11 => Shape 

#### Attributes that comes along with RandomForest function
rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # Sales (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_ 
rf.predict(X)
##############################

data['rf_pred'] = rf.predict(X)
cols = ['rf_pred','Sales']
data[cols].head()
data["Sales"]


from sklearn.metrics import confusion_matrix
confusion_matrix(data['Sales'],data['rf_pred']) # Confusion matrix

pd.crosstab(data['Sales'],data['rf_pred'])



print("Accuracy",(183+25+189)/(183+25+189+2+1)*100)

# Accuracy is 99.609375
data["rf_pred"]
