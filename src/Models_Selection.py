# Import the essential function
import sqlite3
import pandas as pd
import os

# Import the train-test, standard scaler function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the classification measurement function
from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score

# Import the Classification Machine Learning Models function
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier



# Import the created support function  

# - Data cleaning ,standardisation and Feature Engineering based on the EDA finding
from support import DataStandard

# - Measure selected Machine Learning Models on the essential classification metrics
from support import Models_Measurement

# - Show features importance as identified by the Machine Learning Models
from support import features

# For model dump so as to deploy
import joblib


#------------------------------------------------------------------------------------------------------
#1) Data Preparation 

#Ask user to input the filepath
path = input("Please confirm on the data path by typing out ../data/survive.db :" )
filepath = os.path.realpath(os.path.join(os.path.dirname(__file__), path))
print('\n\n\n')

#Connect to the Survive Database

sql_connect = sqlite3.connect(filepath)

# Save the SQL query string

query = "SELECT * FROM survive"

#Create a dataframe 

source = pd.read_sql(query,sql_connect)

#------------------------------------------------------------------------------------------------------
# 2) Data Cleaning,standardisation and Feature engineering using created support function

df = DataStandard(source)

print ('Data View\n')
print (df)

#------------------------------------------------------------------------------------------------------
# 3) Models Selection

#Split the data into Features(X) and Label(y) for classification modeling

X = df.drop('Survive',axis=1)
y = df['Survive']


#Split the Feature and Label data from Training and Testing 
#(random state of 101 was selected to maintain the consistency of the test data used)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Standard Scaling (avoid data leakage)

scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


#The model will be focused on achieving a high degree of recall as all revelant points when so that it can identify those who on higher risk in order to benefit from the application preemptive measures
# (The parameters of the model were obtained from the Classification Machine Learning with GridSearchCV.ipynb to optimise the recall rate and to save the runtime in this Model_selection)

# - Instantiate the Machine Learning(ML) Models with the known parameters

LR_ML = LogisticRegression(solver='saga',multi_class="ovr",max_iter=5000,penalty='none')
KNN_ML = KNeighborsClassifier(n_neighbors = 1)
SVC_ML = SVC(C=1, gamma='scale',decision_function_shape='ovo')
RT_ML = RandomForestClassifier(n_estimators =24)


# - Created list on the selected ML Models
models = (LR_ML,KNN_ML,SVC_ML,RT_ML)


# Classification Metric and Features Importance
Models_Measurement(models, scaled_X_train, y_train, scaled_X_test, y_test)

features(models,X,scaled_X_train, y_train)


# Model Deployment

print('RandomForestClassifier was chosen due to excellent peformance in all classification metrics - Recall,Precision,Accuracy Score')

# - Full training 
final_model = RandomForestClassifier(n_estimators=24,random_state=101)
final_model.fit(X,y)


# - Model Dump

# -- Train Model Dump
joblib.dump (final_model,'RFC_Final.pkl')
print ('RFC_Model Save')

# -- Clean set of X_Features DataframeDump

# --- Extract 1st row of the X_Features data
data = X[0:1]
# --- Reset all value to 0
for col in data.columns:
    data[col].values[:] = 0
data
# --- Rename it to clean
clean = data.copy()
# --- Dump the data
joblib.dump(clean,'cleandf.pkl')
print ('Cleandf Save')

