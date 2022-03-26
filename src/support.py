
import pandas as pd
import numpy as np
import joblib


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,recall_score,precision_score,accuracy_score


#---------------------------------------------------------------------------------------------------------------------
# Support Function 

#Parameters Grid Search Function
#1) DataStandard
#2) ClassMeasure
#3) MLGridSearch

#Model Selection & Evaluation
#1) Models_Measurement
#2) features

# Model Deployment and Prediction
#1) NewDataTransform
#2) NewData

#-------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------
#Parameters Grid Search Function
#----------------------------------------------------------------------------------------------

def DataStandard(df):

    
        """
        Data Cleaning ,standardisation and Feature Enginnering

        """
    
        
        
        # Map to standardise the values in the identified features
        df['Survive'].replace(['No','Yes'],['0','1'],inplace=True)
        df['Smoke'].replace(['NO','YES'],['No','Yes'],inplace=True)
        df['Ejection Fraction'].replace(['L','N'],['Low','Normal'],inplace=True)

        # Change from string to numeric to facilitate the correlation measurement
        df['Survive'].replace(['0','1'],[0,1],inplace=True) 
        
        # Ensure the age are in absolute value
        df['Age']= abs(df['Age'])
        
        #One-hot encode on Categories Data
        X = df[['Smoke','Gender','Diabetes','Ejection Fraction']]
        X = pd.get_dummies(X)

        #Join the One hot encode dataframe
        df = df.join(X)
        
        #Drop Unnecessary column
        df = df.drop(['Favorite color','ID'],axis=1)
        
        #Drop Duplicate column
        df = df.drop(['Smoke','Gender','Diabetes','Ejection Fraction'],axis=1)
        
        #Drop Possible Nan Row
        df = df.dropna(subset=['Creatinine'],axis=0)

        return df


def ClassMeasure(y_test,y_pred):

    """
    Fit in the split test outcome (y_test) with the predicted outcome (y_pred) 

    It will then return the classification report and the confusion matrix

    """
    

    print('\nClassification report')
    print(classification_report(y_test,y_pred))
   
    print('\nConfusion Matrix')
    print(confusion_matrix(y_test,y_pred))
    #plot_confusion_matrix(Model,X_test,y_test)


def MLGridSearch (MLinstance,DefinedParam_grid,X_train,y_train):

    """
    Fit in the ML instance and the Defined Parameter (DefinedParam_grid) together with the choosen
    Scaled/Non-Scaled X_train , X_test and y_test. 

    This will be used to train with GridSearchCV based on recall measurement where it will generate a model declare as ML Grid

    Output =  MLGrid
    """

    import warnings
    warnings.filterwarnings("ignore")

    # - Instantiate  the ML Model

    ML =  MLinstance

    # User will define the parameters for DefinedParam_grid

    # - Instantiate the GridSearchCV Model

    Grid = GridSearchCV(ML,DefinedParam_grid,cv=5,scoring='recall')

    # Fit the training data to the defined GridSearchCV Model

    MLGrid = Grid.fit(X_train,y_train)

    return MLGrid

#----------------------------------------------------------------------------------------------
#Model Selection & Evaluation
#----------------------------------------------------------------------------------------------

def Models_Measurement (Models,X_train,y_train,X_test,y_test):

    """
    Fit single or multiple (in list format) Machine learning Model under the Models parameter with defined train & test split data

    The function will train the model and return classification related measurement as Recall,Precision,Accuracy,True Negative,False Positive,False Negative,True Positive

    """
    print("\n\nClassification Metric on the selected Machine Learning Models")
    print("-------------------------------------------------------------------")

    i=0
    score=[]

    for i in Models:

        ML = i

        ML.fit (X_train,y_train)

        y_pred = ML.predict(X_test)

        Recall = recall_score(y_test,y_pred)
        Precision = precision_score(y_test,y_pred)
        Accuracy = accuracy_score(y_test,y_pred)
        Matrix = confusion_matrix(y_test,y_pred)
        
        score.append((str(ML),Recall,Precision,Accuracy,Matrix[0][0],Matrix[0][1],Matrix[1][0],Matrix[1][1]))
    
    return print(pd.DataFrame(score,columns=['Model','Recall','Precision','Accuracy','TN','FP','FN','TP']))


def features(Models,X,X_train,y_train):

    """
    Fit single or multiple in list format Machine learning Model (Models) under the Models parameter with defined train & test split data (X_train,y_train) on the X.columns (X)

    The function will find the re;evant feature importance and sort it by the highest to lowest importanceS

    """
    print("\n\nFeatures Importance")
    print("-------------------------------------------------------------------")
    i=0
    for i in Models:

        ML = i
        ML = ML.fit(X_train,y_train)
        print('\n')
        
        try:
            Features_corr = ML.feature_importances_ 
            FeatureImp = pd.Series(index=X.columns,data=Features_corr).sort_values(ascending=False)
            print(str(ML))
            print(FeatureImp)
        except:
              try:
                  Features_corr = ML.coef_[0] 
                  FeatureImp = pd.Series(index=X.columns,data=Features_corr).sort_values(ascending=False)
                  print(str(ML))
                  print(FeatureImp)
              except:
                print(str(ML))
                print('No Features found')

#----------------------------------------------------------------------------------------------
# Model Deployment and Prediction
#----------------------------------------------------------------------------------------------

def NewDataTransform(data):
    
    """
    It is data mapping function where it will be call by NewData function
    New Data Mapping to the clean dataframe via cleandf.pkl where it resemblance the training data format.
    
     'Age','Sodium','Creatinine','Pletelets','Creatinine phosphokinase','Blood Pressure','Hemoglobin','Height',
     'Weight','Smoke_No','Smoke_Yes','Gender_Female','Gender_Male','Diabetes_Diabetes','Diabetes_Normal','Diabetes_Pre-diabetes',
     'Ejection Fraction_High','Ejection Fraction_Low','Ejection Fraction_Normal'
    
    """
    
    i=0
    df = []
    
    
    for i in range(data['Creatinine'].count()):
        clean = joblib.load('cleandf.pkl')
        clean['Age'] = np.array(data['Age'].values[i] )
        clean['Sodium'] = np.array(data['Sodium'].values[i]) 
        clean['Creatinine'] = np.array(data['Creatinine'].values[i]) 
        clean['Pletelets'] = np.array(data['Pletelets'].values[i])
        clean['Creatinine phosphokinase'] = np.array(data['Creatinine phosphokinase'].values[i])
        clean['Blood Pressure'] = np.array(data['Blood Pressure'].values[i])
        clean['Hemoglobin'] = np.array(data['Hemoglobin'].values[i])
        clean['Height'] = np.array(data['Height'].values[i])
        clean['Weight'] = np.array(data['Weight'].values[i])



        if data['Smoke'].iloc[i] == "Yes": 
            clean['Smoke_Yes'] = clean['Smoke_Yes'] +1
        else: 
            clean['Smoke_No'] = clean['Smoke_No']+1

        if data['Gender'].iloc[i] == "Male": 
            clean['Gender_Male'] = clean['Gender_Male'] +1
        else: 
            clean['Gender_Female'] = clean['Gender_Female']+1

        if data['Diabetes'].iloc[i] == "Diabetes": 
            clean['Diabetes_Diabetes'] = clean['Diabetes_Diabetes'] +1
        elif data['Diabetes'].iloc[i] == "Normal": 
            clean['Diabetes_Normal'] = clean['Diabetes_Normal']+1
        else:
            clean['Diabetes_Pre-diabetes'] = clean['Diabetes_Pre-diabetes']+1

        if data['Ejection Fraction'].iloc[i] == "High": 
            clean['Ejection Fraction_High'] = clean['Ejection Fraction_High'] +1

        elif data['Ejection Fraction'].iloc[i] == "Low": 
            clean['Ejection Fraction_Low'] = clean['Ejection Fraction_Low']+1
        else:
            clean['Ejection Fraction_Normal'] = clean['Ejection Fraction_Normal']+1

        df.append(clean)
           
    print(range(data['Creatinine'].count()))
    
    return  df


def NewData(data):
    
    """
    Load the desired data onto the function where it will return dataframe for the model to predict the survival rate
    
    """
    
    datapredict=NewDataTransform(data)

    df = pd.DataFrame(columns=list(datapredict[0].columns))
    for i in range(len(datapredict)):
    
        df = df.append(pd.DataFrame(datapredict[i].values[0].reshape(1,19),columns=list(datapredict[0].columns)))
        #print(df)
    
    return df
    
    








