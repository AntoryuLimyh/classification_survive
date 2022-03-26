


## Key findings from EDA

The dataset has 15000 rows with missing 499 Creatinine data and the potential of 958 duplicated data,

It is also discovered that Survive attribute has mixed of 0,1,No and Yes ,the Ejection Fraction attribute  has mixed of Low,High,Normal,L,N and Smoke attribute has a mixed Yes,No,YES,NO. This will be proven disruptive for the data analysis if not standardize.

The subsequent feature engineering process is One-hot encoding on those attributes that are in categorical format like Smoke,Gender,Diabetes,Ejection Fraction so as to facilitate that data exploration.

On dealing with the 499 Creatinine data, a quick correlation check indicated the  Creatinine  attribute has positive correlationship of 0.30 with the desired Survive attributes which is our label. It is also discovered all the missing Creatinine data has 0 survivors where the original dataset has 10185 with the 0 attached to the survive attributes.
Hence it is decided it not worth the risk and effort to try to fill the missing Creatinine data due to its high correlation and there is sufficient data with 0 as Survive attributes left by dropping it. The decision then is drop the 499 affected rows.

As for the duplicated ID through investigation as shown in the EDA.ipynb. Those that are supposed to be duplicated carry different data in all its attributes. Hence, it will be save to deem that the possibility here is due to data entry error on the ID and the ID attributes itself does not seems to be important features to keep logical then it attach data in other attributes. Hence the decision then is to keep it.

A quick check on the possible outlier with describe() found that there are data found below age 0 which is not even possible. A quick sampling check was done on all those affected rows and it is found that this could be another instance of data entry error. Hence all the age that has a negative sign attached was convert back to normal age.

Moving on the data exploration and analysis, it is found that the attribute like Weight,Age, Creatinine and the Erection Fraction has a significant impact on the survival rate.

Particularly, if the weight is heavier, age is older , the creatinine level is below 2 and the Ejection Fraction is low. The survival rate increase.


## Build and Execution Flow and Finding



![Build_Flow](/src/Build_FlowChart.png)


### Folder Structure
___
1. src

	1.1.1 Model_Selection.py

	1.1.2 Survive_Prediction.py

	1.2 support.py

	1.3 Classification Machine Learning with Grid Search.ipynb

	1.4 SampleData

		1.4.1 SampleData.csv

		1.4.2 SampleData_with_Results.csv
		
	1.5 Build_FlowChart.png

2. data

	 2.1 survive.db

3. README.md

4. eda.ipynb

5. equirements.txt

6. run.sh

### Execution Flow (Instruction to operate)

Run the run.sh file assume that requirement.txt is install (`pip install -r requirement.txt`) where you be presented with option as :

1 - Models Selection with Metrics & Model persistence
2 - Model Deployment

if **1** is selected, it will prompt the user to indicate the full path and database (.db) as the dataframe. 
For example
` ../data/survive.db`

It will then display the all the models scoring metric, features importance of model and subsequently create the 
Final model and Clean Feature dataframe dump.

if **2** is chosen, it will prompt the user to indicate the full path and file name of csv file. A SampleData file was provide.
Hence for example `SampleData.csv`

It will do the necessary and return the prediction as 0 and 1 format (patient survives: 0 = No , 1 = Yes). Subsequently, it will prompt the user to enter a name reference for it save the original dataframe with new column as Predicted Survival.



### Build Flow and Metric Consideration
___
It begin with Data cleaning and standardisation and exploration in the inital phrase of handling of data and data analysis.

With that understanding, the importance is placed on the Recall scoring metric as it is crucial in our prediction the survival of coronary artery disease patients. 

#### - Metric Consideration - Recall
The basis to that is we are living in the world where resources could be a constraint and speed matter when it come saving or losing a life.

Hence any preemptive medical treatments used on a False negative (FN) patient, where this is the category of patient that is determine to have lower chances of survival with the current medical treatment where he/she based on the past data could have higher chance of survivor over those False positive (FP) patient, where they are the actual one that need the treatment but model determine that they can probably wait or continue with the current practice due to the predicted higher probability of survivorship. In any case of limited resources, they tend to be one who lose out due the model predictive ability if deployed.

##### - Classification Machine Learning with Grid Search
With that being explained, I have developed Classification Machine Learning with Grid Search.ipynb where it will leverage on the ability of the GridSearchCV on the basis of finding the maximum recall score from Sklearn library to find the optimise parameters for the selected machine learning model namely:
 1. Logistic Regression
 2. K Nearest Neighbor
 3. Support Vector Classifier
 4. Random Tree Classifier

Where the idenified parameters to achieve the highest possible for the abovementioned models were :

LogisticRegression - solver='saga', multi_class='ovr',penalty='none'
K Nearest Neighbor - n_neighbors = 1
Support Vector Classifier - C=1, gamma='scale' (Standard from SVC.sklearn)
RandomForestClassifier - n_estimators =24

This is done by creating the **DataStandard function** from the knowledge in the initial EDA phrase and import it from consolidated  created **support python library (Support.py)** where it is use to clean,standardise and apply one-hot encoding on the categorical attribute of raw data from survive.db. The  output of this function is clean data frame suitable for the machine learning training.

Where the clean dataframe was then split into X and y datasets. By dropping the columns 'survive' in dataset to come with X, where it will formulate as the feature dataset and use the sole 'survive' column as y, where this is the label dataset 

Using sklearn.train_test_split, both X and y was split into ratio 70% training :30% test. The split train and test data will further fit into standard scaler so as to standardise and convert all values to between 0 and 1 with just a specific set of train and test set to avoid any possibility of data leakage.

Further to it, a **MLGridSearch Function** was created where it will instantiate the machine learning model and use the user defined parameters grid with the recall scoring as measurement to find the optimize parameters for the selected machine learning algorithm.
The output of this function is a fit and train Machine learning model for prediction purpose.

##### - Model Selection
By importing the DataStandard function as explained the need for it and the same process of dealing with the train,test and scale the imported raw data.

Instantiate the  selected Machine Learning(ML) Models with the optimize parameters as provided by the MLGridSearch as per below:

LR_ML = LogisticRegression(solver='saga',multi_class="ovr",max_iter=5000,penalty='none')
KNN_ML = KNeighborsClassifier(n_neighbors = 1)
SVC_ML = SVC(C=1, gamma='scale',decision_function_shape='ovo')
RT_ML = RandomForestClassifier(n_estimators =24)

All of the abovementioned was then packed into a list.

The list with defined scaled X_train,y_train,scaled_X_test and y_test data was then load into the created **Model_Measurement function**, where it function is produce the recall,precision,accuracy and the parameters of the confusion matrix to give overview evaluation on the selected Machine learning model. The result were as per below :

|Model |Recall |Precision | Accuracy    | TN   | FP     | FN    | TP   |
|---|---|---|---|---|---|---|---|
|  LR_ML | 0.661258    | 0.776807    | 0.820271    | 2591    |  281   | 501     | 978  |
|  KNN_ML  | 0.994591  |  0.993248 |  0.995863 |  2862 |   10| 8  |  1471 |
| SVC_ML    |  0.843813 |  0.954858 | 0.933349  |  2813 |  59 |   231| 1248  |
| RT_ML    |  1.000000 | 1.000000  | 1.000000  |  2872 |  0 |   0|1479   |

- Analysis 
With expection to the Logistics regression model (LR_ML), all the other models are able to achieve a high Accuracy and Precision scores. However, when it come down to recall metric which is our focus, Random Tree Classifier (RT_ML) and K Nearest Neigbour (KNN_ML) model stand out for the rest, particularly the RT_ML as shown above.

In order to ascertain whether the features found in the EDA process were the same as determined by the Machine learning model. The created **Features Function** was called to load the models with defined scaled X_train,y_train.

The Logistic Regression has it top importance by order as Weight,Creatinine,Age,Ejection Fraction_Low,Creatinine phosphokinase,Gender_Female,Smoke_Yes,Diabetes_Pre-diabetes

While the Random Tree classifier has it top importance as Creatinine,Weight,Creatinine phosphokinase,Pletelets,Sodium ,Age.

Since Random Tree classifier has the best scoring metrics, it is surprising to bagging function of it, it did managed to found new features such as Creatinine phosphokinase,Pletelets,Sodium that played a huge part in the classification model.

As consistency with EDA finding, Weight and Creatinine were still the top features in coming to this classification prediction.

Model Dump
___
Random Forest Classifier is eventually selected due to it excellent metric score and the explaination provided in the section of Choice of Models for each Machine Learning.

The model was then retrain with the whole X (Features) and y (Label) dataset and dump it as RFC_Final.pkl for Model Deployment.

At the same time, a clean set of X(Features) is created and dump as Cleandf.pkl for template purposes.


##### - Model Deployment - Survive_Prediction
It started by asking for the user to indicate the path and its filename (csv format).

The file was then subsequently load and **NewData function** was called upon.

The function work by calling the **NewDataTransform function** where it will map the standard format as the raw data format to the X(Features) data format and then subsequent return the whole dataframe.

The RFC_Final.pkl was then load and use the new dataframe to predict the supposed label.

Finally, it will then ask the user to save the file down as csv format with its relevant path.

___









## Choice of Models for each Machine Learning

The objective is to predict the survival of coronary artery disease patients using the dataset so as to help formulate preemptive medical treatments, which is definitely aim to improve the survivorship of any concerned patient.

It was clear that this is binary classification problem where there is never in-between ground in it and require a discrete value as its output at the end of it.

Hence the Models consideration will revolve around those there are suitable to handle the classification issue.


### Models Considerations
___
##### - Logistic Regression 
Logistic regression is the go-to method for binary classification problems (problems with two class values). It use the sigmoid activation function that does the classification task with great efficiency by adopting a probabilistic approach towards decision making and ranges in between 0 to 1.

The Grid Search CV  with the optimize parameter chosen -  **penalty as none** and  **solver as saga** where it is suitable for multi features  as it is able to handle the multinomial loss. The saga option also allow the algorithm to train faster which is essential here given that time can be essential when it come to any preemptive measure administer to the required patients.

The optimize parameter chosen for the **multiclass parameter is ovr**, which make sense when it come binary problem consideration.
___
Result Metric
___

  | Metrics|  Score|
|--|--|
| Recall  |  0.661258|
| Precision | 0.776807
| Accuracy| 0.820271 |


 | Confusion Matrix|  Count|
|--|--|
| TN|  2591|
| FP| 281
| FN| 501
| TP| 978
  
This is an extremely risker algorithm to use in the context of survival prediction where resources (medical treatment) used for the 501 patients (False negative) who might have a higher survival rate based on the current condition will take precede on those 281 patients (False positive) who might really need it. Whenever, it comes to medical condition, speed save life whenever it is properly consider.
  
##### - K-Nearest Neighbors (KNN)

The KNN algorithm work on the assumption that similar things come together in close proximity. In this instance, it is work on assumption that similar attributes tend to group around the label (survive). Hence it work on classify the label value by identify the distance of that train label between the attributes and hence come to the conclusion with the learned information to predict the future label with the given attributes. 

The pros of this are that it work on the simple concept which is a new label is classified by looking at the nearest classified attributes' distance (K-nearest) and it is easy to implement. It also do not require additional assumptions and rely heavy on the K-nearest metric.

On the hand, the known con was that it could can slower with volume of data, which make it hard in the area where the prediction need to be make rapidly.


Per the training done via the Grid Search CV model with recall as scoring metrics, the identified optimize parameters is **n_neighbors as 1**. It represent a sole neighbors that will vote for the class of the label yet it is able achieve the following results :
___
Result Metric
___
  | Metrics|  Score|
|--|--|
| Recall  |  0.994591 |
| Precision | 0.993248  
| Accuracy| 0.995863 |


 | Confusion Matrix|  Count|
|--|--|
| TN|  2862|
| FP| 10
| FN| 8
| TP| 1471


Second highest scoring model on recall score of 0.9945 among the other models in the relevant measurement. This fantastic result in my opinion based on sole vote in my opinion is too close to perfect.

The risk arise when faced with resources constraint on medical treatment that 8 patients (False negative) who might have a higher survival rate based on the current condition might be prioritize for treatment against those 10 patients (False positive) who might really need it.

##### - Support Vector Classifier (SVC)
Support Vector Machine (SVM) commonly used in classification problems based on the idea of finding a hyperplane that best divides a dataset into multiple attributes.

The optimize decision function selected by the Grid Search based on the relevant data attributes is **One-vs-One (OVO )** , a heuristic method for using binary classification algorithms for multi-class classification. It work on approach splits the dataset into one dataset for each class versus every other class.

The pros of this are that it work works relatively well when there is a clear margin of separation between classes where it is effective in cases where the number of dimensions is greater than the number of samples. It is also an relatively memory efficient algorithms to work with

The cons were support vector classifier works by putting data points around the classifying hyperplane where there is no probable explanation for the classification. Similar to KNN, it also do not work well with large datasets.
___
Result Metric
___
  | Metrics|  Score|
|--|--|
| Recall  |  0.843813|
| Precision | 0.954858
| Accuracy| 0.933349|


 | Confusion Matrix|  Count|
|--|--|
| TN|  2813|
| FP| 59
| FN| 231
| TP| 1248

The SVC classification metric on the current set of data suggested  that there were 231 patients (False negative) who might have a higher survival rate based on the current condition might be prioritize for treatment against those 59 patients (False positive) who might really need it where this not feasible in view of any resources constraint.
The recall score is also below 0.9 mark at 0.84 where recall score as mentioned is a crucial measurement based on the objective of this machine learning modelling


##### - Random Tree Classifier (RT)
Random tree (RT) classifier  is an algorithm that use multiples of  individual decision trees that operate as an ensemble ,a process that use multiple learning algorithms to obtain predictive performance. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction.

The usual decision tree algorithm has a flaw that it is sensitive to the data that they are trained on where any slight change of the training set might result a different tree structure deployed. RT resolved this via the bagging process where it allow the tree to randomly sample from the dataset.

It is also remove any pre-built bias by having each tree in the RT will pick a random subset of the features hence adding more variation among the trees, where it serve to add lower correlation and more diversification in the trees.

The Grid Search CV suggested the optimize parameters to use **n_estimators of 24**.  It represents that 24 decision trees was deployed in the current dataset on the given features.
___
Result Metric
___
 | Metrics|  Score|
|--|--|
| Recall  |  1|
| Precision | 1
| Accuracy| 1|


 | Confusion Matrix|  Count|
|--|--|
| TN|  2872|
| FP| 0
| FN| 0
| TP| 1479

This is the perfect score possible on the dataset. With minimal/no instances of misclassification of the patient survival probability.

The pro of the RTC is that it can handle non-linear parameters efficiently which is extremely common in the any dataset and it is robust to any possible outliers. 

While the con were that it can be complex and required a longer training duration when more trees are require to formulate the classification.

####  Conclusion
___
After going through the considerations and the scoring metric of the various Machine Learning model, the selection of Random Tree Classifier (RTC) should be an obvious choice among all.

Beside the perfect scoring metric where it will render the consideration of logistics regression pale in comparison, it is extremely important not introduce any bias in the medical situation in the features selection where RTC is definitely neutral in that with its known feature randomness ability.

As well the parameters of 24 trees much lesser than the default of 100 based on the sklearn.RTC as suggested by the Grid Search CV process greatly reduce its con on the issue of complexity and training duration.

It is definitely able to overcome the issue of dealing with large amount of data where it is known limitation of the KNN and SVC model. This is crucial for more future additional of data to further fine tune its respective parameters.



## Other consideration for deployment of the model 

The other consideration for the deployment of the Random forest classifier (RFC) is that the model does look perfect and it might probably need continuous data to train it so as to maintain its relevant to the cause.

RFC was choosen for it ability to deal with huge amount of data which we can be expecting in our case with a start of 15000 dataset to begin.

It is no doubt that the bagging and feature randomness attribute as mentioned as the pro of it. will assist to remove any basis but only rely gini impurity process to determine what are the real factors that matters.

   
## Update

1)	Stratified sampling

The data after dropping the missing values as mentioned in the eda.ipynb is 14501 row. This is split between 9686 rows of non-survival (0) and 4815 rows of survival (1), which came to approximately 67% and 33% respectively.

This is the typical case of not having balanced number of examples for each class label. Hence, it is desirable to split the dataset into train and test sets in order to preserves the same proportions of examples in each class as observed in the original dataset.

With the abovementioned, the changes was reflected by adding ``stratify=y`` to the Splitting the Feature and Label data from Training and Testing annotation found in the **Classification Machine Learning with Grid Search.ipynb** 

``X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101,stratify=y)``

2) Drop the first column

The **DataStandard function** found in the **Support.py** was built on the knowledge from the explatory data analysis (eda.ipynb).

Within it has a line on  to deal with the Ordinal Data that was needed for the classification purposes.

``X = df[['Smoke','Gender','Diabetes','Ejection Fraction']]``
``X = pd.get_dummies(X,)``

This original code might create potential issue for the classification modelling - **_Multicollinearity_**

It tend to occur when features (input) are highly correlated with one or more of the other features in the dataset. It affects the performance of regression and classification models.

With this in mind, the ``drop_first=True`` was introduce to the revised line in the **DataStandard function**

``X = df[['Smoke','Gender','Diabetes','Ejection Fraction']]``
``X = pd.get_dummies(X,drop_first=True)``

Along with this revision, there were also related changes made to the **NewDataTransform function** in the same support,py.

``if  data['Smoke'].iloc[i] == "Yes":``
``clean['Smoke_Yes'] = clean['Smoke_Yes'] +1``
``else:``
``clean['Smoke_Yes'] = 0``
``if  data['Gender'].iloc[i] == "Male":``
``clean['Gender_Male'] = clean['Gender_Male'] +1``
``else:``
``clean['Gender_Male'] = 0``
``if  data['Diabetes'].iloc[i] == "Normal":``
``clean['Diabetes_Normal'] = clean['Diabetes_Normal']+1``
``elif  data['Diabetes'].iloc[i] == "Pre-diabetes":``
``clean['Diabetes_Pre-diabetes'] = clean['Diabetes_Pre-diabetes']+1``
``else:``
``clean['Diabetes_Pre-diabetes'] = 0``
``if  data['Ejection Fraction'].iloc[i] == "Normal":``
``clean['Ejection Fraction_Normal'] = clean['Ejection Fraction_Normal']+1``
``elif  data['Ejection Fraction'].iloc[i] == "Low":``
``clean['Ejection Fraction_Low'] = clean['Ejection Fraction_Low']+1``
``else:``
``clean['Ejection Fraction_Low'] = 0``

3)  Enhance the scoring strategy

As mentioned above, recall score is the one important parameters when it comes to survival issue. You will rather have a wrong labelling on survivor as non-survivor than missing up the potential of non-survivor case.

With new introduction to further enhance or define the scoring parameter for the original Gridsearch process found in the **MLGridSearch function** under the support,py - maker_scorer from Sklearn.

The `make_scorer` function takes two arguments: the function you want to transform, and a statement about whether you want to maximize the score (like accuracy,recall,precision) or minimize it (like MSE or MAE). In the standard implementation, it is assumed that the a higher score is better. 

In our case here where recall is concern, the default setting of  **greater_is_better** bool, default=True fit is a good fit for our requirement.

The changes was reflected in the following lines:

``from  sklearn.metrics  import  make_scorer``

``recallscore = make_scorer(recall_score)``

``Grid = GridSearchCV(ML,DefinedParam_grid,cv=5,scoring=recallscore)``

4) Kfold and Cross Validation

By usual split on the training and test set in our case (70-30 split), It could potential leave us with a small test set. Whether the split is a good indicative will remain unknown to us and, such testing has a potential inherent risk of getting any performance on said set only due to chance. 

To overcome this, the adoption of the cross-validation will be crucial check for our this case. By building K different models, we are able to make predictions on  **all** of our data.

We are now can train our model on all our data because if our 4 models had similar performance using different train sets, we assume that by training it on all the data will get similar performance

Withthe abovementioned, the following was added in the **Classification Machine Learning with Grid Search.ipynb** to get the recall score of the model with its maximum hyperparameters setting obtained from the GridSearch process.


Cross Validation Evaluation on the ML model

Define the scoring strategy for cross validation
``recallscore = make_scorer(recall_score)``

Standard Scaling the X data (In line with GridSearch scaled training set)
``scaled_X= scaler.fit_transform(X)``

Define the cross validation function with its selected output measurement 

    def cross_val(ML):

		# prepare the cross-validation procedure
		cv = KFold(n_splits=5, random_state=1, shuffle=True)


		# evaluate model
		scores = cross_val_score(ML, scaled_X, y, scoring=recallscore, cv=cv, n_jobs=-1)
		
		# report performance
		print('ML model :',ML.best_estimator_)
		print('\nScore',scores )
		print('\nRecall (average & Standard deviation): %.3f (%.3f)' % (scores.mean(), scores.std()))


The results on the recall metrics were as followed :

|Model |Average |Standard Deviation| 
|---|---|---|---|---|---|---|---|
|  LR_ML | 0.670    | 0.014| 
|  KNN_ML  | 0.997|  0.001|  
| SVC_ML    |  0.888|  0.010| 
| RT_ML    |  1.000000 | 0.000|  
  

This still in line that the Random Forest Classifier are still the best model to predict the survivor rate on the given data features with almost perfect recall average and the negligence standard deviation with the cross validation measurement.