import joblib
import pandas as pd
from support import NewData

Pathfilecsv = input("\n Enter the filename of your csv with it full path  : ")

df = pd.read_csv(Pathfilecsv)

new = NewData(df)

model = joblib.load('RFC_Final.pkl')

predict = model.predict(new)
print(predict)

#Create predicted data Dataframe 
Pred = pd.DataFrame(predict,columns=['Predicted Survival'])

#Join the Source data with the predicted data
Final = df.join(Pred)

#Generate a CSV file with both source and predicted data
Filename = input("\n Enter your desired csv filename : ")

Final.to_csv( Filename + '.csv')
print("\n{}.csv file generated and save\n".format(Filename))  
