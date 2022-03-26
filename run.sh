
echo "Lim Yuan Hai Andrew"


echo "Classification Machine Learning Mode - Survival rate of coronary artery disease patientsy" 
echo "1 - Models Selection with Metrics & Model persistence"
echo "2 - Model Deployment"
read name

if [ $name = "1" ]; then
      cd "src"
      python3 Models_Selection.py
      cd ..
  elif [ $name = 2 ]; then	
       cd "src"
       python3 Survive_Prediction.py
       cd ..
       else
       echo "Please Select 1 or 2"
       fi



