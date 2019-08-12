"""
PIMA Indian Diabetics datastes
Datasource :- Kaggle
DT :- 12th Aug 2019

Context: This dataset is originally from the National Institute of Diabetes and Digestive 
and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or 
not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
Several constraints were placed on the selection of these instances from a larger database.
 In particular, all patients here are females at least 21 years old of Pima Indian heritage.

***************************
Model accuracy is: 80.73%
Model precision is: 76.60%
Model Recall is: 58.06%
***************************
"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  #To avoid future warnings in output


data_input = pd.read_csv("diabetes.csv")


#['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',  'Outcome']

feature_columns =  ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
X = data_input[feature_columns]
y = data_input.Outcome

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)


LGR = LogisticRegression()

LGR_Train = LGR.fit(X_train,y_train)
y_predict = LGR_Train.predict(X_test)


#Evaluting the model using confusion matrix
model_eval = metrics.confusion_matrix(y_test,y_predict)
#print(model_eval)

#CHecking the Models performance parameters(Confisuion matrix metrcs)
accuracy = metrics.accuracy_score(y_test,y_predict)
precision = metrics.precision_score(y_test,y_predict)
recall = metrics.recall_score(y_test,y_predict)

print(f'Model accuracy is: {(accuracy*100):.2f}%')
print(f'Model precision is: {(precision*100):.2f}%')
print(f'Model Recall is: {(recall*100):.2f}%')

##Single value prediction
"""
Pregnancies = 1, Glucose=0, 'BloodPressure=42.9, 'SkinThickness=22, 'Insulin=199, 'BMI=76, 'DiabetesPedigreeFunction=1.394,

row no = 661

1 = Diabetics 
0 = Non Diabetics
"""
single_pred = LGR_Train.predict([[1,0,80,1,1000,76,1.394]])
if single_pred == 1:
    print("\nDiabetics patients")
else:
    print("\nNon Diabetics patients")
#print((data_input.Outcome).iloc[661])










