import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt

starttime = time.time()
dataset = pd.read_csv(r'C:\Users\n0278588\OneDrive - Liberty Mutual\ML\MLproject1\SummerisedDBTower_ClosedIncidents_2019.csv')


"""
Index(['Ticket #', 'Link', 'Assignee', 'Queue', 'Priority', 'MajorIncident',
       'Summary', 'UnplannedCheckbox', 'SLA', 'SLMStatus', 'Customer',
       'CustomerOrganization', 'CI', 'CIOwner', 'Submitted', 'Closed Date',
       'Count of TicketId'],
      dtype='object')
"""
#X = dataset.iloc[:,['Submitted','Customer','CI','Priority']].values
X = dataset.iloc[:,[14,10,12,4]].values
y= dataset.iloc[:,3].values




##One hot encoding
#Labelencoder
from sklearn.preprocessing import LabelEncoder
labelencoder =LabelEncoder()

#print(X[:,[0,1,2,3]])
X[:,0] = labelencoder.fit_transform(X[:,0])
X[:,1] = labelencoder.fit_transform(X[:,1])
X[:,2] = labelencoder.fit_transform(X[:,2])
X[:,3] = labelencoder.fit_transform(X[:,3])
y = labelencoder.fit_transform(y)




#X[:,[0,1,2,3]] = encoder.fit_transform(X[:,[0,1,2,3]])
##print(f'Lebel encoder X: {X[0]}')
#print(f'y After: {y[0]}')

#Puting one hot encoder on X
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
#print(f'One hot encoder X: {X[0]}')

#Datasplitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=0)

#Train the model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

#print(f"Predicted y_pred {y_pred[0:10]}")
#COnfusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(f"Confusion Matrx : {cm}")

#Calculating Accuracy from Confusion matrix
def accuracyCalc(cnf_matrix):

       

      FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
      FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
      TP = np.diag(cnf_matrix)
      TN = cnf_matrix.sum() - (FP + FN + TP)



      FP = FP.astype(float)
      FN = FN.astype(float)
      TP = TP.astype(float)
      TN = TN.astype(float)

      print(f"True Positve {TP}")
      print(f"False Positve {FP}")
      print(f"True Negative {TN}")
      print(f"False Negative {FN}")


      # Sensitivity, hit rate, recall, or true positive rate
      TPR = TP/(TP+FN)
      #print(f'Sensitivity, hit rate, recall, or true positive rate: {TPR}')
      # Specificity or true negative rate
      TNR = TN/(TN+FP) 
      #print(f'Specificity or true negative rate: {TNR}')
      # Precision or positive predictive value
      PPV = TP/(TP+FP)
      #print(f'Precision or positive predictive value: {PPV}')
      # Negative predictive value
      NPV = TN/(TN+FN)
      #print(f'Negative predictive value: {NPV}')
      # Fall out or false positive rate
      FPR = FP/(FP+TN)
      #print(f'Fall out or false positive rate: {FPR}')
      # False negative rate
      FNR = FN/(TP+FN)
      #print(f'False negative rate: {FNR}')
      # False discovery rate
      FDR = FP/(TP+FP)
      #print(f'False discovery raterate: {FDR}')

      # Overall accuracy
      ACC = (TP+TN)/(TP+FP+FN+TN)
      print(f'Overall accuracy: {ACC}')

accuracyCalc(cm)



##Random Prediction
#'Submitted','Customer','CI','Priority'

X_random = [["12/31/2019 0:00"],  ["Liberty BPPM"],  ["vmpip-5wa8xh2c"], ["P2NM"]]

X_random[0] = labelencoder.fit_transform(X[0])
X_random[1] = labelencoder.fit_transform(X[1])
X_random[2] = labelencoder.fit_transform(X[2])
X_random[3] = labelencoder.fit_transform(X[3])

X_random = onehotencoder.fit_transform(X_random).toarray()

y_Randompred = classifier.predict(X_random)
print(f"Group: {labelencoder.inverse_transform(y)}")

endtime = time.time()
print(f"Total Model execution time(Seconds): {endtime-starttime}")