"""
Artificail NN to solve a classification Probelm
Need to install Keras Library,Tensor flow and Theano
If you install Keras ,Tensorflow will run in background
"""
#Part 1 Data Preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1 = LabelEncoder() #will encode country
X[:,1] = labelencoder_x_1.fit_transform(X[:,1])

labelencoder_x_2 = LabelEncoder() #will encode Gender
X[:,2] = labelencoder_x_2.fit_transform(X[:,2])

#As country has 3 catagorical variable we will prepare the dummy for country
onehotencoder_x = OneHotEncoder(categorical_features=[1])
X= onehotencoder_x.fit_transform(X).toarray()
#Remove first dummy variable colulm to get rid of trap
X = X[:,1:] #Now we have only two dummy variable for country

#Split Data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

#Part 2 - Make the ANN

#Importing the keras library and 

"""
import keras #It will use TensorFlow in backend
from keras.models import Sequential     #Intialise the ANN
from keras.layers import Dense          #Creating layers in
Giving error: 
import tensorflow as tf
ModuleNotFoundError: No module named 'tensor
"""

from keras.models import Sequential
from keras.layers import Dense,Flatten

#Intiating the ANN
classifier = Sequential()

#Adding the input layer and first hidden layer
#Follow the 7 steps here
#relu = Rectifier activation function
#output_dim = NO f nodes in hidden layer
#input_dim = input 
#init = intilaises the weight randomly
classifier.add(Dense(output_dim=6, init="uniform",activation='relu',input_dim=11))

#Adding the second hidden layer(Actually for our probelm here we dont need 2nd layer)
#We dont need input layer here
classifier.add(Dense(output_dim=6, init="uniform",activation='relu'))

#Adding the output layer
#output_dim =1 as we have only one output node y
#activation = We are predicting the people leavig the abnk base on probablistic
#If more output catagores then choose softmax as activation
classifier.add(Dense(output_dim=1, init="uniform",activation='sigmoid'))

#Compiling the ANN (Applying Sochastic Gradient Descent)
#adam = one of the sochastic function popular one
#optimizer = selection of algorithm (we are going to use adam sochastic)
#loss = we are using logarithimic loss  as our output is sigmoid
#binary_crossentropy bcz our output is binary 0 or 1
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fit the ANN to training the set
#There is no thumb rule to seelct two extra parameretes values
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

#Part 3 -Making the prediction and evaluting the model

#Predicting teh Test set results
y_pred = classifier.predict(X_test) #this returns the probabilites
y_pred = (y_pred > 0.5) #If probability is graeter than 0.5 threshld then its True(Leave bank) 

#confusion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,y_pred)
print(f"confusion matrix: {matrix}")

input = [[619],	['France'],	['Male'],	[42],	[2],	[0.0],	[1],	[1],	[1],	[101348.88]]
#input = [619,	'France',	'Female',	42,	2,	0.0,	1,	1,	1,	101348.88]
input[1] = labelencoder_x_1.fit_transform(input[1])
input[2] = labelencoder_x_2.fit_transform(input[2])
#input = list(Flatten(input))
intput = [item for sublist in input for item in sublist]
input= onehotencoder_x.fit_transform(input).toarray()#Random prediction


