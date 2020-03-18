"""
The datasets contains 400 images taken at AT & T lab
"""

import sklearn as sk
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()
#print(faces)
print(faces.keys())

print(faces.images.shape) #(400,64,64) :- 400 images with 64X64 pixels matrices
print(faces.target.shape)
print(faces.data.shape) #(400,4096) :- 400 images but an array of 4096 pixels

##Checking input data normalisation
#print(np.max(faces.data)) #1.0
#print(np.min(faces.data)) #0.0
#print(np.mean(faces.data)) #0.5470426

def print_faces(images,target,top_n):
    #setup the figure size in inches
    fig = plt.figure(figsize=(12,12))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
    for i in range(top_n):
        #plotting the image in a matrix of 20X20
        p = fig.add_subplot(20,20,i+1,xticks=[],yticks=[])
        p.imshow(images[i],cmap=plt.cm.bone) 

        #Label the image with the target value
        p.text(0,14,str(target[i]))
        p.text(0,60,str(i))
        

print_faces(faces.images,faces.target,20)
#plt.show()

##Training the Support Vector machine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
svc_1 = SVC(kernel='linear')
X_train,X_test,y_train,y_test = train_test_split(faces.data,faces.target,
                                        test_size=0.25,random_state=0)

##Defining a function for K-fold cross validation
from sklearn.model_selection import cross_val_score,KFold
from scipy.stats import sem

def evalute_cross_validation(clf,X,y,K):
    #K-fold cross validation iterator
    cv= KFold(K,shuffle=True,random_state=0)
    scores = cross_val_score(clf,X,y,cv=cv)
    print(f"scores: {scores}")
    
    print(f"Mean Score: {(np.mean(scores)):.3f} (+/-{(sem(scores)):.3f})")

evalute_cross_validation(svc_1,X_train,y_train,50)



