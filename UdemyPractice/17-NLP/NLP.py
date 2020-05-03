"""Natural Language Processing
It's recomnded to use tab separated value not CSV.

dataset sample:
                                              Review  Liked
0                           Wow... Loved this place.      1
1                                 Crust is not good.      0
2          Not tasty and the texture was just nasty.      0

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#quoting will ignore the double quotes in dataset to avoind confusion
#Its not mandatory but recommonded
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting = 3)

#CLeaning the texts 
"""We will remove the unnecessery words like 'The' ,numbers,punctuations and 
stemings (Love,Loving and Loved all 3 are same so one is sufficient)"""
"""
import re
review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][0])

#[^a-zA-Z] :- I want to keep only letters
#' ' :- I dont want letters stick together
#dataset['Review'][0]: I want to apply the rule to 

review = review.lower() #Convert to lower case

import nltk
#DOwnload the list of words avilable in nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
review = review.split()

#Loop through review and keep only adds present not in stop words package
#review = [word for word in review if not word in (stopwords.words('english'))] #Set is faster than list
#op: ['wow', 'loved', 'place']

#Lets do the steming now
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in (stopwords.words('english'))]
#op: ['wow', 'love', 'place']

#Join back the words end of cleaing of process
review = ' '.join(review)"""

#Put above learning into a loop for all the lines of data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i]) 
    review = review.lower() #Convert to lower case
    review = review.split()
    review = [ps.stem(word) for word in review if not word in (stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)



#Creating the Bag of words Model
"""Take the first 1000 unique words and each word will become a column of a table 
and each row of the table will be the review
We will get a table of lot of zeros for columns and its a matrix and a matrix with lots of zero called Sparced matrix
Creating this Sparced matrix is called Bag of WOrds"""

#Tokenisation to get Sparced matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #max feature will allow to keeping only 1500 words in corpus
X = cv.fit_transform(corpus).toarray() #Converting X into Matrix
#print(X.shape) #(1000, 1565) with max_feature (1000, 1500)
y = dataset.iloc[:,1].values

#Now select one classification model
#Based on experience most commin model used are Naive Bayes,Random forest and Decison tree in NLP
#We will use Nive Bayes here

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.20,random_state=0)


#Model training
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,y_pred)
print(f"confusion matrix: {matrix}")
"""
[[55 42]
 [12 91]]

 55 :- COrrect Prediction positive reviews
 91 :- Correct Predictions Negative reviews"""
print(f"Accuracy: {((55+91)/200)*100}")

#Random Prediction
z=[[0]] * 1500
z[0] = ["worst","Bad"]


#X_rand = cv.fit_transform(z).toarray() 
print(classifier.predict(z))


if classifier.predict(X_rand)[0] == 1:
    print("Will buy the SUV.")
else:
    print("Will not buy the SUV.")





