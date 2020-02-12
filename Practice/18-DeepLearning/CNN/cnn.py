# -*- coding: utf-8 -*-
"""
Convolutinal Neuralnetworks
Dataset Details:- 
We have 10000 photos of dogs and cats
out of 10k , we split 8k for train and 2k for test sets
Out of 8k , we have 4k each for DOg and Cat similarly in test out of 2k we have
1k each for Dog and Cat.

So above split is our data preprocessing.We dont need any other traditional method of slitting.

"""
#PART-1 Building CNN 

#Importing the Keras libraries and packages
from keras.models import Sequential
#from keras.layers.convolutional import Conv2D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#Intialise the CNN
classifier = Sequential()

#Step-1 Convolution (COnvolution layer is collection of Feature Maps)
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu')) #32 filters each of 3X3 dimensions

"""
Warning:
    classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
__main__:4: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), 
input_shape=(64, 64, 3..., activation="relu")`"""

#STep-2 Pooling step(Gives new feature map with reduced size)
classifier.add(MaxPooling2D(pool_size=(2,2)))

#In first run accuracy was 76%,hence adding another layer for more accuracy
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step-3 Flattening
classifier.add(Flatten())

#Step-4 Fully Connected (Creating a classic ANN)
#Classic ANN is a great classifier for classification of non linear problem
classifier.add(Dense(output_dim = 128,activation='relu'))
"""
Warning:
    128,activation='relu'))
__main__:1: UserWarning: Update your `Dense`
 call to the Keras 2 API: `Dense(activation="relu", units=128)`"""

#Output layer
classifier.add(Dense(output_dim = 1,activation='sigmoid'))

#Compailing the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#PART-2 Fitting th CNN to the Images
from keras.preprocessing.image import ImageDataGenerator

"""
Copy paste the below code from Keras documentation and change as per need.
We are doing this for Image Augmentatioan"""

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'E:\VSCODE\GIT_Hub\myML\Practice\18-DeepLearning\CNN\dataset\training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'E:\VSCODE\GIT_Hub\myML\Practice\18-DeepLearning\CNN\dataset\test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)

"""
`steps_per_epoch` is the number of batches to draw from the generator 
at each epoch.

Basically steps_per_epoch = samples_per_epoch/batch_size. 
Here steps_per_epoch = 250

Similarly `nb_val_samples`->`validation_steps` 




"""

#classifier.predict_classes()






