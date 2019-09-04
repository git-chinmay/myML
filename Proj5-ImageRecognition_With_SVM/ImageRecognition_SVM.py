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

