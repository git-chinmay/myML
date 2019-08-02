"""
Testing various Supervised Learning ML models using a simple dataset
Practice link :- 
    https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2

"""

import pandas as pd
import matplotlib.pyplot as plt

fruit = pd.read_table(r"C:\Users\n0278588\GITHUB-Local\myML\Proj2-FruitSurvey-SimpleClassificationModels\InputDataSet.txt")
#print(fruit.head())
print(fruit.shape)
print(fruit['fruit_name'].unique())
print(fruit.groupby('fruit_name').size())

#The input data visulaisation 
import seaborn as sns 
#sns.countplot(fruit['fruit_name'],label = "Count")
#plt.show()

#Visualizing the input variable distribution

fruit.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), 
                                        title='Box Plot for each input variable')
#plt.savefig('fruits_box')
#plt.show()

#visualising same in Histogram 
import pylab as pl
fruit.drop('fruit_label',axis =1).hist(bins=30,figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('fruits_hist')
plt.show()