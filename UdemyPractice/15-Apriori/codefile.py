"""
We are not going to use any specific package like we do scikitleanr
We will use the apyori.py file instaed.
This file will give us the RULES to our business problem.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#We dont have any header for this dataset so wehave to set accordingly
#Each row of the dataset representing a specific transcation done by a user in a week.
dataset = pd.read_csv(r'E:\VSCODE\GIT_Hub\myML\Practice\15-Apriori\Market_Basket_Optimisation.csv',header=None)

#Here dataset is a dataframe but aprioroi expects input in the foem of a big list contains multiple small lists
#Prepare the input data 
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#Training Apriori on datset
"""
Lets consider a product purchaesed minimum 3 times a day
we have datset for 1 week of trnsactions
so in one week that product purchased 3*7=21
minimum support = 21/total transactions = 21/7500 = 0.0028 ~ 0.003

In R there is defalt value of confident 0.8 means the rules has to be correct 80% of time 
In that case the algorithm will look for more obvious purchased product .Model will show combination not becaus ethey are assoviate better together but because ther are most purchased.
Ex:- In summer buy a lot of water and people like eggs much so they buy eggs too means they have combination of water and egg
It does not mean they purchaed often egg and water .
Here we will take 0.2 (20%) confidence

We can also try different value of lift and see.
Here we will trest with 3 
These value depends upon business values and datset.
"""
from apyori import apriori
rules= apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
result = list(rules)
#Rules we got here are bydeafult by their own relevance in python.we dont have to sort by lift
print(result[0])
"""
RelationRecord(items=frozenset({'light cream', 'chicken'}), support=0.004532728969470737, ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}), items_add=frozenset({'chicken'}), confidence=0.29059829059829057, lift=4.84395061728395)])

People who brought light cream are also purchase chicken
support = trnasctions who cntains chicken and cream/total no of trnsactions
confidence=0.29059829059829057, lift=4.84395061728395

People who buy light cream there is 29% of chance they will buy chicken.
"""