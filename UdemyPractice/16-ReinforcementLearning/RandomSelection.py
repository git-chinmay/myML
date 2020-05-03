"""
Randomoly selct the ads and get the reward if choose
We are not using any ML algorithm here.Its complete base python code.
We will calculate how much reward points we are getting through this approach.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

import random
N = 10000
d = 10
ads_selected = []
total_reward = 0

#Implementing Random Selection
for n in range(0,N):
    ad=random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward+reward
print(f"Total rewards {total_reward}")
#Visulising
plt.hist(ads_selected)
plt.title('Histogram of Ads Selection')
plt.xlabel('Ads')
plt.ylabel('No of times each ad selected')
plt.show()

