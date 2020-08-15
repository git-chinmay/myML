"""
Can we beat UCB in Reward point ?
Can Thompson sampling will give different ad version compare to UBC ?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
N = 10000
d = 10
#A big vector contains ads selcted in each round upto 100000
ads_selected = []

#step-1
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

#step-2
for n in range(0,N):
    max_random = 0
    ad = 0
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1,numbers_of_rewards_0[i]+1) #beta destrubution

        if random_beta > max_random:
            max_random = random_beta
            ad = i #To keep track of ad each time we got a max upper bound

    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward  += reward


print(f"Total Reward {total_reward}")

#Visulising
plt.hist(ads_selected)
plt.title('Histogram of Ads Selection')
plt.xlabel('Ads')
plt.ylabel('No of times each ad selected')
plt.show()



