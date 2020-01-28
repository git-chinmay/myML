"""Upper confidence Bound
   Ad 1  Ad 2  Ad 3  Ad 4  Ad 5  Ad 6  Ad 7  Ad 8  Ad 9  Ad 10        
0     1     0     0     0     1     0     0     0     1      0        
1     0     0     0     0     0     0     0     0     1      0 

first user only clicked Ad1 and Ad5
second user clicked only Ad 9
Reinforcement learning is called online learning as the current round
will be dispalyed based on the outcome of previosu round.

--There is no package for UCB.Need to write from scratch
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB
N = 10000
d = 10
#A big vector contains ads selcted in each round upto 100000
ads_selected = []
#step-1
numbers_of_selections = [0] * d #Vector of size d with 0
sums_of_rewards = [0] * d
total_reward = 0

#step-2
for n in range(0,N):
    max_upperbound = 0
    ad = 0
    for i in range(0,d):

        if (numbers_of_selections[i]>0):
            #This strtegy will applied after 10 first round
            #Otherwise for 1st 10 we will selct all ads one by one wach round

            average_reward = sums_of_rewards[i]/numbers_of_selections[i]

            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
            upper_bound= average_reward+delta_i

        else:
            #For 1st 10 rounds the upper bound will alyws will be 10 to power 400
            upper_bound = 1e400 # Selecting upper bound to very large value to 10 power of 400 for 1st 10 rounds

        if upper_bound > max_upperbound:
            max_upperbound = upper_bound
            ad = i #To keep track of ad each time we got a max upper bound

    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] += reward
    total_reward  += reward

#Visulising
plt.hist(ads_selected)
plt.title('Histogram of Ads Selection')
plt.xlabel('Ads')
plt.ylabel('No of times each ad selected')
plt.show()

print(f"Total Reward {total_reward}")
#for i in ads_selected:
#    print(i)



