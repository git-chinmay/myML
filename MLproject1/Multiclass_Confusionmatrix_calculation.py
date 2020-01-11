import numpy as np


cnf_matrix = np.array([[13,  0,  0],
                       [ 0, 10,  6],
                       [ 0,  0,  9]])
"""
cnf_matrix=np.array([[ 238,    1,    2,    0,   41,   11,    0],
                     [   0,   25,    0,    0,    3,    1,    0],
                     [  21,    1,   32,    0,   17,    4,    0],
                     [   0,    0,    0,    7,    9,    3,    0],
                     [   7,    0,    0,    0, 3633,    8,    0],
                     [  44,    0,    4,    1,  397,  256,    1],
                     [   4,    0,    0,    0,    7,    2,    3]])"""

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)



FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

print(f"True Positve {TP}")
print(f"False Positve {FP}")
print(f"True Negative {TN}")
print(f"False Negative {FN}")


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print(f'Sensitivity, hit rate, recall, or true positive rate: {TPR}')
# Specificity or true negative rate
TNR = TN/(TN+FP) 
print(f'Specificity or true negative rate: {TNR}')
# Precision or positive predictive value
PPV = TP/(TP+FP)
print(f'Precision or positive predictive value: {PPV}')
# Negative predictive value
NPV = TN/(TN+FN)
print(f'Negative predictive value: {NPV}')
# Fall out or false positive rate
FPR = FP/(FP+TN)
print(f'Fall out or false positive rate: {FPR}')
# False negative rate
FNR = FN/(TP+FN)
print(f'False negative rate: {FNR}')
# False discovery rate
FDR = FP/(TP+FP)
print(f'False discovery raterate: {FDR}')

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print(f'Overall accuracy: {ACC}')