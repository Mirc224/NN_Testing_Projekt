import numpy as np
from numpy import genfromtxt

# 0      1     2     3         4
# open, high, low, close    predict
my_data = genfromtxt("metat_data.csv", delimiter=";")

print(my_data.shape)
NPones = np.ones(my_data.shape[0])
#my_data[:,4]
difference = np.subtract(my_data[:,4], my_data[:,3])
AAPE = np.arctan(np.abs(np.divide(difference[:], my_data[:,4])))
AAPEsum = np.sum(AAPE)

result = float((AAPEsum/ my_data.shape[0])*100)
print(str(result) + "%")