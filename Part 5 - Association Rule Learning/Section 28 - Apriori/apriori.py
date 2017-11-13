# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
dataset.fillna("", inplace=True)
transactions = dataset.values.tolist()


#[[str(j) for j in i] for i in transactions]

#Training Apriori on the dataset
from apyori import apriori
#min_support = 3*7/7500=0.0028~0.003
#min_confidence = 20% = 0.2
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualizing the results
results = list(rules)

myresults = [list(x) for x in results]
myRes = []
for j in range(0,153):
    myRes.append([list(x) for x in myresults[j][2]])