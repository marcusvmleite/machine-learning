import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import Imputer

#Importing the dataset
dataset = pd.read_csv('Data.csv')

#Independent values
x = dataset.iloc[:, :-1].values 

#Dependent values
y = dataset.iloc[:, 3].values

#Taking care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)