import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import data from the files
dataset = pd.read_csv('Concrete_database.csv')

#Data Preprocessing
#format as a dataframe
dataset = pd.DataFrame(dataset)#
#check for null values
dataset.isnull().sum()
#check for duplicates
dataset.duplicated().sum()
#check for data types
dataset.dtypes

print(dataset.head(5))

#how do i push this to github
