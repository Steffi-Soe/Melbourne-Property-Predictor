# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:18:56 2022

@author: steff
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\steff\OneDrive\Desktop\Project\Property Prediction\Melbourne_housing_FULL.csv')
df.head()

## Drop unrelated columns
df = df.drop(df.columns[[0, 1, 4, 5, 6, 8, 14, 15, 16, 17, 19]], axis = 1)


## FIll / Drop null values 
#Distance, Regionname, Price)'s rows are dropped indefinitely since their sum numbers are low
#And We can't predict price values since it's a dependent value 
df = df.dropna(subset = df.columns[[2, 8, 9]], axis = 0)

#Bedroom2
#median is used since the histogram leans more to the left side
df['Bedroom2'] = df['Bedroom2'].fillna(value = df['Bedroom2'].median())
# sns.histplot(data = df, x = df['Bedroom2'], bins = 20)
# plt.show()

#Bathroom
df['Bathroom'] = df['Bathroom'].fillna(value = df['Bathroom'].median())
# sns.histplot(data = df, x =df['Bathroom'], bins = 20)
# plt.show()

#Car
df['Car'] = df['Car'].fillna(value = df['Car'].median())
# sns.histplot(data = df, x = df['Car'], bins = 20)
# plt.show()

#Landsize
df['Landsize'] = df['Landsize'].fillna(value = df['Landsize'].median())
# sns.histplot(data = df, x = df['Landsize'], bins = 20)
# plt.show()

#BuildingArea will be dropped since 1/3 of the data is empty
df = df.drop(df.columns[[7]], axis = 1)


## Categorical to Numeric
df_trans = df.iloc[:, [1, 7]]
df_trans = pd.get_dummies(df_trans)

#Delete the Categorical data left in the table since the numeric version of them are available
df = df.drop(df.columns[[1, 7]], axis = 1) 

#Combine df_trans which has numeric values with the main table
df = pd.concat([df, df_trans], axis = 1)


## Overfitting, Underfitting, and Outlier solving
df = df[df['Landsize'] > 0]
df = df[df['Landsize'] < 2000]
# sns.histplot(data = df, x = df['Landsize'], bins = 50)

df= df[df['Bedroom2'] < 10]
# sns.histplot(data = df, x = df['Bedroom2'], bins = 10)

df = df[df['Bathroom'] < 9]
# sns.histplot(data = df, x = df['Bathroom'], bins = 20)

df = df[df['Car'] < 10]
# sns.histplot(data = df, x = df['Car'], bins = 20)


## Divide into independent and dependent
y = df.iloc[:, 6]
df = df.drop(df.columns[[6]], axis = 1)
x = df.copy()


## Train and Test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


## Scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()

sc.fit_transform(x_train)
sc.transform(x_test)


##Modeling
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

#Evaluation
from sklearn.metrics import r2_score
acc_score = r2_score(y_test, y_pred)



