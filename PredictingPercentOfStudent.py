#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
style.use('ggplot')
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
df.head()
# Plotting the data
df.plot(x="Scores", y="Hours", style="o")
plt.title("Hours vs Pecentage")
plt.xlabel("Scores")
plt.ylabel("Hours")
plt.show()
# From the above plot, it is clearly visible that hours and scores have linear relationship. 
# So the further prediction could be done using 
# Linear Regression Algorithm
X = df.iloc[:, :-1].values  
y = df.iloc[:, 1].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = LinearRegression()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# # plotting the Linear Regression Model
line = clf.coef_*X+clf.intercept_
plt.scatter(X, y, color='g')
plt.plot(X, line)
plt.show
# # Predicting the Data
X_test
y_predict = clf.predict(X_test)
df1 = pd.DataFrame({'Actual' : y_test, 'Predicted': y_predict})
df1.head()
# # Checking the Accuracy of Prediction
clf.score(X_test, y_test)
# It is evidently visible that our prediction is 94.54% accurate.
# although it could be more accurate using different ML algorithms
# # Evaluating the Model
print("Mean Absolute Error is" , metrics.mean_absolute_error(y_test, y_predict))
