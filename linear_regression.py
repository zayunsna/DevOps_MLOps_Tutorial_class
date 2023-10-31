import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.DataFrame({'hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'score': [60, 63, 64, 67, 68, 71, 72, 75, 76, 78]})

#df = pd.read_csv("score.csv")

X = df.drop('score', axis = 1) # Feature select
Y = df['score']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
mse = mean_absolute_error(Y_test, Y_pred)

print('Mean Squared Error: ', mse)

future_data = pd.DataFrame({'hours': [11,12,13,14,15,16,17,18]})

predictions = model.predict(future_data)


print(predictions)

plt.plot(X,Y,'bs--',label='Data')
plt.plot(future_data['hours'], predictions,'r^', label='Predictions')
plt.xlabel('Hours')
plt.ylabel('Test Score')
plt.grid()
plt.legend()
plt.show()