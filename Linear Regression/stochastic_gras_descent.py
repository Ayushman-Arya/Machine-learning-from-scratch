import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Untitled spreadsheet - Sheet1.csv')


area = np.array(df['Area'].values)
area = (area - np.mean(area))/np.std(area)
price = np.array(df['Price'].values)

alpha = float(input("Enter the value of alpha: "))
t0 = 0.0
t1 = 0.0
epochs = 1000

for epoch in range(epochs):
    indices = np.random.permutation(len(area))
    for idx in indices:
        x = area[idx]
        y = price[idx]

        # Prediction
        prediction = t0 + t1 * x

        # Error
        error = prediction - y

        # SGD update
        t0 = t0 - alpha * error
        t1 = t1 - alpha * error * x


print("The value of t0 is: ", t0)
print("The value of t1 is: ", t1)
y = t0 + t1 * area
plt.plot(area, price, 'o', label='Data points', color='red')
plt.plot(area, y, label='Regression line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Stochastic Gradient Descent')
plt.legend()
plt.show() 




