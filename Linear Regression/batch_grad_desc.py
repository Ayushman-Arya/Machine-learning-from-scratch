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
cost_history = []
for i in range(1000):
    sum0 =0
    sum1 = 0  
    cost = 0
    for j in range(len(area)):
        prediction = t0 + t1 * area[j]
        error = (prediction - price[j])
        sum0 = sum0 + error
        sum1 = sum1 + error * area[j]
        cost = cost + (error ** 2)
    cost = cost / (2 * len(area))
    cost_history.append(cost)
    t0 = t0 - (alpha * sum0/len(area))
    t1 = t1 - (alpha * sum1/len(area))
    

print("The value of t0 is: ", t0)
print("The value of t1 is: ", t1)
y = t0 + t1*area
plt.plot(area, price, 'o', label='Data points', color='red')
plt.plot(area, y, label='Regression line')
plt.plot(range(1000), cost_history, label='Cost function', color='green')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Batch Gradient Descent')
plt.legend()
plt.show() 




