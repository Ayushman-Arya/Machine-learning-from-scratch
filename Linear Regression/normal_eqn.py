import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('Untitled spreadsheet - Sheet1.csv')
area = np.array(df['Area'].values)
area = (area - np.mean(area))/np.std(area)
price = np.array(df['Price'].values)

area = np.c_[np.ones(area.shape[0]), area]
theta = np.linalg.inv(area.T.dot(area)).dot(area.T).dot(price)

plt.plot(area[:, 1], price, 'o', label='Data points', color='red')
y = area.dot(theta)
plt.plot(area[:, 1], y, label='Regression line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Normal Equation')
plt.legend()
plt.show()