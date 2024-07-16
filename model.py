import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

t = pd.date_range('2019Q1', '2023Q3', freq='Q')
X = pd.read_excel("./ans.xlsx")
y1p = np.array(X.values[:, 1])
y2p = np.array(X.values[:, 2])
y3p = np.array(X.values[:, 3])
y1 = np.array(X.values[:, 4])
y2 = np.array(X.values[:, 5])
y3 = np.array(X.values[:, 6])
y1_loss = np.sum(np.abs(y1p - y1))
y2_loss = np.sum(np.abs(y2p - y2))
y3_loss = np.sum(np.abs(y3p - y3))
print("y1_loss:", y1_loss, "y2_loss:", y2_loss, "y3_loss:", y3_loss)
plt.plot(t, y3p, label='predict')
plt.plot(t, y3, label='reality')
plt.legend()
plt.show()
