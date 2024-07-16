import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


X = pd.read_excel("./data.xlsx")
Y = pd.read_excel("./month.xlsx")
time_M = pd.date_range(start='2009-1', end='2023-8', freq='M')
time_Q = pd.date_range(start='2009Q1', end='2023Q3', freq='Q')
GDP1 = np.array(X.values[:, 2])
GDP2 = np.array(X.values[:, 3])
GDP3 = np.array(X.values[:, 4])
CPI = np.array(X.values[:, 5]).astype('float')
Meat = np.array(X.values[:, 6]).astype('float')
Farm = np.array(X.values[:, 7]).astype('float')
PPI = np.array(X.values[:, 8]).astype('float')
Elec = np.array(X.values[:, 9]).astype('float')
PMI2 = np.array(X.values[:, 10]).astype('float')
PMI2_new = np.array(X.values[:, 11]).astype('float')
ISP = np.array(X.values[:, 12]).astype('float')
PMI3 = np.array(X.values[:, 13]).astype('float')
Travel = np.array(X.values[:, 14]).astype('float')
gdp = np.array([])
pgdp1 = np.array([])
pgdp2 = np.array([])
pgdp3 = np.array([])
pGDP1 = np.array(Y.values[:, 1])
pGDP2 = np.array(Y.values[:, 2])
pGDP3 = np.array(Y.values[:, 3])
for i in range(3, 54, 3):
    pgdp2 = np.append(pgdp2, pGDP2[i])

for j in range(2, 174, 3):
    gdp = np.append(gdp, GDP2[j])

endog_Q = pd.DataFrame({"GDP3": gdp})
endog_Q.index = time_Q
endog = pd.DataFrame({"Elec": Elec})
endog.index = time_M
endog2 = pd.DataFrame({"PPI": PPI})
endog2.index = time_M
endog3 = pd.DataFrame({"pGDP3": pgdp2})
endog3.index = time_Q[41:58]
plt.plot(endog3, label='pgdp2')
plt.plot(endog_Q, label='GDP2')
# plt.plot(endog, label='Elec')
# plt.plot(endog2, label='PPI')

plt.legend()
plt.show()
