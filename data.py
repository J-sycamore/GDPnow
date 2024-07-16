import numpy as np
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(r"D:\stata\utilities")


def bayes(N):
    from pystata import config
    config.init("mp")
    from pystata import stata
    stata.run('''
    clear
    import excel "E:/python/GDPNOW/test.xlsx", sheet("Sheet1") firstrow
    gen date = tq(2007q3) + _n - 1
    format %tq date
    tsset date
    bayes, rseed(20): var y1 y2 y3, lags(1/10)
    bayes, saving(bvarsim1)
    bayesfcast compute f_, step(1)
    ''')
    df = stata.pdataframe_from_data()
    os.remove(r"./bvarsim1.dta")
    return df.loc[N-2, 'f_y1'], df.loc[N-2, 'f_y2'], df.loc[N-2, 'f_y3']


def At(t: int, lam):
    e_t = np.zeros(t)
    e_t[-1] = 1
    I_t = np.identity(t - 2)
    Q_t = np.zeros((t - 2, t))
    for i in range(t - 2):
        Q_t[i, i], Q_t[i, i + 1], Q_t[i, i + 2] = 1, -2, 1
    A_t = np.matrix(e_t) @ Q_t.T @ (np.linalg.inv(Q_t @ Q_t.T + I_t / lam)) @ Q_t
    return A_t


def one_sided_HP_filter(df, lam):
    df_local = df.copy()
    data_series = np.array(df_local)
    length = len(df)
    list_cycle = [math.nan, math.nan]
    for i in range(2, length):
        sub_series = data_series[:i + 1]
        sub_A_t = At(i + 1, lam)
        cycle_t = (sub_A_t @ sub_series)[0, 0]
        list_cycle.append(cycle_t)
    return np.array(df_local) - np.array(list_cycle), np.array(list_cycle)


def predict(GDP):
    y1, y1_inv = one_sided_HP_filter(GDP.values[:, 1], 10000)
    y2, y2_inv = one_sided_HP_filter(GDP.values[:, 2], 10000)
    y3, y3_inv = one_sided_HP_filter(GDP.values[:, 3], 10000)
    # plt.plot(y1, label='HP1')
    # plt.plot(GDP.values[:, 1], label='origin1')
    # plt.plot(y1_inv, label='inv1')
    # plt.plot(y2, label='HP2')
    # plt.plot(GDP.values[:, 2], label='origin2')
    # plt.plot(y2_inv, label='inv')
    # plt.plot(y3, label='HP3')
    # plt.plot(GDP.values[:, 3], label='origin3')
    # plt.plot(y3_inv, label='inv3')
    # plt.legend()
    # plt.show()

    x11 = np.array([])
    x21 = np.array([])
    x31 = np.array([])
    yp11 = np.array([])
    yp21 = np.array([])
    yp31 = np.array([])
    x12 = np.array([])
    x22 = np.array([])
    x32 = np.array([])
    yp12 = np.array([])
    yp22 = np.array([])
    yp32 = np.array([])
    x13 = np.array([])
    x23 = np.array([])
    x33 = np.array([])
    yp13 = np.array([])
    yp23 = np.array([])
    yp33 = np.array([])
    x14 = np.array([])
    x24 = np.array([])
    x34 = np.array([])
    yp14 = np.array([])
    yp24 = np.array([])
    yp34 = np.array([])
    for i in range(2, 48):
        if i % 4 == 0:
            x11 = np.append(x11, i)
            x21 = np.append(x21, i)
            x31 = np.append(x31, i)
            yp11 = np.append(yp11, y1_inv[i])
            yp21 = np.append(yp21, y2_inv[i])
            yp31 = np.append(yp31, y3_inv[i])

        elif i % 4 == 1:
            x12 = np.append(x12, i)
            x22 = np.append(x22, i)
            x32 = np.append(x32, i)
            yp12 = np.append(yp12, y1_inv[i])
            yp22 = np.append(yp22, y2_inv[i])
            yp32 = np.append(yp32, y3_inv[i])

        elif i % 4 == 2:
            x13 = np.append(x13, i)
            x23 = np.append(x23, i)
            x33 = np.append(x33, i)
            yp13 = np.append(yp13, y1_inv[i])
            yp23 = np.append(yp23, y2_inv[i])
            yp33 = np.append(yp33, y3_inv[i])

        elif i % 4 == 3:
            x14 = np.append(x14, i)
            x24 = np.append(x24, i)
            x34 = np.append(x34, i)
            yp14 = np.append(yp14, y1_inv[i])
            yp24 = np.append(yp24, y2_inv[i])
            yp34 = np.append(yp34, y3_inv[i])

    print(yp34)
    model_y11 = LinearRegression()
    model_y12 = LinearRegression()
    model_y13 = LinearRegression()
    model_y14 = LinearRegression()
    model_y21 = LinearRegression()
    model_y22 = LinearRegression()
    model_y23 = LinearRegression()
    model_y24 = LinearRegression()
    model_y31 = LinearRegression()
    model_y32 = LinearRegression()
    model_y33 = LinearRegression()
    model_y34 = LinearRegression()
    p1 = np.array([])
    p2 = np.array([])
    p3 = np.array([])
    for i in range(48, 66):
        df = pd.DataFrame({'y1': y1[2:i], 'y2': y2[2:i], 'y3': y3[2:i]})
        df.to_excel("test.xlsx", index=False)
        y1p, y2p, y3p = bayes(i)
        if i % 4 == 0:
            model_y11.fit(x11.reshape((-1, 1)), yp11)
            model_y21.fit(x21.reshape((-1, 1)), yp21)
            model_y31.fit(x31.reshape((-1, 1)), yp31)
            pre1 = model_y11.predict(np.array([[i]]))
            pre2 = model_y21.predict(np.array([[i]]))
            pre3 = model_y31.predict(np.array([[i]]))
            x11 = np.append(x11, i)
            x21 = np.append(x21, i)
            x31 = np.append(x31, i)
            yp11 = np.append(yp11, y1_inv[i])
            yp21 = np.append(yp21, y2_inv[i])
            yp31 = np.append(yp31, y3_inv[i])
        elif i % 4 == 1:
            model_y12.fit(x12.reshape((-1, 1)), yp12)
            model_y22.fit(x22.reshape((-1, 1)), yp22)
            model_y32.fit(x32.reshape((-1, 1)), yp32)
            pre1 = model_y12.predict(np.array([[i]]))
            pre2 = model_y22.predict(np.array([[i]]))
            pre3 = model_y32.predict(np.array([[i]]))
            x12 = np.append(x12, i)
            x22 = np.append(x22, i)
            x32 = np.append(x32, i)
            yp12 = np.append(yp12, y1_inv[i])
            yp22 = np.append(yp22, y2_inv[i])
            yp32 = np.append(yp32, y3_inv[i])
        elif i % 4 == 2:
            model_y13.fit(x13.reshape((-1, 1)), yp13)
            model_y23.fit(x23.reshape((-1, 1)), yp23)
            model_y33.fit(x33.reshape((-1, 1)), yp33)
            pre1 = model_y13.predict(np.array([[i]]))
            pre2 = model_y23.predict(np.array([[i]]))
            pre3 = model_y33.predict(np.array([[i]]))
            x13 = np.append(x13, i)
            x23 = np.append(x23, i)
            x33 = np.append(x33, i)
            yp13 = np.append(yp13, y1_inv[i])
            yp23 = np.append(yp23, y2_inv[i])
            yp33 = np.append(yp33, y3_inv[i])
        elif i % 4 == 3:
            model_y14.fit(x14.reshape((-1, 1)), yp14)
            model_y24.fit(x24.reshape((-1, 1)), yp24)
            model_y34.fit(x34.reshape((-1, 1)), yp34)
            pre1 = model_y14.predict(np.array([[i]]))
            pre2 = model_y24.predict(np.array([[i]]))
            pre3 = model_y34.predict(np.array([[i]]))
            x14 = np.append(x14, i)
            x24 = np.append(x24, i)
            x34 = np.append(x34, i)
            yp14 = np.append(yp14, y1_inv[i])
            yp24 = np.append(yp24, y2_inv[i])
            yp34 = np.append(yp34, y3_inv[i])

        print(pre1, pre2, pre3)
        p1 = np.append(p1, y1p + pre1)
        p2 = np.append(p2, y2p + pre2)
        p3 = np.append(p3, y3p + pre3)

    return p1, p2, p3


if __name__ == '__main__':
    X = pd.read_excel('./realdata.xlsx')
    predict(X)
    p1, p2, p3 = predict(X)
    y1 = X.values[:, 1]
    y2 = X.values[:, 2]
    y3 = X.values[:, 3]
    rp1 = np.array([])
    rp2 = np.array([])
    rp3 = np.array([])
    r1 = np.array([])
    r2 = np.array([])
    r3 = np.array([])
    t = pd.date_range('2019Q1', '2023Q3', freq='Q')
    for i in range(48, 66):
        rp1 = np.append(rp1, (p1[i - 48] / y1[i - 4] - 1) * 100)
        rp2 = np.append(rp2, (p2[i - 48] / y2[i - 4] - 1) * 100)
        rp3 = np.append(rp3, (p3[i - 48] / y3[i - 4] - 1) * 100)
        r1 = np.append(r1, (y1[i] / y1[i - 4] - 1) * 100)
        r2 = np.append(r2, (y2[i] / y2[i - 4] - 1) * 100)
        r3 = np.append(r3, (y3[i] / y3[i - 4] - 1) * 100)

    pre = pd.DataFrame({'r1p': rp1, 'r2p': rp2, 'r3p': rp3, 'r1': r1, 'r2': r2, 'r3': r3})
    pre.to_excel("ans.xlsx")
