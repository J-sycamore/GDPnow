import statsmodels.api as sm
import pandas as pd
import numpy as np

k_factors = 2
factor_order = 2
error_order = 2


def month_value(value, size):
    for i in range(size - 1, 1, -1):
        value[i] = (value[i] + value[i - 1] + value[i - 2]) / 3

    ratio = value[2:size]
    for i in range(size - 2, 11, -1):
        ratio[i] = ratio[i] / ratio[i - 12] - 1

    return ratio[12:]


def sum_value(value, size):
    for i in range(size - 1, 2, -1):
        value[i] = (value[i] - value[i - 3]) / 3

    ratio = value[3:size]
    for i in range(size - 3, 11, -1):
        ratio[i] = ratio[i] / ratio[i - 12] - 1

    return ratio[12:]


def month_ratio(value, size):
    for i in range(size - 1, 1, -1):
        value[i] = (value[i] + value[i - 1] + value[i - 2]) / 3

    return value[2:size]


def sum_ratio(value, size):
    for i in range(size - 1, 2, -1):
        value[i] = (value[i] - value[i - 3]) / 3

    return value[3:size]


def last_ratio(value, size):
    for i in range(size - 1, 1, -1):
        value[i] = (value[i] + value[i - 1] + value[i - 2]) / 3

    ratio = value[2:size]
    for i in range(1, size - 2):
        ratio[i] = ratio[i - 1] * (1 + ratio[i])

    for i in range(size - 3, 11, -1):
        ratio[i] = ratio[i] / ratio[i - 12] - 1

    return ratio[12:]


def pre_GDP1(GDP1, CPI, Meat, Farm):
    time_M = pd.date_range(start='2010-3', end='2023-7', freq='M')
    time_Q = pd.date_range(start='2010Q1', end='2023Q3', freq='Q')
    rcpi = last_ratio(CPI, CPI.size)
    rmeat = last_ratio(Meat, Meat.size)
    rfarm = last_ratio(Farm, Farm.size)
    pre1 = np.array([])
    r1 = np.array([])
    for i in range(120, 174):
        endog = pd.DataFrame({"CPI": rcpi[:i - 14], "Meat": rmeat[:i - 14], "Farm": rfarm[:i - 14]})
        endog.index = time_M[:i - 14]
        gdp = np.array([])
        for j in range(14, i, 3):
            gdp = np.append(gdp, GDP1[j])

        endog_Q = pd.DataFrame({"GDP1": gdp})
        endog_Q.index = time_Q[:(i - 15) // 3 + 1]
        model = sm.tsa.DynamicFactorMQ(endog, k_factors=k_factors, factor_order=factor_order,
                                       error_order=error_order, endog_quarterly=endog_Q)
        res = model.fit()
        pre = res.predict(start=time_M[((i - 15) // 3 + 1) * 3], end=time_M[((i - 15) // 3 + 1) * 3])
        # print(pre)
        pre1 = np.append(pre1, pre["GDP1"])
        r1 = np.append(r1, GDP1[(i // 3 + 1) * 3 - 1])

    return pre1, r1


def pre_GDP2(GDP2, PPI, Elec, PMI2, PMI2_new):
    time_M = pd.date_range(start='2009-7', end='2023-7', freq='M')
    time_Q = pd.date_range(start='2009Q3', end='2023Q3', freq='Q')
    pre2 = np.array([])
    r2 = np.array([])
    for i in range(120, 174):
        endog = pd.DataFrame({"PPI": PPI[6:i], "Elec": Elec[6:i], "PMI2": PMI2[6:i], "PMI2_new": PMI2_new[6:i]})
        endog.index = time_M[:i - 6]
        gdp = np.array([])
        for j in range(8, i, 3):
            gdp = np.append(gdp, GDP2[j])

        endog_Q = pd.DataFrame({"GDP2": gdp})
        endog_Q.index = time_Q[:(i - 9) // 3 + 1]
        model = sm.tsa.DynamicFactorMQ(endog, k_factors=2, factor_order=factor_order,
                                       error_order=error_order, endog_quarterly=endog_Q)
        res = model.fit()
        pre = res.predict(start=time_M[((i - 9) // 3 + 1) * 3 + 2], end=time_M[((i - 9) // 3 + 1) * 3 + 2])
        # print(pre, GDP2[(i // 3 + 1) * 3 - 1])
        pre2 = np.append(pre2, pre['GDP2'])
        r2 = np.append(r2, GDP2[(i // 3 + 1) * 3 - 1])

    return pre2, r2


def pre_GDP3(GDP3, ISP, PMI3, Travel):
    time_M = pd.date_range(start='2017-2', end='2023-7', freq='M')
    time_Q = pd.date_range(start='2017Q1', end='2023Q3', freq='Q')
    pre3 = np.array([])
    r3 = np.array([])
    ISP = month_ratio(ISP[95:], ISP.size - 95)
    PMI3 = month_ratio(PMI3[95:], PMI3.size - 95)
    Travel = month_ratio(Travel[95:], Travel.size - 95)
    for i in range(120, 174):
        endog = pd.DataFrame({"ISP": ISP[:i - 97], "PMI3": PMI3[:i - 97], "Travel": Travel[:i - 97]})
        endog.index = time_M[:i - 97]
        gdp = np.array([])
        for j in range(98, i, 3):
            gdp = np.append(gdp, GDP3[j])

        endog_Q = pd.DataFrame({"GDP3": gdp})
        endog_Q.index = time_Q[:(i - 99) // 3 + 1]
        model = sm.tsa.DynamicFactorMQ(endog, k_factors=k_factors, factor_order=factor_order,
                                       error_order=error_order, endog_quarterly=endog_Q)
        res = model.fit()
        pre = res.predict(start=time_M[((i - 99) // 3 + 1) * 3], end=time_M[((i - 99) // 3 + 1) * 3])
        # print(pre, GDP3[(i // 3 + 1) * 3 - 1])
        pre3 = np.append(pre3, pre["GDP3"])
        r3 = np.append(r3, GDP3[(i // 3 + 1) * 3 - 1])

    return pre3, r3


if __name__ == '__main__':
    X = pd.read_excel("./data.xlsx")
    GDP1 = np.array(X.values[:, 2])
    GDP2 = np.array(X.values[:, 3])
    GDP3 = np.array(X.values[:, 4])
    CPI = np.array(X.values[:, 5]).astype('float')
    Meat = np.array(X.values[:, 6]).astype('float')
    Farm = np.array(X.values[:, 7]).astype('float')
    Farm = Farm / 100 - 1
    PPI = np.array(X.values[:, 8]).astype('float')
    Elec = np.array(X.values[:, 9]).astype('float')
    PMI2 = np.array(X.values[:, 10]).astype('float')
    PMI2_new = np.array(X.values[:, 11]).astype('float')
    ISP = np.array(X.values[:, 12]).astype('float')
    PMI3 = np.array(X.values[:, 13]).astype('float')
    Travel = np.array(X.values[:, 14]).astype('float')
    p1, r1 = pre_GDP1(GDP1, CPI, Meat, Farm)
    p2, r2 = pre_GDP2(GDP2, PPI, Elec, PMI2, PMI2_new)
    p3, r3 = pre_GDP3(GDP3, ISP, PMI3, Travel)
    df = pd.DataFrame({"p1": p1, "p2": p2, "p3": p3, "r1": r1, "r2": r2, "r3": r3})
    df.to_excel("./month.xlsx")



