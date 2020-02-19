# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF, arma_order_select_ic
import numpy as np
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.graphics.api import qqplot
import statsmodels.api as sm

data = pd.read_csv('./data/data_without_province_update_20200218.csv', header=0,
                   names=['date', 'total_confirmed', 'total_recoveries', 'total_deaths', 'new_confirmed',
                          'new_recoveries', 'new_deaths'])

new_confirmed_df = data[['date', 'new_confirmed']]
new_recoveries_df = data[['date', 'new_recoveries']]

new_confirmed_series = pd.Series(index=data['date'], data=data['new_confirmed'].values)
new_recoveries_series = pd.Series(index=data['date'], data=data['new_recoveries'].values)
new_confirmed_series.index = pd.to_datetime(new_confirmed_series.index)
new_recoveries_series.index = pd.to_datetime(new_recoveries_series.index)

# 时序图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
new_confirmed_series.plot()
# plt.show()

# 自相关图
# plot_acf(new_confirmed_series).show()

# 平稳性检测
print(u'原始序列的ADF检验结果为：', ADF(new_confirmed_series.values))
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

# 差分后的结果
D_data = new_confirmed_series.diff().dropna()
print(u'差分序列的ADF检验结果为：', ADF(D_data.values))  # 平稳性检测
D_data1 = D_data.diff().dropna()
print(u'差分序列的ADF检验结果为：', ADF(D_data1.values))  # 平稳性检测
D_data2 = D_data1.diff().dropna()
print(u'差分序列的ADF检验结果为：', ADF(D_data2.values))  # 平稳性检测

# pmax = int(len(D_data2) / 10)  # 一般阶数不超过length/10
# qmax = int(len(D_data2) / 10)  # 一般阶数不超过length/10
# order = arma_order_select_ic(D_data2, max_ar=pmax, max_ma=qmax, ic=['aic', 'bic', 'hqic'])
# print(order)

# (0, 3)
# MODEL20 = ARMA(D_data2, (0, 1)).fit()
#
# resid = MODEL20.resid
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111)
# fig = qqplot(resid, line='q', ax=ax, fit=True)
# plt.show()
#
# print(sm.stats.durbin_watson(resid.values))
#
# print(MODEL20.forecast(7))

# predict_arma = MODEL20.predict(start=0, end=len(D_data2)+7)
# predict_arma.index=

# 白噪声检验
# print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data2, lags=1))  # 返回统计量和p值

# # new_confirmed_series.values.astype(float)
# # 定阶
# pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
# qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
# bic_matrix = []  # bic矩阵
# for p in range(pmax + 1):
#     tmp = []
#     for q in range(qmax + 1):
#         try:  # 存在部分报错，所以用try来跳过报错。
#             tmp.append(ARIMA(D_data2, (p, 1, q)).fit().bic)
#         except:
#             tmp.append(None)
#     bic_matrix.append(tmp)
#
# # 从中可以找出最小值
# bic_matrix = pd.DataFrame(bic_matrix)
#
# # 先用stack展平，然后用idxmin找出最小值位置。
# p, q = bic_matrix.stack().idxmin()
# print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
# # 建立ARIMA(0, 1, 1)模型
model = ARIMA(D_data2, (0, 1, 1)).fit()
# 给出一份模型报告
print(model.summary2())
# 作为期7天的预测，返回预测结果、标准误差、置信区间。
print(model.forecast(10))






