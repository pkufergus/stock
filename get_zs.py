# coding: utf-8
import pandas as pd
import tushare as ts

# df = ts.get_k_data('sh', start='2016-01-01')
# df.to_csv('shzs.csv')
# df = ts.get_k_data('sz', start='2016-01-01')
# df.to_csv('szzs.csv')
df = ts.get_hist_data('hs300', start='2018-01-01')
df.to_csv('hs300.csv')
# df = ts.get_k_data('sz50', start='2016-01-01')
# df.to_csv('sz50.csv')
# df = ts.get_k_data('zxb', start='2016-01-01')
# df.to_csv('zxb.csv')
# df = ts.get_k_data('cyb', start='2016-01-01')