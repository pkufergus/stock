# encoding=utf-8

# 数据预处理以及绘制图形需要的模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 构建长短时神经网络需要的方法
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization

# 需要之前60次的数据来预测下一次的数据
need_num = 60
# 训练数据的大小
training_num = 500
# 迭代10次
epoch = 10
batch_size = 32

# 训练数据的处理，我们选取整个数据集的前500个数据作为训练数据，后面的数据为测试数据
# 从csv读取指数 数据
zs = 'hs300'
dataset = pd.read_csv(zs + '.csv')
print(zs, ':', dataset.shape)

dataset = dataset.sort_values(by='date', axis=0)
print(zs, ':', dataset.shape)
# 我们需要预测收盘价
data = dataset.iloc[:, 3:4].values
data_org = dataset.iloc[:, [0, 3]].values
print(zs, ':', data.shape, data[0])
print("data_org:", data_org.shape, data_org[0])
# 训练数据就是上面已经读取数据的前500行
training_data = data[:training_num]
# 因为数据跨度几年，随着时间增长，数字也随之增长，因此需要对数据进行归一化处理
# 将所有数据归一化为0-1的范围
sc = MinMaxScaler(feature_range=(0, 1))
'''
fit_transform()对部分数据先拟合fit，
找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），
然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
'''
training_data_scaled = sc.fit_transform(X=training_data)
print('training_data_scaled:', training_data_scaled.shape)

x_train = []
y_train = []
# 每60个数据为一组，作为测试数据，下一个数据为标签
for i in range(need_num, training_data_scaled.shape[0]):
    x_train.append(training_data_scaled[i - need_num: i])
    y_train.append(training_data_scaled[i, 0])
# 将数据转化为数组
x_train, y_train = np.array(x_train), np.array(y_train)
print("x_train:", x_train.shape)
# 因为LSTM要求输入的数据格式为三维的，[training_number, time_steps, 1]，因此对数据进行相应转化
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print("x_train3:", x_train.shape)

# 构建网络，使用的是序贯模型
model = Sequential()
# return_sequences=True返回的是全部输出，LSTM做第一层时，需要指定输入shape
model.add(LSTM(units=128, return_sequences=True, input_shape=[x_train.shape[1], 1]))
model.add(BatchNormalization())

model.add(LSTM(units=128))
model.add(BatchNormalization())

model.add(Dense(units=1))
# 进行配置
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size)

# 进行测试数据的处理
# 前500个为测试数据，但是将500-60个数据作为输入数据，因为这样可以获取
# 测试数据的潜在规律
# training_num+=60

inputs = data[training_num - need_num:]

inputs = inputs.reshape(-1, 1)
# 这里使用的是transform而不是fit_transform，因为我们已经在训练数据找到了
# 数据的内在规律，因此，仅使用transform来进行转化即可
inputs = sc.transform(X=inputs)
x_validation = []

pred_num = 1
for i in range(need_num, inputs.shape[0] - pred_num):
    x_validation.append(inputs[i - need_num:i, 0])

x_validation = np.array(x_validation)
print("x shape:{} {}".format(x_validation.shape, x_validation[0]))
x_validation = np.reshape(x_validation, (x_validation.shape[0], x_validation.shape[1], 1))

# 这是真实的股票价格，是源数据的[500:]即剩下的数据的价格
real_stock_price = data[training_num:]
real_x = []
for x in range(len(data_org[training_num:])):
    d = data_org[training_num:][x]
    if x % 3 == 0:
        real_x.append(d[0])
    else:
        real_x.append('')
# 进行预测
predictes_stock_price = model.predict(x=x_validation)
pred_y = predictes_stock_price[-1].tolist()[0]
# 使用 sc.inverse_transform()将归一化的数据转换回原始的数据，以便我们在图上进行查看
predictes_stock_price = sc.inverse_transform(X=predictes_stock_price)
prices = predictes_stock_price.tolist()
pred = predictes_stock_price
x_validation = []
x_validation.append(inputs[len(inputs) - need_num:, 0])
tmp_x = inputs[len(inputs) - need_num:, 0]
tmp_x = tmp_x.tolist()
for i in range(pred_num):
    tmp_x.pop(0)
    tmp_x.append(pred_y)
    print("i={} tmp_x:{}".format(i, tmp_x))
    x_validation = []
    x_validation.append(np.array(tmp_x))
    # print("i={} {} {}".format(i, len(x_validation), x_validation[0]))
    x_validation = np.array(x_validation)
    x_validation = np.reshape(x_validation, (x_validation.shape[0], x_validation.shape[1], 1))
    pred = model.predict(x=x_validation)
    pred_y = pred[-1].tolist()[0]
    pred = sc.inverse_transform(X=pred)
    print("i={} pred={} p={} pred_y={}".format(i, pred, pred[-1], pred_y))
    prices.append([pred[-1].tolist()[0]])

print("data_org len:{} {} {} {}".format(len(data_org[training_num:]), len(real_stock_price), len(prices), len(real_x)))
for i in range(len(data_org[training_num:])):
    d = data_org[training_num:][i]
    print("i={} {} {} {} {}".format(i, d[0],d[1], real_stock_price[i], prices[i]))

# 绘制数据图表，红色是真实数据，蓝色是预测数据
plt.plot(real_stock_price, color='red', label=zs + ' Stock Price')
plt.plot(prices, color='blue', label='Predicted Stock Price')
plt.title(zs + ' Stock Price Prediction')
plt.xlabel('date')
plt.xticks([x for x in range(len(real_stock_price))],real_x, rotation=300)
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.show()