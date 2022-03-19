import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset(dataset, look_back=24):#從第24HR開始 試試24 48 72HR前
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

#讀資料近來分階
data = pd.read_csv('CSIE-6F-0823-0912.csv',parse_dates =["Time"], index_col ="Time")
data = data.resample("1H").sum()
# print(data.iloc[0])
data_array = np.array(data)
# print(data_array)
smallest = 1
biggest = 0
for i in range(len(data)):
	if data_array[i] > biggest:
		biggest = data_array[i]
	if data_array[i] < smallest:
		smallest = data_array[i]
level = biggest - smallest
level /= 15
for i in range(len(data)):
	data_array[i] //= level


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data_array)
# print(dataset)
# 2/3 資料為訓練資料， 1/3 資料為測試資料
look_back = 168
train_size = 336
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size-look_back:len(dataset),:]

# 產生 (X, Y) 資料集, Y 是下一期的乘客數(reshape into X=t and Y=t+1)
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# print(trainX)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1,trainX.shape[1]))
# trainX = trainX.astype('float64')
# print("======================",trainX)
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# 建立及訓練 LSTM 模型
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=16, verbose=2)

# 預測
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 回復預測資料值為原始數據的規模
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# print(testX)
# print(testY)
# calculate 均方根誤差(root mean squared error)
ty = 0
py = 0
for i in range(len(trainY[0])):
	ty += trainY[0][i]
	py += trainPredict[i]
tempy = (ty-py)
print('Train Total MAE: %.2f' %abs(tempy))
trainScore = (mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Average MAE: %.2f ' % (trainScore))
tx = 0
px = 0
for i in range(len(trainX[0])):
	tx += testY[0][i]
	px += testPredict[i]
tempx = (tx-px)
print('Test Total MAE: %.2f' %abs(tempx))
testScore = (mean_squared_error(testY[0], testPredict[:,0]))
print('Test Average MAE: %.2f ' % (testScore))
# print("MAE :",mean_squared_error(predict_MAE,answer_MAE, squared=False ))
# 畫訓練資料趨勢圖
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# 畫測試資料趨勢圖
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back)+1:len(dataset)-1, :] = testPredict

# 畫原始資料趨勢圖
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()