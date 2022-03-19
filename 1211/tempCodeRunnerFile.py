
# testPredict = model.predict(testX)

# # 回復預測資料值為原始數據的規模
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])

# # calculate 均方根誤差(root mean squared error)
# # print(trainY[0])
# trainScore = (mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f MAE' % (trainScore))
# testScore = (mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f MAE' % (testScore))
# # print("MAE :",mean_squared_error(predict_MAE,answer_MAE, squared=False ))
# # 畫訓練資料趨勢圖
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# # 畫測試資料趨勢圖
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# # 畫原始資料趨勢圖
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
