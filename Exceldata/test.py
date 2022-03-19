import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from tbats import TBATS, BATS
from sklearn.metrics import mean_squared_error

total = []
total2 = []
total3 = []
train = []
test = []
y_forecasted = []
if __name__ == '__main__':
	index=pd.date_range('20120601','20120601235958',freq='S')
	for info in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/01Fridge_continue'): 
		domain = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/01Fridge_continue') #獲取文件夾的路徑 
		info = os.path.join(domain,info) #將路徑與文件名結合起來就是每個文件的完整路徑 
		data = pd.read_csv(info)
		data= np.array(data,dtype = np.float64) 
		df = pd.DataFrame(data,index=index)
		df = df.resample('1H').sum()
		# print(df[0])
		total.append(df[0])
	for info2 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue'): 
		domain2 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue') #獲取文件夾的路徑 
		info2 = os.path.join(domain2,info2) #將路徑與文件名結合起來就是每個文件的完整路徑 
		data2 = pd.read_csv(info2)
		data2 = np.array(data2,dtype = np.float64) 
		# index=pd.date_range('20120601','20120601235958',freq='S')
		df2 = pd.DataFrame(data2,index=index)
		df2 = df2.resample('1H').sum()
		# print(df[0])
		total2.append(df2[0])
	for info3 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/05Fridge_continue'): 
		domain3 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/05Fridge_continue') #獲取文件夾的路徑 
		info3 = os.path.join(domain3,info3) #將路徑與文件名結合起來就是每個文件的完整路徑 
		data3 = pd.read_csv(info3)
		data3 = np.array(data3,dtype = np.float64) 
		# index=pd.date_range('3010601','20120601235958',freq='S')
		df3 = pd.DataFrame(data3,index=index)
		df3 = df3.resample('1H').sum()
		# print(df[0])
		total3.append(df3[0])
	# total3 = np.array(total3)
	# print(total3.shape)
	for i in range(21):
		for j in range(24):
			total[i][j] = total[i][j] + total2[i][j] + total3[i][j]
	train = list(chain.from_iterable(total[:15]))
	train = np.array(train)
	test = list(total[15])
	test = np.array(test)
	indextrain = pd.date_range('20120726','20120809235959',freq='1H')
	indexday = pd.date_range('20120810','20120810235959',freq='1H')
	# indexday = pd.date_range('20120809','20120815235959',freq='1D')
	train = pd.DataFrame(train,index=indextrain)
	print(train.shape)
	# train.plot()
	# plt.title("Train")
	ans = pd.DataFrame(test,index=indexday)
	train = np.asarray(train)


	estimator = TBATS(
		seasonal_periods=(24,168)
	)

	fitted_model = estimator.fit(train)
	y_forecasted = fitted_model.forecast(steps=24)
	predict = pd.DataFrame(y_forecasted,index=indexday)
	print("MEA With Seasonal(24,168):",mean_squared_error(predict, ans, squared=False ))

	# estimator2 = TBATS(
	# 	# seasonal_periods=(24,168)
	# 	seasonal_periods=(2,24)
	# )

	# fitted_model2 = estimator2.fit(train)
	# y_forecasted2 = fitted_model2.forecast(steps=24)
	# predict_s24 = pd.DataFrame(y_forecasted2,index=indexday)
	# print("MEA With Seasonal(2,24):",mean_squared_error(predict_s24, ans, squared=False ))


	estimator_n = TBATS(
		# seasonal_periods=(24,168)
		# seasonal_periods=(1,24)
	)

	fitted_model_n = estimator_n.fit(train)
	y_forecasted_n = fitted_model_n.forecast(steps=24)
	predict_n = pd.DataFrame(y_forecasted_n,index=indexday)
	print("MEA With No Seasonal:",mean_squared_error(predict_n, ans, squared=False ))

	plt.plot(ans)
	plt.plot(predict)
	# plt.plot(predict_s24)
	plt.plot(predict_n)
	plt.title("08/10 Predict with Seasonal_periods House 1 4 5")
	# plt.legend(['ANSWER','seasonal(24,168)','seasonal(2,24)','No-Seasonal'])
	plt.legend(['ANSWER','seasonal(24,168)','No-Seasonal'])
	plt.show()