import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from tbats import TBATS, BATS
from sklearn.metrics import mean_squared_error
import csv
if __name__ == '__main__':
	levelsecond = 130.531653
	levelminute = 2191.454
	levelhour = 22207.203774665475
	total = []
	predict_s1 = []
	index=pd.date_range('20120601','20120601235958',freq='S')
	for info in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/01Fridge_continue'):        #讀資料近來
		domain = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/01Fridge_continue')
		info = os.path.join(domain,info) 
		data = pd.read_csv(info)
		data= np.array(data,dtype = np.float64) 
		df = pd.DataFrame(data,index=index)
		df = df.resample('1H').sum()
		# print(df[0])
		total.append(df[0])
	indexpredict = pd.date_range('20120809','20120815235959',freq='1H')
	indexans = pd.date_range('20120726','20120815235959',freq='1H')
	answer = list(chain.from_iterable(total))
	print(len(answer))
	for i in range(len(answer)):
		# print(answer[i])
		answer[i] //= levelhour
		if answer[i] < 0 : answer[i] = 0
	# print(answer)
	back7day = list(chain.from_iterable(total[14:21]))
	
	for i in range(len(back7day)):
		back7day[i] //= levelhour
		if back7day[i] < 0 : back7day[i] = 0
	back7day = pd.DataFrame(back7day,index=indexpredict)
	# print(back7day)
	# total = list(chain.from_iterable(total))
	print(len(total))
	answer = pd.DataFrame(answer,index=indexans)
	# answer.to_csv('house5_real_15levelized.csv',index=0)
	# total = pd.DataFrame(total,index=indexans)
	for i in range(14,21):
		train = list(chain.from_iterable(total[:i])) #分train和test
		for k in range(len(train)):
			# print(answer[i])
			train[k] //= levelhour
			if train[k] < 0 : train[k] = 0
		train = np.array(train)
		# test = list(total[i])
		# total = list(chain.from_iterable(total)) 
		print('預估第%d天'%(i-13))
		train = np.asarray(train)
		estimator = TBATS(
			# seasonal_periods=(24,168)
			seasonal_periods=(2,24)
		)
		fitted_model = estimator.fit(train)
		y_forecasted = fitted_model.forecast(steps=24)
		predict_s1.append(y_forecasted) 
		# print(y_forecasted.shape)
	predict_s1 = list(chain.from_iterable(predict_s1))
	predict_s1 = pd.DataFrame(predict_s1,index=indexpredict)
	predict_s1.to_csv('house1_predict_15levelized(2,24)_index.csv')
	# predict_s1.plot(0)
	# plt.title("Predict")
	# back7day.plot()
	# plt.title("Truth")
	predict_MAE = np.array(predict_s1)
	answer_MAE = np.array(back7day)
	print("MEA With Seasonal:",mean_squared_error(predict_MAE,answer_MAE, squared=False ))
	# answer.plot()
	# total.plot()ㄅ
	# plt.show()
	plt.plot(answer)
	plt.plot(predict_s1)
	# plt.plot(predict_s24)
	# plt.plot(predict_n)
	plt.title("HOUSE1 Fridge Predict")
	# plt.legend(['ANSWER','seasonal(24,168)','seasonal(2,24)','No-seasonal'])
	plt.legend(['Real','Seasonal(24,168)'])
	plt.show()