import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from tbats import TBATS
from sklearn.metrics import mean_squared_error
import csv


if __name__ == '__main__':
	total = []
	total2 = []
	total3 = []
	train = []
	test = []
	predict_s1 = []

	y_forecasted = []
	index=pd.date_range('20120601','20120601235958',freq='S')
	for info in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue'):        #讀資料近來
		domain = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue')
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
	back7day = list(chain.from_iterable(total[14:21]))
	answer_MAE = np.array(back7day)
	Answer_MAE = 0
	for i in range(len(answer_MAE)):
		Answer_MAE += answer_MAE[i]
	print(Answer_MAE)
	Answer_MAE = np.array(Answer_MAE)
	answer = pd.DataFrame(answer,index=indexans)
	# answer.to_csv('house5_real.csv',index=0)
	for i in range(14,21):
		train = list(chain.from_iterable(total[:i])) #分train和test
		train = np.array(train)
		test = list(total[i])
		# total = list(chain.from_iterable(total)) 
		print('預估第%d天'%(i-13))
		train = np.asarray(train)
		estimator = TBATS(
			seasonal_periods=(24,168)
			# seasonal_periods=(2,24)
		)
		fitted_model = estimator.fit(train)
		y_forecasted = fitted_model.forecast(steps=24)

		predict_s1.append(y_forecasted) 
		# print(y_forecasted.shape)

	# print(predict_s1.shape)
	predict_s1 = list(chain.from_iterable(predict_s1))
	predict_MAE = np.array(predict_s1)
	Predict_MAE = 0
	for i in range(len(predict_MAE)):
			Predict_MAE += predict_MAE[i]
	predict_s1 = pd.DataFrame(predict_s1,index=indexpredict)
	# predict_s1.to_csv('house5_predict_No.csv',index=0)
	Predict_MAE = np.array(Predict_MAE)
	print("MEA With No Seasonal:",mean_squared_error(predict_MAE,answer_MAE, squared=False ))
	plt.plot(answer)
	plt.plot(predict_s1)
	# plt.plot(predict_s24)
	# plt.plot(predict_n)
	plt.title("HOUSE4 Fridge Predict")
	# plt.legend(['ANSWER','seasonal(24,168)','seasonal(2,24)','No-seasonal'])
	plt.legend(['Real','Seasonal(24,168)'])
	plt.show()
