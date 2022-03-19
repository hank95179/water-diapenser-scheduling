import sys
import csv
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from itertools import chain
real = pd.read_csv('house4_real_levelized.csv')
real = np.array(real,dtype = np.float64) 
# print(real[336:])
answer_MAE = real[336:]
predict = pd.read_csv('house4_predict_levelized(2,24).csv')



predict = np.array(predict,dtype = np.float64) 
# for i in range(len(predict)):
# 	predict[i] //= 1
predict_MAE = np.array(predict)
real_all = 0
predict_all = 0
print(len(answer_MAE))
for i in range(len(answer_MAE)):
	real_all += answer_MAE[i]
	predict_all += predict_MAE[i]
# back7day = list(chain.from_iterable(real[14]))
# print(real_all,predict_all)
# answer_MAE = np.array(back7day)
# print(answer_MAE.shape)
print("MEA With Seasonal Period(2,24):",mean_squared_error(predict_MAE,answer_MAE, squared=False ))
print("MEA With Seasonal Period(2,24) 7 days:",mean_squared_error(real_all,predict_all, squared=False ))
print('Average:',real_all/len(answer_MAE))
