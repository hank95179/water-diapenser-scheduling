import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from tbats import TBATS, BATS
from sklearn.metrics import mean_squared_error
import csv

total = []
levelorder = [10]
max = 0
min = 334300
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

for info2 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/02Fridge_continue'):        #讀資料近來
	domain2 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/02Fridge_continue')
	info2 = os.path.join(domain2,info2) 
	data2 = pd.read_csv(info2)
	data2= np.array(data2,dtype = np.float64) 
	df2 = pd.DataFrame(data2,index=index)
	df2 = df2.resample('1H').sum()
	# print(df[0])
	total.append(df2[0])

for info3 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue'):        #讀資料近來
	domain3 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue')
	info3 = os.path.join(domain3,info3) 
	data3 = pd.read_csv(info3)
	data3= np.array(data3,dtype = np.float64) 
	df3 = pd.DataFrame(data3,index=index)
	df3 = df3.resample('1H').sum()
	# print(df[0])
	total.append(df3[0])

for info4 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/05Fridge_continue'):        #讀資料近來
	domain4 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/05Fridge_continue')
	info4 = os.path.join(domain4,info4) 
	data4 = pd.read_csv(info4)
	data4= np.array(data4,dtype = np.float64) 
	df4 = pd.DataFrame(data4,index=index)
	df4 = df4.resample('1H').sum()
	# print(df[0])
	total.append(df4[0])







total = list(chain.from_iterable(total))
for i in range(len(total)):
	if total[i] > max and total[i] != 0: max = total[i]
	elif total[i] < min and total[i] > 0: min = total[i]
level = (max-min)/15
print(min,max,level)
for i in range(len(total)):
	total[i] //= level
	# total[i] += 1
	total[i] = int(total[i])
	# levelorder[int(total[i])] += 1
# print(levelorder)
max = 0
min = 10
for i in range(len(total)):
	if total[i] > max and total[i] > 0: max = total[i]
	elif total[i] < min and total[i] >= 0: min = total[i]
print(min,max,level)

