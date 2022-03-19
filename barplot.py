import sys
import csv
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import chain
from itertools import chain
indexans = pd.date_range('20120726','20120815235959',freq='1H')
# indexans = np.array(indexans)
# indexans = list(chain.from_iterable(indexans))
indexans = []
# print(indexans)
real = pd.read_csv('house5_real.csv')
real = np.array(real,dtype = np.float64) 
# real = pd.DataFrame(predict,index=indexans)
# print(real[336:])
volt = real
levelized = pd.read_csv('house5_real_15levelized.csv')
levelized = np.array(levelized,dtype = np.float64) 
# print("ARRAY:",predict)
levelized = list(levelized)
# print("LIST:",predict)
levelized = pd.Series(levelized)
# print("SERIES:",predict)
lv = pd.DataFrame()
start = 2012080900
time = []
for i in range(21):
	for j in range(24):
		time.append(start+100*i+j)
time = pd.Series(time)
lv["VOLT"] = levelized.str[0]
lv["TIME"] =  pd.to_datetime(time, format = "%Y%m%d%H").dt.hour

# fig,all = plt.subplots(3,7)
for i in range(21):
	sns.set(style="white", rc={"lines.linewidth":3})
	# plt.subplot(3,7,1+i)
	fig,ax1 = plt.subplots()
	if(i < 6):
		plt.title("7/%d"%(26+i))
	else:
		plt.title("8/%d"%(i-5))
	ax2 = ax1.twinx()
	sns.barplot(
				x='TIME',
				y='VOLT',
				data=lv[0+i*24:(1+i)*24],
				ci=0,
				color='#CC8800',
				ax = ax1
				)
	sns.lineplot(
				# x='time',
				# y='voltage',
				data=volt[0+i*24:(1+i)*24],
				color='#004488',
				marker="o",
				ax = ax2
	)
	# plt.subplot(3,7,i+1)
	sns.set()
	if(i < 6):
		plt.savefig("H5_7月%d日_15lv.png"%(i+26))
	elif(i < 15):
		plt.savefig("H5_8月0%d日_15lv.png"%(i-5))
	else:
		plt.savefig("H5_8月%d日_15lv.png"%(i-5))
plt.show()
# plt.title("House 4 levelized")
# sns.set()
