import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.cluster import KMeans
df1 = pd.read_csv('occupancy/01_summer.csv')
df2 = pd.read_csv('occupancy/02_summer.csv')
# df3 = pd.read_csv('occupancy/03_summer.csv')
df4 = pd.read_csv('occupancy/04_summer.csv')
df5 = pd.read_csv('occupancy/05_summer.csv')
summer1 = []
df1 = np.array(df1)
for i in range(10,36):
	if i == 13 or i == 16 or i == 19 or i == 25 or i == 28 :
		continue
	summer1.append(df1[i][1:])
summer1 = np.array(summer1,dtype=int)
index=pd.date_range('20120601','20120601235959',freq='S')
summer1 = pd.DataFrame(summer1,columns=index)
data1 = pd.DataFrame()
temp1 = pd.DataFrame()
for i in range(21):
	temp1 = summer1.iloc[i].resample('1H').sum()
	data1 = data1.append(temp1,ignore_index=True)
print("---------------------")
data1 = np.array(data1,dtype=int)
print("data1",data1.shape)
data1 = list(chain.from_iterable(data1))
data1 = np.array(data1,dtype=int)
# print(data1)

summer2 = []
df2 = np.array(df2)
for i in range(48,72):
	if  i == 55 or i == 61 or i == 64 :
		continue
	summer2.append(df2[i][1:])
summer2 = np.array(summer2,dtype=int)
summer2 = pd.DataFrame(summer2,columns=index)
data2 = pd.DataFrame()
temp2 = pd.DataFrame()
for i in range(21):
	temp2 = summer2.iloc[i].resample('1H').sum()
	data2 = data2.append(temp2,ignore_index=True)
print("---------------------")
data2 = np.array(data2,dtype=int)
# print("data2",data2.shape)
data2 = list(chain.from_iterable(data2))
data2 = np.array(data2,dtype=int)
# print(data2)

# summer3 = []
# df3 = np.array(df3)
# for i in range(21):
# 	summer3.append(df3[i][1:])
# summer3 = np.array(summer3,dtype=int)
# summer3 = pd.DataFrame(summer3,columns=index)
# data3 = pd.DataFrame()
# temp3 = pd.DataFrame()
# for i in range(21):
# 	temp3 = summer3.iloc[i].resample('min').sum()
# 	data3 = data3.append(temp3,ignore_index=True)
# print("---------------------")
# data3 = np.array(data3,dtype=int)
# # print("data3",data3.shape)
# data3 = list(chain.from_iterable(data3))
# data3 = np.array(data3,dtype=int)
# print(data3)

summer4 = []
df4 = np.array(df4)
for i in range(1,26):
	if i == 4 or i == 7 or i == 16 or i == 15 :
		continue
	summer4.append(df4[i][1:])
summer4 = np.array(summer4,dtype=int)
summer4 = pd.DataFrame(summer4,columns=index)
data4 = pd.DataFrame()
temp4 = pd.DataFrame()
for i in range(21):
	temp4 = summer4.iloc[i].resample('min').sum()
	data4 = data4.append(temp4,ignore_index=True)
print("---------------------")
data4 = np.array(data4,dtype=int)
# print("data4",data4.shape)
data4 = list(chain.from_iterable(data4))
data4 = np.array(data4,dtype=int)
# print(data4)

summer5 = []
df5 = np.array(df5)
for i in range(2,29):
	if i == 5 or i == 8 or i == 11 or i == 17 or i == 18 or i == 21:
		continue
	summer5.append(df5[i][1:])
summer5 = np.array(summer5,dtype=int)
summer5 = pd.DataFrame(summer5,columns=index)
data5 = pd.DataFrame()
temp5 = pd.DataFrame()
for i in range(21):
	temp5 = summer5.iloc[i].resample('min').sum()
	data5 = data5.append(temp5,ignore_index=True)
print("---------------------")
data5 = np.array(data5,dtype=int)
# print("data5",data5.shape)
data5 = list(chain.from_iterable(data5))
data5 = np.array(data5,dtype=int)
# print(data5)

X = np.array([data1,data2,data4,data5])
print(X,X.shape)
kmeans = KMeans(n_clusters=2,max_iter = 300, random_state=0).fit(X)
# kmeans.labels_
print(kmeans.predict([data1,data2,data4,data5]))