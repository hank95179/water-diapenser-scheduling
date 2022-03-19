import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
i = 0
j = 0
total = []
# total2 = []
# total3 = []
for info in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/Fridge02'): 
	domain = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/Fridge02') #獲取文件夾的路徑 
	info = os.path.join(domain,info) #將路徑與文件名結合起來就是每個文件的完整路徑 
	data = pd.read_csv(info)
	data= np.array(data,dtype = np.float) 
	index=pd.date_range('20120601','20120601235958',freq='S')
	df = pd.DataFrame(data,index=index)
	df = df.resample('min').sum()
	# print(df[0])
	total.append(df[0])
	i += 1

total = list(chain.from_iterable(total))
total = np.array(total)
print(total.shape)
indexday=pd.date_range('20120726','20120815235959',freq='min')
ans = pd.DataFrame(total,index=indexday)
print(ans)
ans.plot()
plt.title("House 2 Fridge")
plt.show()

for info2 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/Fridge04'): 
	domain2 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/Fridge04') #獲取文件夾的路徑 
	info2 = os.path.join(domain2,info2) #將路徑與文件名結合起來就是每個文件的完整路徑 
	data2 = pd.read_csv(info2)
	data2 = np.array(data2,dtype = np.float) 
	# index=pd.date_range('20120601','20120601235958',freq='S')
	df2 = pd.DataFrame(data2,index=index)
	df2 = df2.resample('min').sum()
	# print(df[0])
	total2.append(df2[0])
	j += 1
total2 = list(chain.from_iterable(total2))
total2 = np.array(total2)
print(total2.shape)
# indexday=pd.date_range('20120726','20120815235959',freq='min')
ans2 = pd.DataFrame(total2,index=indexday)
print(ans2)
# ans2.plot()
# plt.title("House 4 Fridge")
# plt.show()

for info3 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/Fridge05'): 
	domain3 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/Fridge05') #獲取文件夾的路徑 
	info3 = os.path.join(domain3,info3) #將路徑與文件名結合起來就是每個文件的完整路徑 
	data3 = pd.read_csv(info3)
	data3 = np.array(data3,dtype = np.float) 
	# index=pd.date_range('3010601','20120601235958',freq='S')
	df3 = pd.DataFrame(data3,index=index)
	df3 = df3.resample('min').sum()
	# print(df[0])
	total3.append(df3[0])
	j += 1
total3 = list(chain.from_iterable(total3))
total3 = np.array(total3)
print(total3.shape)
# indexday=pd.date_range('20120726','20120815235959',freq='min')
ans3 = pd.DataFrame(total3,index=indexday)
print(ans3)
# # ans3.plot()
# # plt.title("House 5 Fridge")
# plt.plot(ans)
# plt.plot(ans2)
# plt.plot(ans3)
# plt.title("House 1 4 5 Fridge")
# plt.legend(['House 1','House 4','House 5'])
# plt.show()