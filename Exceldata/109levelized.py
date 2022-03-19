import sys
import os
import csv
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import chain

def biggest(day):
	big = day[0]
	for i in range(24):
		if day[i] > big :
			big = day[i]
	return big

total = []
total2 = []
total3 = []
total4 = []
total5 = []
total6 = []
total7 = []
total8 = []
total9 = []
total10 = []
total11 = []
level = 22274.596610665474
index=pd.date_range('20120601','20120601235958',freq='S')
for info in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/01Fridge_continue'):        #讀資料近來
	domain = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/01Fridge_continue')
	info = os.path.join(domain,info) 
	data = pd.read_csv(info)
	data= np.array(data,dtype = np.float64) 
	df = pd.DataFrame(data,index=index)
	df = df.resample('1H').sum()
	df //= level
	# print(df[0])
	total.append(df[0])
total = np.array(total)
total = list(total)

for info2 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/02Fridge_continue'):        #讀資料近來
	domain2 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/02Fridge_continue')
	info2 = os.path.join(domain2,info2) 
	data2 = pd.read_csv(info2)
	data2= np.array(data2,dtype = np.float64) 
	df2 = pd.DataFrame(data2,index=index)
	df2 = df2.resample('1H').sum()
	df2 //= level
	# print(df[0])
	total2.append(df2[0])
total2 = np.array(total2)
total2 = list(total2)

for info3 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/05Fridge_continue'):        #讀資料近來
	domain3 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/05Fridge_continue')
	info3 = os.path.join(domain3,info3) 
	data3 = pd.read_csv(info3)
	data3= np.array(data3,dtype = np.float64) 
	df3 = pd.DataFrame(data3,index=index)
	df3 = df3.resample('1H').sum()
	df3 //= level
	# print(df[0])
	total3.append(df3[0])
total3 = np.array(total3)
total3 = list(total3)

for info4 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/01Fridge_continue2'):        #讀資料近來
	domain4 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/01Fridge_continue2')
	info4 = os.path.join(domain4,info4) 
	data4 = pd.read_csv(info4)
	data4= np.array(data4,dtype = np.float64) 
	df4 = pd.DataFrame(data4,index=index)
	df4 = df4.resample('1H').sum()
	df4 //= level
	# print(df[0])
	total4.append(df4[0])
total4 = np.array(total4)
total4 = list(total4)

for info5 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/02Fridge_continue2'):        #讀資料近來
	domain5 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/02Fridge_continue2')
	info5 = os.path.join(domain5,info5) 
	data5 = pd.read_csv(info5)
	data5= np.array(data5,dtype = np.float64) 
	df5 = pd.DataFrame(data5,index=index)
	df5 = df5.resample('1H').sum()
	df5 //= level
	# print(df[0])
	total5.append(df5[0])
total5 = np.array(total5)
total5 = list(total5)

for info6 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue'):        #讀資料近來
	domain6 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue')
	info6 = os.path.join(domain6,info6) 
	data6 = pd.read_csv(info6)
	data6= np.array(data6,dtype = np.float64) 
	df6 = pd.DataFrame(data6,index=index)
	df6 = df6.resample('1H').sum()
	df6 //= level
	# print(df[0])
	total6.append(df6[0])
total6 = np.array(total6)
total6 = list(total6)

for info7 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue2'):        #讀資料近來
	domain7 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue2')
	info7 = os.path.join(domain7,info7) 
	data7 = pd.read_csv(info7)
	data7= np.array(data7,dtype = np.float64) 
	df7 = pd.DataFrame(data7,index=index)
	df7 = df7.resample('1H').sum()
	df7 //= level
	# print(df[0])
	total7.append(df7[0])
total7 = np.array(total7)
total7 = list(total7)

for info8 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue3'):        #讀資料近來
	domain8 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/04Fridge_continue3')
	info8 = os.path.join(domain8,info8) 
	data8 = pd.read_csv(info8)
	data8= np.array(data8,dtype = np.float64) 
	df8 = pd.DataFrame(data8,index=index)
	df8 = df8.resample('1H').sum()
	df8 //= level
	# print(df[0])
	total8.append(df8[0])
total8 = np.array(total8)
total8 = list(total8)

for info9 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/05Fridge_continue2'):        #讀資料近來
	domain9 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/05Fridge_continue2')
	info9 = os.path.join(domain9,info9) 
	data9 = pd.read_csv(info9)
	data9= np.array(data9,dtype = np.float64) 
	df9 = pd.DataFrame(data9,index=index)
	df9 = df9.resample('1H').sum()
	df9 //= level
	# print(df[0])
	total9.append(df9[0])
total9 = np.array(total9)
total9 = list(total9)
for info10 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/06Fridge_continue'):        #讀資料近來
	domain10 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/06Fridge_continue')
	info10 = os.path.join(domain10,info10) 
	data10 = pd.read_csv(info10)
	data10= np.array(data10,dtype = np.float64) 
	df10 = pd.DataFrame(data10,index=index)
	df10 = df10.resample('1H').sum()
	df10 //= level
	# print(df[0])
	total10.append(df10[0])
total10 = np.array(total10)
total10 = list(total10)

for info11 in os.listdir('C:/Users/Hank/Desktop/python學習/Exceldata/06Fridge_continue2'):        #讀資料近來
	domain11 = os.path.abspath(r'C:/Users/Hank/Desktop/python學習/Exceldata/06Fridge_continue2')
	info11 = os.path.join(domain11,info11) 
	data11 = pd.read_csv(info11)
	data11= np.array(data11,dtype = np.float64) 
	df11 = pd.DataFrame(data11,index=index)
	df11 = df11.resample('1H').sum()
	df11 //= level
	print(df[0])
	total11.append(df11[0])
total11 = np.array(total11)
total11 = list(total11)
day = []
all = np.zeros((1,504))
for i in range(0,113,7):
	day.clear()
	# print("第%d人"%(i//7+1))
	for k in range(21):
		big = biggest(total[i+k])
		# print("第%d天"%(k+1))
		for j in range(24):
			day.append(total[i+k][j]) 
	# day.pop(0)
	all = np.insert(all,0,values=day,axis=0)
all = np.delete(all,-1,axis=0)
	# people = pd.DataFrame(day)
	# people.to_csv('people%d.csv'%(i//7+1),index=0)
	# print("__________________________")
for i in range(0,134,7):
	day.clear()
	# print("第%d人"%(i//7+18))
	for k in range(21):
		big = biggest(total2[i+k])
		# print("第%d天"%(k+1))
		for j in range(24):
			day.append(total2[i+k][j])
	all = np.insert(all,0,values=day,axis=0)
# 	day.pop(0) 
# 	# people = pd.DataFrame(day)
# 	# people.to_csv('people%d.csv'%(i//7+18),index=0)

# # 	print("__________________________")
for i in range(0,134,7):
	day.clear()
	# print("第%d人"%(i//7+38))
	for k in range(21):
		big = biggest(total3[i+k])
		# print("第%d天"%(k+1))/
		for j in range(24):
			day.append(total3[i+k][j]) 
	all = np.insert(all,0,values=day,axis=0)
	# people = pd.DataFrame(day)
	# people.to_csv('people%d.csv'%(i//7+38),index=0)

# 	print("__________________________")
for i in range(0,71,7):
	day.clear()
	# print("第%d人"%(i//7+58))
	for k in range(21):
		big = biggest(total4[i+k])
		# print("第%d天"%(k+1))
		for j in range(24):
			day.append(total4[i+k][j])
	all = np.insert(all,0,values=day,axis=0)
	# people = pd.DataFrame(day)
	# people.to_csv('people%d.csv'%(i//7+58),index=0)

# 	print("__________________________")
for i in range(0,43,7):
	day.clear()
	# print("第%d人"%(i//7+69))
	for k in range(21):
		big = biggest(total5[i+k])
		# print("第%d天"%(k+1))
		for j in range(24):
			day.append(total5[i+k][j]) 
	all = np.insert(all,0,values=day,axis=0)
	# people = pd.DataFrame(day)
	# people.to_csv('people%d.csv'%(i//7+69),index=0)

# 	print("__________________________")
for i in range(0,50,7):
	day.clear()
	# print("第%d人"%(i//7+76))
	for k in range(21):
		big = biggest(total6[i+k])
		# print("第%d天"%(k+1))
		for j in range(24):
			day.append(total6[i+k][j]) 
	all = np.insert(all,0,values=day,axis=0)
	# people = pd.DataFrame(day)
	# people.to_csv('people%d.csv'%(i//7+76),index=0)

# 	print("__________________________")

for i in range(0,22,7):
	day.clear()
# 	print("第%d人"%(i//7+84))
	for k in range(21):
		big = biggest(total7[i+k])
		# print("第%d天"%(k+1))
		for j in range(24):
			day.append(total7[i+k][j])
	all = np.insert(all,0,values=day,axis=0)
	# people = pd.DataFrame(day)
	# people.to_csv('people%d.csv'%(i//7+84),index=0)

# 	print("__________________________")
for i in range(0,57,7):
	day.clear()
# 	print("第%d人"%(i//7+88))
	for k in range(21):
		big = biggest(total8[i+k])
		# print("第%d天"%(k+1))
		for j in range(24):
			day.append(total8[i+k][j]) 
	all = np.insert(all,0,values=day,axis=0)
	# people = pd.DataFrame(day)
	# people.to_csv('people%d.csv'%(i//7+88),index=0)

# 	print("__________________________")
for i in range(0,43,7):
	day.clear()
# 	print("第%d人"%(i//7+97))
	for k in range(21):
		big = biggest(total9[i+k])
		# print("第%d天"%(k+1))
		for j in range(24):
			day.append(total9[i+k][j]) 
	all = np.insert(all,0,values=day,axis=0)
	# people = pd.DataFrame(day)
	# people.to_csv('people%d.csv'%(i//7+97),index=0)

# 	print("__________________________")
for i in range(0,36,7):
	day.clear()
	# print("第%d人"%(i//7+104))
	for k in range(21):
		big = biggest(total10[i+k])
		# print("第%d天"%(k+1))
		for j in range(24):
			day.append(total10[i+k][j]) 
	all = np.insert(all,0,values=day,axis=0)
for i in range(109):
	for j in range(504):
		if all[i][j] < 0:
			all[i][j] = 0
all2 = np.transpose(all)
print(all2)
all2 = pd.DataFrame(all2)
all2.to_csv("109people_level2.csv",index = 0)
# 	day.pop(0)
	# people = pd.DataFrame(day)
	# people.to_csv('people%d.csv'%(i//7+104),index=0)

# 	print("__________________________")
# for i in range(0,36,7):
# 	print("第%d人"%(i//7+110))
# 	for k in range(21):
# 		big = biggest(total11[i+k])
# 		print("第%d天"%(k+1))
# 		for j in range(24):
# 			if total11[i+k][j] == big:
# 				print("%d點"%j)

# 	print("__________________________")