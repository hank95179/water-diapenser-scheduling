import csv
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain

origin = pd.read_csv('CSIE-6F-0821-0822.csv',parse_dates =["Time"], index_col ="Time")
origin = origin.resample("1H").sum()
# all = np.zeros((504,1))
print(origin)
# walk = 0
# volt = 0
# for i in range(336):
# 	if i <= 45 and i >= 16:		
# 		volt += origin.iloc[i] * 3.42
# 		volt += origin.iloc[i+336] * 3.42
# 	elif i <= 93 and i >= 64:		
# 		volt += origin.iloc[i] * 3.42
# 		volt += origin.iloc[i+336] * 3.42
# 	elif i <= 141 and i >= 112:		
# 		volt += origin.iloc[i] * 3.42
# 		volt += origin.iloc[i+336] * 3.42
# 	elif i <= 189 and i >= 160:		
# 		volt += origin.iloc[i] * 3.42
# 		volt += origin.iloc[i+336] * 3.42
# 	elif i <= 237 and i >= 208:		
# 		volt += origin.iloc[i] * 3.42
# 		volt += origin.iloc[i+336] * 3.42
# 	elif i <= 285 and i >= 256:
# 		volt += origin.iloc[i] * 2.14
# 		volt += origin.iloc[i+336] * 2.14
# 	else:
# 		volt += origin.iloc[i] * 1.46
# 		volt += origin.iloc[i+336] * 1.46
# print (volt*5)
# print(volt*5/14)
# print(origin)
# for i in range(len(origin)):
# 	if float(origin.iloc[i]) >= 0.48:
# 		print(origin.iloc[i]) 
	# elif float(origin.iloc[i]) <= 0.02:
	# 	all[i] = 1
# all = all.flatten()
# people = pd.read_csv("all.csv")
# people = np.array(people)
# print(people[0])
# for j in range(109):
# 	for i in range(len(all)):
# 		if all[i] == 1 and people[j][i] == 1:
# 			walk += 1
# print(walk/109)
# total = origin.resample("1D").sum()
# print(total)
# for i in range(len(total)):
# 	volt += total.iloc[i]
# print(volt*5)
# print(volt*5/21)
# origin.plot()
# plt.title("Sheng Li 6th  8F 8/23")
# plt.show()
# total = origin.resample("1H").sum()
# print("一天總用電量:",total)
# print("平均每小時用電量:",total/24)