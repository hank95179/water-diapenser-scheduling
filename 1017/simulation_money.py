import csv
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain

heat = pd.read_csv("CSIE-6F-0821-0822.csv",parse_dates =["Time"], index_col ="Time")
heat = heat.resample("30min").sum()
heat = np.array(heat)
volt0730 = 0
volt1700 = 0
reheat = 0
for i in range(len(heat)):
	reheat += heat[i]
date0820 = pd.read_csv("CSIE-6F-0820.csv",parse_dates =["Time"], index_col ="Time")
index = pd.date_range('20210830','20210912235959',freq='30min')
date0820 = date0820.resample("30min").sum()
heat1700 = np.array(date0820)
heat0730 = np.array(date0820)
count = 0
# index = pd.date_range('2021','20120601235958',freq='S')
# print("1700:",heat1700)
# print("0730:",heat0730)
for i in range(len(heat1700)):
	if i > 14 and i <= 33:
		# volt0730 += heat1700[i]
		heat1700[i] = 0.0078
	else:
		# volt1700 += heat1700[i]
		heat0730[i] = 0.0078
for i in range(len(heat1700)):
	if i <= 14 or i > 33:
		# volt0730 += heat1700[i]
		count += heat1700[i] 
print("count:",heat0730,count/29*2)
for i in range(len(heat)):
	heat1700[i+27] = heat[i]
	heat0730[i+8] = heat[i]
final1700 = np.zeros((1,1))
final0730 = np.zeros((1,1))
for i in range(4):
	heat1700 = np.append(heat1700,heat1700)
	heat0730 = np.append(heat0730,heat0730)
# print("1700:",heat1700.shape)
# print("0730:",heat0730)
F2 = pd.DataFrame(heat1700[:672],index=index)
F1 = pd.DataFrame(heat0730[:672],index=index)
# F1 = F1.resample("1H").sum()
# F2 = F2.resample("1H").sum()
F1volt = 0
F2volt = 0
for i in range(336):
	if i <= 45 and i >= 16:		
		F1volt += F1.iloc[i] * 3.42
		F1volt += F1.iloc[i+336] * 3.42
		F2volt += F2.iloc[i] * 3.42
		F2volt += F2.iloc[i+336] * 3.42
	elif i <= 93 and i >= 64:		
		F1volt += F1.iloc[i] * 3.42
		F1volt += F1.iloc[i+336] * 3.42
		F2volt += F2.iloc[i] * 3.42
		F2volt += F2.iloc[i+336] * 3.42
	elif i <= 141 and i >= 112:		
		F1volt += F1.iloc[i] * 3.42
		F1volt += F1.iloc[i+336] * 3.42
		F2volt += F2.iloc[i] * 3.42
		F2volt += F2.iloc[i+336] * 3.42
	elif i <= 189 and i>= 160:		
		F1volt += F1.iloc[i] * 3.42
		F1volt += F1.iloc[i+336] * 3.42
		F2volt += F2.iloc[i] * 3.42
		F2volt += F2.iloc[i+336] * 3.42
	elif i <= 237 and i >= 208:		
		F1volt += F1.iloc[i] * 3.42
		F1volt += F1.iloc[i+336] * 3.42
		F2volt += F2.iloc[i] * 3.42
		F2volt += F2.iloc[i+336] * 3.42
	elif i <= 285 and i >= 256:
		F1volt += F1.iloc[i] * 2.14
		F1volt += F1.iloc[i+336] * 2.14
		F2volt += F2.iloc[i] * 2.14
		F2volt += F2.iloc[i+336] * 2.14
	else:
		F1volt += F1.iloc[i] * 1.46
		F1volt += F1.iloc[i+336] * 1.46
		F2volt += F2.iloc[i] * 1.46
		F2volt += F2.iloc[i+336] * 1.46
print("F1 volt",F1volt)
print("F2 volt",F2volt)
totalvolt = F1volt*3 + F2volt*2
print("Total",totalvolt)
print("Average",totalvolt/14)
print("0730:",F1)
F1.plot()
plt.title("Floor 1 3 5")
F2.plot()
plt.title("Floor 2 4")
plt.show()
print("0730~1700",volt0730)
daycos135 = reheat + volt0730
daycos24 = reheat + volt1700
daycost = 3*daycos135 + daycos24*2
print(daycost)
total = daycost*14
print(total)
