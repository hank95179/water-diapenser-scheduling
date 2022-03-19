import csv
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain

stop = 0.0156
standby = 0.298
returncost = 0.41559746
origin = pd.read_csv('all.csv')
origin = np.array(origin)
water = 0
walk1 = np.zeros((1,336))
walk2 = np.zeros((1,336))
walk3 = np.zeros((1,336))
walk4 = np.zeros((1,336))
walk5 = np.zeros((1,336))
walknum = 0
stop_hr1 = 0
stop_hr2 = 0
stop_hr3 = 0
stop_hr4 = 0
stop_hr5 = 0
return_time1 = 0;
return_time2 = 0;
return_time3 = 0;
return_time4 = 0;
return_time5 = 0;
for i in range(109):
	if (i + 1) % 5 == 1:
		for j in range(168,504):
			if origin[i][j] == 1:
				walk1[0][j-168] = 1
	elif (i + 1) % 5 == 2:
		for j in range(168,504):
			if origin[i][j] == 1:
				walk2[0][j-168] = 1
	elif (i + 1) % 5 == 3:
		for j in range(168,504):
			if origin[i][j] == 1:
				walk3[0][j-168] = 1
	elif (i + 1) % 5 == 4:
		for j in range(168,504):
			if origin[i][j] == 1:
				walk4[0][j-168] = 1
	else:
		for j in range(168,504):
			if origin[i][j] == 1:
				walk5[0][j-168] = 1
# print("F1",walk1)
for i in range(335):
	if walk1[0][i] == 0 and walk1[0][i+1] == 0:
		stop_hr1 += 1;
		if i < 334 and walk1[0][i+2] == 1:
			return_time1 += 1;

print("一樓停機時數:",stop_hr1)
print("一樓重啟時數:",return_time1)
# print("============================")
cost1 = ((336-stop_hr1-return_time1)*standby)+(stop_hr1*stop)+(return_time1*returncost)
print("Cost:",cost1)

# print("F2",walk2)
for i in range(335):
	if walk2[0][i] == 0 and walk2[0][i+1] == 0:
		stop_hr2 += 1;
		if i < 334 and walk2[0][i+2] == 1:
			return_time2 += 1;
print("二樓停機時數:",stop_hr2)
print("二樓重啟時數:",return_time2)
cost2 = ((336-stop_hr2-return_time2)*standby)+(stop_hr2*stop)+(return_time2*returncost)
print("Cost:",cost2)
# print("============================")
# print("F3",walk3)
for i in range(335):
	if walk3[0][i] == 0 and walk3[0][i+1] == 0:
		stop_hr3 += 1;
		if i < 334 and walk3[0][i+2] == 1:
			return_time3 += 1;
print("三樓停機時數:",stop_hr3)
print("三樓重啟時數:",return_time3)
cost3 = ((336-stop_hr3-return_time3)*standby)+(stop_hr3*stop)+(return_time3*returncost)

print("Cost:",cost3)
# print("============================")
# print("F4",walk4)
for i in range(335):
	if walk4[0][i] == 0 and walk4[0][i+1] == 0:
		stop_hr4 += 1;
		if i < 334 and walk4[0][i+2] == 1:
			return_time4 += 1;
print("四樓停機時數:",stop_hr4)
print("四樓重啟時數:",return_time4)
cost4 = ((336-stop_hr4-return_time4)*standby)+(stop_hr4*stop)+(return_time4*returncost)
print("Cost:",cost4)
# print("============================")
# print("F5",walk5)
for i in range(335):
	if walk5[0][i] == 0 and walk5[0][i+1] == 0:
		stop_hr5 += 1;
		if i < 334 and walk5[0][i+2] == 1:
			return_time5 += 1;
print("五樓停機時數:",stop_hr5)
print("五樓重啟時數:",return_time5)
cost5 = ((336-stop_hr5-return_time5)*standby)+(stop_hr5*stop)+(return_time5*returncost)
print("Cost:",cost5)
cost_total = cost1 +cost2 +cost4 +cost3 +cost5
print("cost_total",cost_total)