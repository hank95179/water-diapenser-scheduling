import csv
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain


origin = pd.read_csv('all.csv')
origin = np.array(origin)
water = 0
walk = np.zeros((109,504))
walknum = 0
for i in range(109):
	if (i + 1) % 5 == 2 or (i + 1) %5 ==4:
		for j in range(168,504):
			water += origin[i][j]
			if j % 24 < 15 and j%24 >= 8:
				if origin[i][j] == 1 :
					walk[i][j] += 1
					walknum += 1
	else:
		for j in range(168,504):
			water += origin[i][j]
			if j %24 >= 17 or j%24 < 5:
				if origin[i][j] == 1 :
					walk[i][j] += 1
					walknum += 1
print(water/109/14)
nwalk = water - walknum
print(walknum/water)
print(nwalk)
a = 216 // 24
print(a)
second = np.zeros((109,14))
secondwater = np.zeros((109,14))
for i in range(109):
	for j in range(14):
		for k in range(24):
			second[i][j] += walk[i][j*24+k+168]
			secondwater[i][j] += origin[i][j*24+k+168]
print(second.shape)
# for i in range(109):
	# for j in range(14):
	# 	secondwater[i][j] = second[i][j] / secondwater[i][j]
print(secondwater)
print(second)
# for i in range(14):
		# print(second[i])
# temp = 0
final = np.zeros((109,1))
for i in range(109):
	a=0
	b=0
	for j in range(14):
		a += secondwater[i][j]
		b += second[i][j]
	final[i] = b*10//a
	# final[i] /= 10
print(final)
third = [0,0,0,0,0,0,0,0,0,0]
# third =np.array(third)
for i in range(109):
		temp = int(final[i])
		# # print (temp)
		third[temp] += 1
print(third)
# for i in range(109):
# 	for j in range(14):
# 		temp = int(second[i][j])
# 		# # print (temp)
# 		third[temp] += 1

prob = [nwalk/water,walknum/water]
# walks = ['0F', '1F']
walks = ['[0,0.1)', '[0.1,0.2)', '[0.2,0.3)', '[0.3,0.4)','[0.4,0.5)','[0.5,0.6)','[0.6,0.7)','[0.7,0.8)','[0.8,0.9)','[0.9,1)']
x = np.arange(len(walks))
plt.bar(x, third)
plt.xticks(x, walks)
plt.xlabel('Walk Rate')
plt.ylabel('People')
# plt.title('Final Term')
plt.show()