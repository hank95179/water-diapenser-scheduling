import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
dataframe = read_csv('CSIE-6F-0903-0905.csv')
# dataset = dataframe.values
dataset = np.array(dataframe)
# dataset = dataset[0:100][0]
data0903 = []
data0904 = []
data0905 = []
day = len(dataset)/3
print(day)
for i in range(int(day)):
    data0903.append(dataset[i][1])
    data0904.append(dataset[i+(int(day))][1])
    data0905.append(dataset[i+(2*(int(day)))][1])

# dataset = dataset.astype('float32')
# 正規化(normalize) 資料，使資料值介於[0, 1]
# print("0903:",data0903)
# print("0904:",data0904)
# print("0905:",data0905)
edge0903 = 0
edge0904 = 0
edge0905 = 0
plot0903 = []
plot0904 = []
plot0905 = []
for i in range(int(day)):
    if data0903[i] > 0.01 :
        edge0903 = 1
    if data0903[i] < 0.005 and edge0903 == 1:
        print("0903:",i)
        plot0903.append(1)
        edge0903 = 0
    else:
        plot0903.append(0)
print("===========================")
for i in range(int(day)):
    if data0904[i] > 0.01 :
        edge0904 = 1
    if data0904[i] < 0.005 and edge0904 == 1:
        print("0904:",i)
        edge0904 = 0

print("===========================")
for i in range(int(day)):
    if data0905[i] > 0.01 :
        edge0905 = 1
    if data0905[i] < 0.005 and edge0905 == 1:
        print("0905:",i)
        edge0905 = 0
time0903 = pd.date_range('20210903','20210903235959',freq='min')
time0904 = pd.date_range('20210904','20210904235959',freq='min')
time0905 = pd.date_range('20210905','20210905235959',freq='min')
# print(time)   
plt.figure(figsize=(15, 5), dpi=100)
plt.scatter(time0903,plot0903,s=3)
plt.title("0903")
plt.savefig("0903.png")
plt.figure(figsize=(15, 5), dpi=100)
plt.scatter(time0904,plot0903,s=3)
plt.title("0904")
plt.savefig("0904.png")
plt.figure(figsize=(15, 5), dpi=100)
plt.scatter(time0905,plot0903,s=3)
plt.title("0905")
plt.savefig("0905.png")
plt.show()