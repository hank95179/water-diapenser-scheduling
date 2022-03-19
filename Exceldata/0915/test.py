import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open('01/01/2012-06-01.csv') as csvfile:
  rows = csv.reader(csvfile)
  df1 = list(rows)
df1 = np.array(df1,dtype = np.float)
with open('01/01/2012-06-02.csv') as csvfile:
  rows = csv.reader(csvfile)
  df2 = list(rows)
df2 = np.array(df2,dtype = np.float)
with open('01/01/2012-06-04.csv') as csvfile:
  rows = csv.reader(csvfile)
  df3 = list(rows)
df3 = np.array(df2,dtype = np.float)
with open('01/01/2012-06-05.csv') as csvfile:
  rows = csv.reader(csvfile)
  df4 = list(rows)
df4 = np.array(df2,dtype = np.float)
with open('01/01/2012-06-06.csv') as csvfile:
  rows = csv.reader(csvfile)
  df5 = list(rows)
df5 = np.array(df2,dtype = np.float)
with open('01/01/2012-06-07.csv') as csvfile:
  rows = csv.reader(csvfile)
  df7 = list(rows)
df7 = np.array(df2,dtype = np.float)

for x in range(86400):
	df1[x] = df2[x] + df1[x] + df3[x] + df4[x] + df5[x] + df7[x]
index = pd.date_range('20120601','20120601235959',freq='S')
data = pd.DataFrame(df4,index=index)
# data = map(eval, data)
# data.plot()
# plt.show()
data = data.resample('min').sum() 
data.plot()
plt.show()
print(data)