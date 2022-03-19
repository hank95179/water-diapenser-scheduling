import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df1 = pd.read_csv('01/01/2012-06-01.csv')
df2 = pd.read_csv('01/02/2012-06-01.csv')
df4 = pd.read_csv('01/04/2012-06-01.csv')
df5 = pd.read_csv('01/05/2012-06-01.csv')
df6 = pd.read_csv('01/06/2012-06-01.csv')
df7 = pd.read_csv('01/07/2012-06-01.csv')
df1 = np.array(df1,dtype = np.float)
df2 = np.array(df2,dtype = np.float)
df4 = np.array(df4,dtype = np.float)
df5 = np.array(df5,dtype = np.float)
df6 = np.array(df6,dtype = np.float)
df7 = np.array(df7,dtype = np.float)
for x in range(86399):
  df1[x] = df2[x] + df1[x] + df4[x] + df6[x] + df5[x] + df7[x]

print(df1)
index=pd.date_range('20120601','20120601235958',freq='S')
data = pd.DataFrame(df1,index=index)
data = data.resample('min').sum()  #用不了mean所以直接除
data.plot() #加了會跳TypeError: no numeric data to plot
# plt.show()
