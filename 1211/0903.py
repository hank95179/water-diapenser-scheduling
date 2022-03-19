import sys
import csv
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import chain

origin = pd.read_csv('CSIE-6F-0903-0905.csv',parse_dates =["Time"], index_col ="Time")
origin = origin.resample("1H").sum()
origin_list = np.array(origin)


biggest = -1
smallest = 2
for i  in range(len(origin)):
    if origin_list[i] > biggest:
        biggest = origin_list[i]
    if origin_list[i] < smallest:
        smallest = origin_list[i]
level = biggest - smallest
level /= 15

print("Big %f Small %f level %f"%(biggest,smallest,level))
for i in range(len(origin_list)):
	origin_list[i]//=level
origin_level = origin_list.tolist()
temp = np.array(origin_list).flatten()
temp = temp.tolist()
# print(temp.shape)
larger6_label = ['1d0h', '1d1h', '1d2h', '1d3h', '1d4h','1d5h', '1d6h', '1d7h','1d8h', '1d9h',
				 '1d10h', '1d11h', '1d12h', '1d13h', '1d14h','1d15h', '1d16h', '1d17h','1d18h', '1d19h',	
				 '1d20h', '1d21h', '1d22h', '1d23h',
				 '2d0h', '2d1h', '2d2h', '2d3h', '2d4h','2d5h', '2d6h', '2d7h','2d8h', '2d9h',
				 '2d10h', '2d11h', '2d12h', '2d13h', '2d14h','2d15h', '2d16h', '2d17h','2d18h', '2d19h',	
				 '2d20h', '2d21h', '2d22h', '2d23h',
				 '3d0h', '3d1h', '3d2h', '3d3h', '3d4h','3d5h', '3d6h', '3d7h','3d8h', '3d9h',
				 '3d10h', '3d11h', '3d12h', '3d13h', '3d14h','3d15h', '3d16h', '3d17h','3d18h', '3d19h',	
				 '3d20h', '3d21h', '3d22h', '3d23h']
# print(temp)
x = np.arange(len(larger6_label))
plt.bar(x, temp)
plt.xticks(x, larger6_label,rotation = 90)
plt.xlabel('11d:0903, 2d:0904, 3d:0905')
plt.ylabel('Time')
plt.title('0903~0905')
plt.show()

# total = origin.resample("1D").sum()
# print("一天總用電量:",total)
# print("平均每小時用電量:",total/24)