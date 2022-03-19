import pandas as pd
import numpy as np
ac1 = pd.read_csv('AC01.csv')
ac2 = pd.read_csv('AC02.csv')
ac3 = pd.read_csv('AC03.csv')
a1 = 0
a2 = 0
a3 = 0
for i in range(0,len(ac1) - 1,2):
	a1 += ac1.iloc[i]
	a2 += ac2.iloc[i]
	a3 += ac3.iloc[i]
print(a1 * 2 / (len(ac1) - 1))
print(a2 * 2 / (len(ac1) - 1))
print(a3 * 2 / (len(ac1) - 1))