import sys
import csv
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import chain

origin = pd.read_csv('Sheng Li 6th  8F 0823.csv',parse_dates =["Time"], index_col ="Time")
origin = origin.resample("1H").sum()
origin.plot()
plt.title("Sheng Li 6th  8F 8/23")
plt.show()
total = origin.resample("1D").sum()
print("一天總用電量:",total)
print("平均每小時用電量:",total/24)