import csv
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain

heat = pd.read_csv("CSIE-6F-0830-0912.csv",parse_dates =["Time"], index_col ="Time")

heat = heat.resample("1H").sum()
heat.plot()
plt.title("8/30~9/12")
plt.show()
