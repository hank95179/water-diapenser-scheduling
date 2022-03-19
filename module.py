import sys
import pickle
import csv
import pandas as pd
from sklearn.metrics import mean_squared_error
from itertools import chain
real = pd.read_csv('house4_real.csv')
predict = pd.read_csv('house4_predict.csv')

print(ac1)