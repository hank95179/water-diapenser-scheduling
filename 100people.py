import sys
import csv
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import chain

house1 = pd.read_csv("house1.csv")