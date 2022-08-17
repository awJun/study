import pandas as pd
import warnings
warnings.filterwarnings("ignore")

path = "D:/study_data/_data/"

datasets = pd.read_csv(path+'winequality-white.csv',index_col=None, header=0, sep=';')

import numpy as np
datasets2 = datasets.values

x = datasets[:, :11]
y = datasets[:, 11]

x_new = x[:-25]
y_new = y[:-25]
















