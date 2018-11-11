import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
csv = pd.read_csv('../fer2013.csv')

# csv=csv[:20000]

csv_data = [[np.uint8(e) for e in i.split(" ")] for i in csv["pixels"]]
csv_label = [int(i) for i in csv["emotion"]]

train_data, test_data, train_label, test_label = \
train_test_split(csv_data, csv_label)

xy = (train_data, test_data, train_label, test_label)
np.save("./svm.npy", xy)
print('npy Save Complete')