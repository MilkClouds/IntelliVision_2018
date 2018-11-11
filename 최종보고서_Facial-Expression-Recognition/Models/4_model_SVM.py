import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

train_data, test_data, train_label, test_label = np.load("./svm.npy")

clf = svm.SVC()
clf.fit(train_data, train_label)

joblib.dump(clf,'svm.pkl')
print('clf dump Complete')

pre = clf.predict(test_data)
# 정답률 구하기
ac_score = metrics.accuracy_score(test_label, pre)
print("정답률 =", ac_score)