
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf

#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target
# print(x.shape, y.shape)   # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y,     # 판다스 먹힌다. 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234,
                                                    )

# placeholder
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])   # None : 행 무시하겟다 라는 느낌 / 컬럼은 3이다!
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])   # None : 행 무시하겟다 라는 느낌 / 컬럼은 3이다!



















