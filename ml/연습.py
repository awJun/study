import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train,x_test, axis=0)
# print(x.shape)

x = x.reshape(70000, 28*28)

from sklearn.decomposition import PCA
a=[]
pca = PCA(n_components=x.shape[1])
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR)
a.append(np.argmax(cumsum >= 0.95)+1)






















