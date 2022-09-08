# 탄젠트 -1 ~ 1 사이에 수렴시킨다.  LSTM에서 많이 사용했다.  gate쪽에서 많이 사용했다.

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()












