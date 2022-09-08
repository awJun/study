"""
최대값을 잡아서 0이하는 거는 0으로 잡고 x 이상인것은  x로 잡는다.
"""

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)  # 최대값을 잡아서 0이하는 거는 0으로 잡고 x 이상인것은  x로 잡는다.


relu2 = lambda x : np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu2(x)

plt.plot(x, y)
plt.grid()
plt.show()
