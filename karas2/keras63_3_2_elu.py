"""
최대값을 잡아서 0이하는 거는 0으로 잡고 x 이상인것은  x로 잡는다.
"""

import numpy as np
import matplotlib.pyplot as plt

def elu(x,alpha):
    return (x>0)*x + (x<=0)*(alpha*(np.exp(x) - 1))

# 람다는 다시 연구
# elu2 = lambda x : (x>0)*x + (x<=0)*(alpha*(np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)
alpha = 0.5
y = elu(x, alpha)

plt.plot(x, y)
plt.grid()
plt.show()
