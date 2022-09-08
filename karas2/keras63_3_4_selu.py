import numpy as np
import matplotlib.pyplot as plt

def selu(x, scale, alpha):
    if x > 0: return scale * x
    if x < 0: return scale * alpha * (np.exp(x) - 1)


# Leaky_ReLU2 = lambda x : np.maximum(0.01*x, x)


x = np.arange(-5, 5, 0.1)
scale = 1.05070098
alpha = 1.67326324
y = selu(x, scale, alpha)

plt.plot(x, y)
plt.grid()
plt.show()

미완성