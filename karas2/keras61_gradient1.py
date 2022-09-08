import numpy as np
import matplotlib.pylab as plt

f = lambda x: x**2 - 4*x + 6

x = np.linspace(-1, 6, 100)   # -1 ~ 6까지 100개 순차적으로
print(x, len(x))

y = f(x)


##### 그려!!!
plt.plot(x, y)
plt.plot(2, 2, )  # 꼭지점 지점
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.show()












