# 난 정말 시그모이드 ~~~   랄라 랄라 ~~~
# 활성화함수는 레이어를 축소시키는 역할임

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 1 / (1 + np.exp(-x)  시그모이드 공식임   /  0 과 1사이에 수렴하는게 sigmoid이다 0과 1은 아니다 사이를 수렴하는 것이다 !

# 위 아래 똑같음

sigmoid2 = lambda x : 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)  # -5 ~ 5까지 0.1씩 증가
print(x)
print(len(x))

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()













