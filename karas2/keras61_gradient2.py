import numpy as np

f = lambda x : x**2 - 4*x + 6   # 최저경사 연산 원래씩

# 위랑 아래 같은 거임

# def f(x):
#     temp = x**2 - 4*x + 6
#     return temp

gradient = lambda x : 2*x - 4   # 위에껄 편의상 미분까지 구해지는 경사의 미분
# gradient 디셉트 방식이다  / gradient 디셉트  얘는 미분값임
# 역전파할때 미분한 값이 들어간다


x = 10.0   # 초기값
epochs = 20
learning_rate = 0.25

print("step\t x\t f(x")
print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(0, x, f(x)))

for i in range(epochs):
    x = x - learning_rate * gradient(x)  # 미분 공식
    
    print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(i+1, x, f(x)))






