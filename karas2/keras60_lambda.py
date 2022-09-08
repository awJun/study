gradient1 = lambda x : 2*x - 4
# 변수 = lambda x : 로 변수를 받겠더      2*x - 4   이건 연산되는 식임  


def gradient2(x):
    temp = 2*x - 4
    return temp
    
    
x = 3

print(gradient1(x))  # 2
print(gradient2(x))  # 2