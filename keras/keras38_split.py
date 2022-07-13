import numpy as np
from sklearn import datasets

a = np.array(range(1, 11))   # range(1, 11-1) 
# print(a)  # [ 1  2  3  4  5  6  7  8  9 10]

size = 5


def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):   # 10 - 5 + 1  =6 /  i가 반복하는 건 6번  /  0 1 2 3 4 5 <--  i가 들어가는 값의 순서
        subset = dataset[i : (i + size)]    #  subset = dataset[0 : 5]   1, 2, 3, 4, 5 
        aaa.append(subset)   
        
    return np.array(aaa)


bbb = split_x(a, size)
print(bbb)         # 
print(bbb.shape)   # 

x = bbb[:, :-1]      # 
y = bbb[:, -1]       
print(x)              # 
print(y)              # 
print(x.shape, y.shape)  # 
print(x.shape, y.shape)  # 


#- - - - - - - - - - - - - - - - -

# print(bbb)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

# print(x) 
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]

# print(y) 
# [ 5  6  7  8  9 10]




















































