"""=[ train_test_split 사용 ]=======================================================

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    # train_size=0.7, 
                                                    # shuffle=True, 
                                                    random_state =66)

print(x_train)  # [2 7 6 3 4 8 5]
print(x_test)   # [ 1  9 10]
print(y_train)  # [2 7 6 3 4 8 5]
print(y_test)   # [ 1  9 10]

========================================================================================================================
"""