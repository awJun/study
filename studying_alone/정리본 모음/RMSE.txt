"""=[ RMSE 사용 ]==============================================================================================

from sklearn.metrics import mean_squared_error 

def RMSE(a, b):
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

===[ mean_squared_error 선언 이유 ]==============================================================================================

loss에서는 RMSE를 사용할 수 없으므로 mean_squared_error에서 MSE를 불러온 후 sqrt(루트)를 해줘서
RMSE를 직접 만들어서 사용한다.

"""