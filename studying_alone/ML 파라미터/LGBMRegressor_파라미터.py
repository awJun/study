# 추후 정리필요(확신x)

Bayesian_params = {
    "max_depth" : (6, 16),
    "num_leaves" : (24, 64),
    "min_child_samples" : (10, 200),
    "min_child_weight" : (1, 50),
    "subsample" : (0.5, 1),
    "colsample_bytree" : (0.5, 1),
    "max_bin" : (10, 500),
    "reg_lambda" : (0.001, 10),
    "reg_alpha" : (0.01, 50)
}

def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha
              ):
    params = {
        'n_estimators':500, "learning_rate":0.02,
        'max_depth' : int(round(max_depth)),                             # int형태로 받아야해서 flout를 반올림 후 int로 변환 (무조건 정수형)
        'num_leaves' : int(round(num_leaves)),                           # int형태로 받아야해서 flout를 반올림 후 int로 변환 (무조건 정수형)
        'min_child_samples' : int(round(min_child_samples)),             # int형태로 받아야해서 flout를 반올림 후 int로 변환 (무조건 정수형)
        'min_child_weight' : int(round(min_child_weight)),               # int형태로 받아야해서 flout를 반올림 후 int로 변환 (무조건 정수형)
        'subsample' : max(min(subsample, 1), 0),                         # 0~1 사이의 값이 들어와야한다.  1이상이면 1 
        'colsample_bytree' : max(min(colsample_bytree, 1), 0),           # 0~1 사이의 값이 들어와야한다.  1이상이면 1 
        'max_bin' : max(int(round(max_bin)), 10),                        # 무조건 10 이상의 정수를 받아야한다.
        'reg_lambda' : max(reg_lambda, 0),                               # 무조건 양수만 받아야한다.
        'reg_alpha' : max(reg_alpha, 0)                                  # 무조건 양수만 받아야한다.
    }