"""
[해당 프로젝트 설명]
keras68_preprocess_input.py


"""


from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions   # 케라스에서 지원하는 것은 이것으로 스케일링 할 수 있게 지원해줌 근데 이거말고 정네처럼 스케일링 해도 괜찮음
import numpy as np

model = ResNet50(weights="imagenet")   # 1000개의 이미지 데이터를 찾는 대회에서 만든 모델임 여기에 우리가 사용한 이미지가 있어서 결과가 도출됨!
                                       # 이건 이미지넷에서 제공한 것이여서 위가 사용할 땐 직접 만들어야함 
                                       # 해당 모델에 있는 사진을 활용할 때만 이방법으로 사용가능하다.
img_path = "D:/study_data/_data/dog/sheep_dog.PNG"
img = image.load_img(img_path, target_size=(224, 224))  # 이미지 사이즈를 224, 224로 잡아준 것임
print(img)  # <PIL.Image.Image image mode=RGB size=224x224 at 0x23693194850>

x = image.img_to_array(img)  #  image.img_to_array : 위에서 불러온 이미지를 수치화 시켜준 것임
print("======================== image.img_to_array(img) ============================")
print(x, "\n", x.shape)  #  (224, 224, 3)


x = preprocess_input(x)      # 데이터를 전처리 해준다. 즉, 스케일링! 
x = np.expand_dims(x, axis=0)   # 차원을 늘려준다 / reshape해도 괜찮음  맨 앞에 행부분을 처리하려고 이걸함  (0, 1, 2 ,3)으로 위치조절 가능
print("======================== image.img_to_array(img) ============================")
print(x, "\n", x.shape)   #  (1, 224, 224, 3)
print(np.min(x), np.max(x))   # 최소값과 최대값 출력
# -98.779 75.061


# ResNet50 애는 이미 훈련이 다 된 모델이미르 그냥 predict을 하면 돼~ 
print("======================== image.img_to_array(predict) ============================")
preds = model.predict(x)     # 모델이 1000개이므로 결과도 1000개 나옴 그래서 우리는 argmax할거임
print(preds, "\n", preds.shape)


print("결과는 : ", decode_predictions(preds, top=5)[0]) # 해당 데이터에서 어떤 클래스를 사용 할 거인지 찾아준다

# 결과는 :  [('n04548280', 'wall_clock', 0.36907104), ('n04118776', 'rule', 0.20115228),
#         ('n02708093', 'analog_clock', 0.1780444), ('n03804744', 'nail', 0.08412291),
#         ('n03729826', 'matchstick', 0.06460645)]


# ('n03729826', 'matchstick', 0.06460645)
#   고유번호        종           acc







