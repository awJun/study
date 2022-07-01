class Car:
    def __init__ (self, name, color):
        self.name = name
        self.color = color
    def show_info(self):
        print("이름 : ", self.name, "/ 색상", self.color)
        
car1 = Car("우유", "흰색")
car1.show_info()

car2 = Car("콩", "검정색")
car2.show_info()

class test(Car):
    def __init__(self, a, b):
        self.a = a
        self.b = b

Test = test("에이 ", "a", "/ 비 ", "b")








