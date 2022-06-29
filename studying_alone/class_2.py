# 상속: 다른 클래스의 맴버 변수와 메소드를 물려 받아 사용하는 기법
# 부모와 자식 관계가 존재한다.
# 자식 클래스: 부모 클래스를 상속 받은 클래스

class Unit:   # 부모 클래스 생성
    def __init__(self, name, power):
        self.name = name
        self.power = power
    def attack(self):
        print(self.name, "이(가) 공격을 수행합니다. [전투력:", self.power, "]")
        
    






































