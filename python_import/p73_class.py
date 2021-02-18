class Person:
    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address

    def greeting(self): #행위=함수
        print('안녕. 난 {0}(이)다.'.format(self.name))

# 클래스 안에 있는 건 기능과 변수, 함수... 이것저것 다 가능 (사물, 행위 등)
# init는 받아들이는 부분 적는다.(입력값) 자기 자신은 명시 안한다. 자기 자신이기 때문
# 입력값과 self가 반드시 명시되어야한다. 
# greeting은 행위(함수) 표현. 나 자신 self라고 반드시 넣어준다. 