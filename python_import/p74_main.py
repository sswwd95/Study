import p73_class as p73

# aaa = Modecheckpoint()
# class는 대문자로 쓴다. (통상적) 클래스로 했을 때 앞에 반환되는건 변수=인스턴스.인스턴스를 생성한다.

star = p73.Person('별이',20,'서울시 종로구')
# self 는 자기 자신이니 필요없고, 이름, 나이, 주소만 적어준다.

star.greeting()