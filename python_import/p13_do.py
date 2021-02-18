import p11_car
import p12_tv

# 운전하다
# p11_car.py의 module 이름은 :  p11_car
# 시청하다
# p12_tv.py의 module 이름은 :  p12_tv
# import 통해서 땡겨왔을 때 메인이 아니면 파일 이름이 나온다. 

print('________________________')
print('p13_do.py의 module 이름은 : ', __name__)
# p13_do.py의 module 이름은 :  __main__
print('________________________')

p11_car.drive() 
p12_tv.watch()

# 운전하다
# 시청하다