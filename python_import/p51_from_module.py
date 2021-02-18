from machine.car import drive
from machine.tv import watch

drive()
watch()

print("_____________________________")
# 운전하다2
# 시청하다2

# from machine import car
# from machine import tv
from machine import car, tv

car.drive()
tv.watch()
# 운전하다2
# 시청하다2

print("______________test_______________")
from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()

from machine.test import car
from machine.test import tv

car.drive()
tv.watch()

from machine import test

test.car.drive()
test.tv.watch()

# 문제점 ?  같은 폴더안에서만 가능하다. 
# sklearn은 어떻게 가능? 아나콘다에서 환경변수 설정해줬기 때문. 전체 어느지점에서도 설정 가능하게 해준 것.