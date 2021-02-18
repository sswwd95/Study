from test_import import p62_import

p62_import.sum2()
# 이 파일은 아나콘다 폴더에 들어있을 것이다.
# 작업그룹 import 썸탄다..!
print("+++++++++++++++++++")

from test_import.p62_import import sum2
sum2()

# 작업그룹 import 썸탄다..!
# path가 걸려있는 폴더에 있으면 실행가능하다.