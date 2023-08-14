# 파일 생성하기
f = open("D:/새파일.txt", 'w') # 파일 객체 생성
for i in range(1, 6):
    data = "%d번째 줄입니다. \n" % i
    f.write(data) # 파일 쓰기
f.close # 파일 닫기

# 파일 추가하기
f = open("D:/새파일.txt", 'a') # 파일 추가하기 모드로 열기
for i in range(6, 10):
    data = "%d번째 줄 추가입니다. \n" % i
    f.write(data) # 파일 쓰기
f.close # 파일 닫기

# 파일 읽기 1
f = open("D:/새파일.txt", 'r') # 파일 읽기 모드로 열기
line = f.readline()
print(line)
while True:
    line = f.readline()
    if not line: break
    print(line)
f.close()

# 파일 읽기 2
f = open("D:/새파일.txt", 'r') # 파일 읽기 모드로 열기
lines = f.readlines()
print(lines)
for line in lines:
    print(line)
f.close()

# 파일 읽기 3
f = open("D:/새파일.txt", 'r') # 파일 읽기 모드로 열기
data = f.read()
data
f.close()

# 파일 처리 후 파일 닫기(자동)
with open("D:/새파일.txt", 'w') as f:
    f.write("Now is better than never.")