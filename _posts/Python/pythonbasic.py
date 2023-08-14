# # 파일 생성하기
# f = open("D:/새파일.txt", 'w') # 파일 객체 생성
# for i in range(1, 6):
#     data = "%d번째 줄입니다. \n" % i
#     f.write(data) # 파일 쓰기
# f.close # 파일 닫기

# # 파일 추가하기
# f = open("D:/새파일.txt", 'a') # 파일 추가하기 모드로 열기
# for i in range(6, 10):
#     data = "%d번째 줄 추가입니다. \n" % i
#     f.write(data) # 파일 쓰기
# f.close # 파일 닫기

# # 파일 읽기 1
# f = open("D:/새파일.txt", 'r') # 파일 읽기 모드로 열기
# line = f.readline()
# print(line)
# while True:
#     line = f.readline()
#     if not line: break
#     print(line)
# f.close()

# # 파일 읽기 2
# f = open("D:/새파일.txt", 'r') # 파일 읽기 모드로 열기
# lines = f.readlines()
# print(lines)
# for line in lines:
#     print(line)
# f.close()

# # 파일 읽기 3
# f = open("D:/새파일.txt", 'r') # 파일 읽기 모드로 열기
# data = f.read()
# data
# f.close()

# # 파일 처리 후 파일 닫기(자동)
# with open("D:/새파일.txt", 'w') as f:
#     f.write("Now is better than never.")
import pandas as pd
# Series
data1 = [10, 20, 30, 40, 50]
data2 = ['1반', '2반', '3반', '4반', '5반']
sr1 = pd.Series(data1)
sr2 = pd.Series(data2)
sr3 = pd.Series([101, 102, 103, 104, 105])
sr4 = pd.Series(['월', '화', '수', '목', '금'])
sr5 = pd.Series(data1, index = [1000, 1001, 1002, 1003, 1004]) # index change
sr6 = pd.Series(data1, index = data2)
sr7 = pd.Series(data2, index = data1)
sr8 = pd.Series(data2, index = sr4)
sr8[0:4] # slicing
sr8.index
sr8.values
sr1 + sr3 # 둘 다 숫자이므로 덧셈 연산 수행
sr4 + sr2 # 둘 다 string이므로 문자연결 수행

# DataFrame
# Dictionary to dataframe
data_dic = {'year': [2018, 2019, 2020], 'sales': [350, 480, 1099]}
df1 = pd.DataFrame(data_dic)
# List to dataframe
df2 = pd.DataFrame([[89.2, 92.5, 90.8],[92.8, 89.9, 95.2]], index = ['중간고사', '기말고사'], columns = data2[0:3])
print(df2)

data_df = [['20201101', 'Hong', '90', '95'],['20201102', 'Kim', '93', '94'],['20201103', 'Lee', '87', '97']]
df3 = pd.DataFrame(data_df)
df3.columns = ['학번', '이름', '중간고사', '기말고사']
print(df3)
