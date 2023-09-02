year = 2000

if (year%4==0 and year % 100 != 0) or year%400==0:
    print(year, '는 윤년입니다.')
else:
    print(year , '는 윤년이 아닙니다.')
