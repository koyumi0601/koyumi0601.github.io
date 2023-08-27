import numpy as np, cv2

m = np.random.randint(0,100, 1000000).reshape(1000,1000)           # 임의 난수 생성
# 행렬 원소 정렬
sort1 = cv2.sort(m, cv2.SORT_EVERY_ROW)                       # 행단위 오름차순
sort2 = cv2.sort(m, cv2.SORT_EVERY_COLUMN)                    # 열단위(세로) 오름차순
sort3 = cv2.sort(m, cv2.SORT_EVERY_ROW + cv2.SORT_DESCENDING) # 행단위(가로) 내림차순
sort4 = np.sort(m, axis=1)                                      # 세로축 정렬
sort5 = np.sort(m, axis=0)                                      # 가로축 정렬
sort6 = np.sort(m, axis=1)[:, ::-1]                             # 가로축 내림차순 정렬

titles= ['m','sort1','sort2','sort3','sort4','sort5', 'sort6', 'sort7']
for title in titles:
        print("[%s] = \n%s\n" %(title, eval(title)))
