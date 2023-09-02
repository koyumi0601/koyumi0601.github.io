import numpy as np, cv2

def preprocessing(coin_no):                                # 전처리 함수
    fname = "images/coin/{0:02d}.png".format(coin_no)
    image = cv2.imread(fname, cv2.IMREAD_COLOR)             # 영상읽기
    if image is None: return None, None                     # 예외처리는 메인에서

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          # 명암도 영상 변환
    gray = cv2.GaussianBlur(gray, (7, 7), 2, 2)             # 블러링 
    flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU              # 이진화 방법
    _, th_img = cv2.threshold(gray, 130, 255, flag)         # 이진화

    mask = np.ones((3, 3), np.uint8)
    th_img = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, mask) # 열림 연산
      
    return image, th_img

def find_coins(image):
    results = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    # 반복문 방식
    # circles = []
    # for contour in contours:
    #     center, radius = cv2.minEnclosingCircle(contour)        # 외각을 둘러싸는 원 검출
    #     circle = (tuple(map(int, center)), int(radius))
    #     if radius>25: circles.append(circle)

    # 리스트 생성 방식
    circles = [cv2.minEnclosingCircle(c) for c in contours] # 외각을 둘러싸는 원 검출
    circles = [(tuple(map(int, center)), int(radius))
               for center, radius in circles if radius>25]
    return circles
    



