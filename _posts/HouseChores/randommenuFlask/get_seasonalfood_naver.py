import time
from selenium import webdriver
import chromedriver_autoinstaller
from bs4 import BeautifulSoup
import pandas as pd

def categorize_food(ingredient):
    fruits = list(set(["딸기", "한라봉", "매실", "참외", "복분자", "토마토", "블루베리", "복숭아", "자두", "배", "귤", "석류", "은행", "유자", "사과", "포도", "블루베리", "수박", "복숭아", "참외", "자두"]))  # 과일 리스트
    vegetables = list(set(["우엉", "더덕", "달래", "냉이", "취나물", "쑥", "씀바귀", "두릅", "감자", "옥수수", "도라지", "배추", "고구마", "무", "옥수수", "늙은호박",  "토마토", "감자", "고구마", "도라지", "참나물"]))  # 채소 리스트
    seafood = list(set(["꼬막", "삼치", "명태", "아귀", "도미", "과메기", "바지락", "소라", "주꾸미", "키조개", "참다랑어", "미더덕", "가리비", "장어", "멍게", "다슬기", "해삼", "삼치", "갈치", "굴","게", "광어", "홍합", "꽁치", "", "고등어", "대하",  "전복", "갈치"]))  # 수산물 리스트
    ingredient = ingredient.strip()
    if ingredient in fruits:
        return "fruits"
    elif ingredient in vegetables:
        return "vegetables"
    elif ingredient in seafood:
        return "seafood"
    else:
        return "notclassified"

def get_seasonalfood():
    sfood=[]
    print('Searching naver with 제철음식 >>>>>>>>>>>>')
    URL = 'https://search.naver.com/search.naver?where=nexearch&sm=tab_jum&query=제철음식'
    chrome_ver = chromedriver_autoinstaller.get_chrome_version()
    chromedriver_autoinstaller.install(True)
    chromedriver_path = f'./{chrome_ver.split(".")[0]}/chromedriver.exe'
    wd = webdriver.Chrome()
    wd.get(URL)
    time.sleep(1)
    html = wd.page_source
    soup = BeautifulSoup(html, 'html.parser')
    # change space to .
    # div.food_tab _sfood-filter-food -> div.food_tab._sfood-filter-food
    li_elements = soup.select("div.api_subject_bx > div.api_cs_wrap > div.food_tab._sfood-filter-food > div.stab_area._sfood-rolling-area > ul.stab._sfood-rolling-list > li._sfood-rolling-item")
    sfood = [li.a.text for li in li_elements]
    sfood.pop(0) # remove 전체
    sfood_category = []
    for i in range(len(sfood)):
        category = categorize_food(sfood[i])
        sfood_category.append(category)
    sfood_df = pd.DataFrame({'Category': sfood_category, 'Item': sfood})
    sfruit_df = sfood_df[sfood_df['Category'] == 'fruits']
    sseafood_df = sfood_df[sfood_df['Category'] == 'seafood']
    svegetables_df = sfood_df[sfood_df['Category'] == 'vegetables']
    random_fruit = sfruit_df["Item"].sample(n=1).tolist()
    random_seafood = sseafood_df["Item"].sample(n=1).tolist()
    random_vegetable = svegetables_df["Item"].sample(n=1).tolist()
    return sfood, str(random_fruit), str(random_seafood), str(random_vegetable)

def main():
    # sfood = []
    sfood, random_fruit, random_seafood, random_vegetable = get_seasonalfood() # [CODE 1]
    print(sfood)
    print(f"recommendation: %s, %s, %s " %(random_fruit, random_seafood, random_vegetable))


if __name__ == '__main__':
    main()
