import time
from selenium import webdriver
import chromedriver_autoinstaller
from bs4 import BeautifulSoup
import pandas as pd
import random

def get_food_ranking(type):
    sfood=[]
    print('Searching 10000 recipe rank >>>>>>>>>>>>')
    if type=="찌개":
        # 찌개 / 전체 / 전체 / 전체
        URL = "https://www.10000recipe.com/recipe/list.html?q=&query=&cat1=&cat2=&cat3=&cat4=55&fct=&order=reco&lastcate=cat2&dsearch=&copyshot=&scrap=&degree=&portion=&time=&niresource="
    elif type=="메인:소":
        # 메인반찬 / 일상 / 소, 돼지, 닭, 해물, 달걀 유제품, 콩
        URL = "https://www.10000recipe.com/recipe/list.html?q=&query=&cat1=&cat2=12&cat3=70&cat4=56&fct=&order=reco&lastcate=cat3&dsearch=&copyshot=&scrap=&degree=&portion=&time=&niresource="
    elif type=="메인:돼지":
        URL = 'https://www.10000recipe.com/recipe/list.html?q=&query=&cat1=&cat2=12&cat3=71&cat4=56&fct=&order=reco&lastcate=cat3&dsearch=&copyshot=&scrap=&degree=&portion=&time=&niresource='
    elif type=="메인:닭":
        URL = "https://www.10000recipe.com/recipe/list.html?q=&query=&cat1=&cat2=12&cat3=72&cat4=56&fct=&order=reco&lastcate=cat3&dsearch=&copyshot=&scrap=&degree=&portion=&time=&niresource="
    elif type=="메인:해물":
        URL = "https://www.10000recipe.com/recipe/list.html?q=&query=&cat1=&cat2=12&cat3=24&cat4=56&fct=&order=reco&lastcate=cat3&dsearch=&copyshot=&scrap=&degree=&portion=&time=&niresource="
    elif type=="메인:달걀":
        URL = "https://www.10000recipe.com/recipe/list.html?q=&query=&cat1=&cat2=12&cat3=50&cat4=56&fct=&order=reco&lastcate=cat3&dsearch=&copyshot=&scrap=&degree=&portion=&time=&niresource="
    elif type=="메인:콩":
        URL = "https://www.10000recipe.com/recipe/list.html?q=&query=&cat1=&cat2=12&cat3=27&cat4=56&fct=&order=reco&lastcate=cat3&dsearch=&copyshot=&scrap=&degree=&portion=&time=&niresource="        
    elif type=="밑반찬":
        # 밑반찬 / 일상 / 전체 / 전체
        URL = "https://www.10000recipe.com/recipe/list.html?q=&query=&cat1=&cat2=12&cat3=&cat4=63&fct=&order=reco&lastcate=cat2&dsearch=&copyshot=&scrap=&degree=&portion=&time=&niresource="
    elif type=="다이어트":
        URL = "https://www.10000recipe.com/recipe/list.html?q=&query=&cat1=&cat2=21&cat3=&cat4=64&fct=&order=reco&lastcate=cat3&dsearch=&copyshot=&scrap=&degree=&portion=&time=&niresource="

    chrome_ver = chromedriver_autoinstaller.get_chrome_version()
    chromedriver_autoinstaller.install(True)
    chromedriver_path = f'./{chrome_ver.split(".")[0]}/chromedriver.exe'
    wd = webdriver.Chrome()
    wd.get(URL)
    time.sleep(1)
    html = wd.page_source
    soup = BeautifulSoup(html, 'html.parser')
    li_elements = soup.select("ul.tag_cont")
    li_tags = soup.find('ul', class_='tag_cont').find_all('li')
    item_list = [li.a.get_text() for li in li_tags]
    random_item = random.choice(item_list)
    print(random_item)
    return random_item

def main():
    type = "찌개" # "찌개", "메인:소", 메인:돼지", "메인:닭", "메인:해물", "메인:달걀", "밑반찬", "다이어트"
    random_food = get_food_ranking(type) # [CODE 1]


if __name__ == '__main__':
    main()
