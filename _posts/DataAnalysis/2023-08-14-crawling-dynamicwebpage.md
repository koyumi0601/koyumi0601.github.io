---
layout: single
title: "Crawling without API, dynamic webpage using selenium"
categories: dataanalysis
tags: [Big Data, Data Analysis, Crawling]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*동적 웹페이지 크롤링*

- install chrome browser

```bash
sudo apt-get update
sudo apt-get install -y google-chrome-stable
```

- download chrome driver matched with chrome browser

```bash
pip install chromedriver-autoinstaller
```

- install selenium

```bash
pip install selenium
```

- Check with python script

```py
import os
from selenium import webdriver
import chromedriver_autoinstaller

chrome_ver = chromedriver_autoinstaller.get_chrome_version()
print(chrome_ver)
chromedriver_autoinstaller.install(True)
chromedriver_path = f'./{chrome_ver.split(".")[0]}/chromedriver.exe'

url = 'https://google.com'
driver = webdriver.Chrome()
driver.get(url)
driver.implicitly_wait(3)
```

- 실습 예제

- 커피빈 매장찾기 [https://www.coffeebeankorea.com/store/store.asp](https://www.coffeebeankorea.com/store/store.asp)

- javascript 함수 확인: javascript:storeLocal2('서울')

![2023-08-14_19-04-dynamicwebpage]({{site.url}}/images/$(filename)/2023-08-14_19-04-dynamicwebpage.png)

- HTML source를 확인(Ctrl + U)해도 매장 목록이 없다.
- 자세히 보기 javascript 함수 확인: javascript:storePop2('31')

![2023-08-14_19-10-dynamicwebpage2]({{site.url}}/images/$(filename)/2023-08-14_19-10-dynamicwebpage2.png)

![2023-08-14_19-12-dynamicwebpage-detail]({{site.url}}/images/$(filename)/2023-08-14_19-12-dynamicwebpage-detail.png)

- F12 개발자모드에서 요소 검사모드를 클릭

![2023-08-14_20-22-static-popup]({{site.url}}/images/$(filename)/2023-08-14_20-22-static-popup.png)





- 실습 예제

```py
import os, time
from selenium import webdriver
import chromedriver_autoinstaller
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import datetime


# chrome_ver = chromedriver_autoinstaller.get_chrome_version()
# chromedriver_autoinstaller.install(True)
# chromedriver_path = f'./{chrome_ver.split(".")[0]}/chromedriver.exe'
# url = 'https://www.coffeebeankorea.com/store/store.asp'
# wd = webdriver.Chrome()
# wd.get(url)
# wd.execute_script("storePop2('31');")
# time.sleep(3)
# html = wd.page_source
# soupCB = BeautifulSoup(html, 'html.parser')
# store_name_h2 = soupCB.select("div.store_txt>h2")[0].string
# store_info = soupCB.select("div.store_txt>table.store_table>tbody>tr>td")
# store_address_list = list(store_info[2])
# store_address = store_address_list[0]
# store_phone=store_info[3].string
# print(store_name_h2)
# print(store_address)
# print(store_phone)

# [CODE 1]
def CoffeeBean_store(result):
    CoffeeBean_URL = 'https://www.coffeebeankorea.com/store/store.asp'
    chrome_ver = chromedriver_autoinstaller.get_chrome_version() # chrome driver version should be same as chrome version
    chromedriver_autoinstaller.install(True)
    chromedriver_path = f'./{chrome_ver.split(".")[0]}/chromedriver.exe'
    wd = webdriver.Chrome()

    for i in range(1, 370):
        wd.get(CoffeeBean_URL)
        time.sleep(3) # wait page response. need to be changed
        try:
            wd.execute_script("storePop2(%s);" %str(i)) # javascript when click detail
            time.sleep(3)
            html = wd.page_source
            soupCB = BeautifulSoup(html, 'html.parser')
            store_name_h2 = soupCB.select("div.store_txt > h2") # check page source, F12
            store_name = store_name_h2[0].string
            print(store_name)
            store_info = soupCB.select("div.store_txt>table.store_table>tbody>tr>td")
            store_address_list = list(store_info[2])
            store_address = store_address_list[0]
            store_phone = store_info[3].string
            result.append([store_name]+[store_address]+[store_phone])
        except:
            continue
    return

# [CODE 0]
def main():
    result = []
    print('CoffeeBean store crawling >>>>>>>>>>>"')
    CoffeeBean_store(result) # [CODE 1]

    CB_tbl = pd.DataFrame(result, columns = ('store', 'address', 'phone'))
    CB_tbl.to_csv('./CoffeeBean.csv', encoding='cp949', mode ='w', index = True)

if __name__ == '__main__':
    main()

```






# Reference

- 데이터 과학 기반의 파이썬 빅데이터 분석
- Blog [웹크롤링 11 동적 페이지](https://charimlab.tistory.com/entry/ep01%EC%9B%B9%ED%81%AC%EB%A1%A4%EB%A7%81-11-%EB%8F%99%EC%A0%81-%ED%8E%98%EC%9D%B4%EC%A7%80%EC%9B%B9-%EB%8F%99%EC%9E%91-%EC%9E%90%EB%8F%99%ED%99%94Selenium-with-%ED%8C%8C%EC%9D%B4%EC%8D%AC)

- Blog [https://dachata.com/google-tag-manager-tips/post/track-non-url-popup-open/](https://dachata.com/google-tag-manager-tips/post/track-non-url-popup-open/)