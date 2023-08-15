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
    chrome_ver = chromedriver_autoinstaller.get_chrome_version()
    chromedriver_autoinstaller.install(True)
    chromedriver_path = f'./{chrome_ver.split(".")[0]}/chromedriver.exe'
    wd = webdriver.Chrome()

    for i in range(1, 370):
        wd.get(CoffeeBean_URL)
        time.sleep(3)
        try:
            wd.execute_script("storePop2(%s);" %str(i))
            time.sleep(3)
            html = wd.page_source
            soupCB = BeautifulSoup(html, 'html.parser')
            store_name_h2 = soupCB.select("div.store_txt > h2")
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

