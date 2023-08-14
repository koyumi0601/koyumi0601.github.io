---
layout: single
title: "Crawling without API, static webpage using beautifulSoup"
categories: dataanalysis
tags: [Big Data, Data Analysis, Crawling]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*정적 웹페이지 크롤링*

- install beautifulsoup

```bash
pip install beautifulsoup4
```

- 실습: [http://www.hollys.co.kr](http://www.hollys.co.kr)

- 크롤링 허용 여부 확인 [http://www.hollys.co.kr/robots.txt](http://www.hollys.co.kr/robots.txt)
    - 모든 사용자 에이전트(크롤러)에 적용
    - /membership 경로로 시작하는 URL은 크롤링 금지
    - /myHollys 경로로 시작하는 URL은 크롤링 금지
    - 매장정보[https://www.hollys.co.kr/store/korea/korStore2.do](https://www.hollys.co.kr/store/korea/korStore2.do)는 크롤링 가능 

```
User-agent: *

Disallow: /membership

Disallow: /myHollys
```
- HTML 코드 확인(F12) > tbody

![2023-08-14_17-52-staticwebpage]({{site.url}}/images/$(filename)/2023-08-14_17-52-staticwebpage.png)

```html
<tr class=""> ~ </tr> # 모든 매장 정보
td[0] # 지역
td[1] # 이름
td[2] # 영업여부
td[3] # 주소
td[4] # 첨부 이미지
td[5] # 전화번호
```

- page number - URL 확인

```
https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=1&sido=&gugun=&store= # 1 page
https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=2&sido=&gugun=&store= # 2 page
https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=3&sido=&gugun=&store= # 3 page
# until 52 page
```

- 실습 예제

```py
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import datetime

# [CODE 1]
def hollys_store(result):
    for page in range(1, 59):
        Hollys_url = 'https://www.hollys.co.kr/store/korea/korStore2.do?pageNo=%d&sido=&gugun=&store=' %page
        print(Hollys_url)
        html = urllib.request.urlopen(Hollys_url)
        soupHollys = BeautifulSoup(html, 'html.parser')
        tag_tbody = soupHollys.find('tbody')
        for store in tag_tbody.find_all('tr'):
            if len(store) <= 3:
                break
            store_td = store.find_all('td')
            store_name = store_td[1].string
            store_sido = store_td[0].string
            store_address = store_td[3].string
            store_phone = store_td[5].string
            result.append([store_name]+[store_sido]+[store_address]+[store_phone])
    return

# [CODE 0]
def main():
    result = []
    print('Hollys store crawling >>>>>>>>>>>>')
    hollys_store(result) # [CODE 1]
    hollys_tbl = pd.DataFrame(result, columns = ('store', 'sido-gu', 'address', 'phone'))
    hollys_tbl.to_csv('./hollys1.csv', encoding = 'cp949', mode = 'w', index = True)
    del result[:]

if __name__ == '__main__':
    main()

```


# Reference

- 데이터 과학 기반의 파이썬 빅데이터 분석
