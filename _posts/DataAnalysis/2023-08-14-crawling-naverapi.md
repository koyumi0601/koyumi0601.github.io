---
layout: single
title: "Crawling using Naver API"
categories: dataanalysis
tags: [Big Data, Data Analysis, Crawling]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*네이버 API를 이용한 웹크롤링*



- 네이버 개발자 센터 가입 [https://developers.naver.com/main/](https://developers.naver.com/main/)
- 서비스 API > 검색 > 오픈 API 이용 신청

![2023-08-14_13-08-naver-api]({{site.url}}/images/$(filename)/2023-08-14_13-08-naver-api.png)

- Copy client id and secret

![2023-08-14_13-18-clientid]({{site.url}}/images/$(filename)/2023-08-14_13-18-clientid.png)

- 예제: Documents > 서비스 API > 검색 > API 이용 안내 페이지

```py
# 네이버 검색 API 예제 - 블로그 검색
import os
import sys
import urllib.request
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
encText = urllib.parse.quote("검색할 단어")
url = "https://openapi.naver.com/v1/search/blog?query=" + encText # JSON 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
```



- 네이버 검색 API 개발자 가이드 [https://developers.naver.com/docs/search/news](https://developers.naver.com/docs/search/news)

  - URL
    - 뉴스 [https://openapi.naver.com/v1/search/news.json](https://openapi.naver.com/v1/search/news.json)
    - 블로그 [https://openapi.naver.com/v1/search/blog.json](https://openapi.naver.com/v1/search/blog.json)
    - 카페 [https://openapi.naver.com/v1/search/cafearticle.json](https://openapi.naver.com/v1/search/cafearticle.json)
    - 영화 [https://openapi.naver.com/v1/search/movie.json](https://openapi.naver.com/v1/search/movie.json)
    - 쇼핑 [https://openapi.naver.com/v1/search/shop.json](https://openapi.naver.com/v1/search/shop.json)

  - 요청 변수
    - query: 검색을 원하는 문자열. UTF-8 인코딩
    - start: 검색 시작 위치. 기본 1, 최대 1000
    - display: 검색 출력 건수. 기본 10, 최대 100
  - 응답 변수
    - items: 검색 결과. title, link, originallink, description, pubDate를 포함
    - title: 검색 결과 문서의 제목
    - link: 검색 결과 문서를 제공하는 네이버의 하이퍼텍스트 link
    - originallink: 검색 결과 문서를 제공하는 언론사의 하이퍼텍스트 link
    - description: 검색 결과 문서의 내용 요약
    - pubDate: 검색 결과 문서가 네이버에 제공된 시간





- 실습예제

```py
import os, sys, urllib.request, datetime, time, json
client_id = 'your naver api client id'
client_secret = 'your naver api client secret'

# [CODE 1]
def getRequestUrl(url):
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", client_id)
    req.add_header("X-Naver-Client-Secret", client_secret)
    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            print("[%s] Url Request Success" %datetime.datetime.now())
            return response.read().decode('utf-8')
    except Exception as e:
        print(e)
        print("[%s] Error for URL : %s " %(datetime.datetime.now(), url))
        return None

# [CODE 2]
def getNaverSearch(node, srcText, start, display):
    base = "https://openapi.naver.com/v1/search"
    node = "/%s.json" %node
    parameters = "?query=%s&start=%s&display=%s"%(urllib.parse.quote(srcText), start, display)
    url = base + node + parameters
    responseDecode = getRequestUrl(url) #[CODE 1]
    if (responseDecode == None):
        return None
    else:
        return json.loads(responseDecode)

# [CODE 3]
def getPostData(post, jsonResult, cnt):
    title = post['title']
    description = post['description']
    org_link = post['originallink']
    link = post['link']
    pDate = datetime.datetime.strptime(post['pubDate'], '%a, %d %b %Y %H:%M:%S %z')
    pDate = pDate.strftime('%Y-%m-%d %H:%M:%S')
    jsonResult.append({'cnt': cnt, 'title': title, 'description': description, 'org_link': org_link, 'link': link, 'pDate': pDate })
    return

# [CODE 0]
def main():
    node = 'news'
    srcText = input('검색어를 입력하세요: ') # 월드컵
    cnt = 0
    jsonResult = []
    jsonResponse = getNaverSearch(node, srcText, 1, 100) # [CODE 2]
    total = jsonResponse['total']
    while ((jsonResponse != None) and (jsonResponse['display'] != 0)):
        for post in jsonResponse['items']:
            cnt += 1
            getPostData(post, jsonResult, cnt) # [CODE 3]
        start = jsonResponse['start'] + jsonResponse['display']
        jsonResponse = getNaverSearch(node, srcText, start, 100) # [CODE 2]
    print('전체 검색 : %d 건' %total)

    with open('%s_naver_%s.json' %(srcText, node), 'w', encoding='utf-8') as outfile:
        jsonFile = json.dumps(jsonResult, indent = 4, sort_keys = True, ensure_ascii = False)
        outfile.write(jsonFile)

    print("가져온 데이터 : %d 건" %(cnt))
    print('%s_naver_%s.json SAVED' % (srcText, node))


if __name__ == '__main__':
    main()

```

- Result

![2023-08-14_14-21-naver-api-terminal]({{site.url}}/images/$(filename)/2023-08-14_14-21-naver-api-terminal.png)

![2023-08-14_14-23-naver-api-result]({{site.url}}/images/$(filename)/2023-08-14_14-23-naver-api-result.png)




# Reference

- 데이터 과학 기반의 파이썬 빅데이터 분석