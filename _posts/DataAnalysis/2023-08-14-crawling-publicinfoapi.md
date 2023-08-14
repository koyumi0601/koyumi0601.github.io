---
layout: single
title: "Crawling using Public Information API"
categories: dataanalysis
tags: [Big Data, Data Analysis, Crawling]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*공공데이터 API 기반 크롤링*

- 공공데이터포털 가입 [https://www.data.go.kr/](https://www.data.go.kr/)

- 검색: 출입국관광통계서비스 > 오픈 API(xxxx) > xml 한국문화관광연구원_출입국관광통계서비스 > 활용신청

- 연구(논문 등) > 공공데이터 활용 학습 > 상세기능정보 선택 > 출입국관광통계조회 > 라이선스 표시 > 동의합니다 > 활용신청

- 일반인증키 (복사 필요)

- 사용방법 .docs 로 제공

- [오픈 API 상세](https://www.data.go.kr/tcs/dss/selectApiDataDetailView.do?publicDataPk=15000297) > 상세기능 > 서비스 URL [http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList](http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList)

- 요청 변수: 서비스 URL 뒤에 추가할 매개변수 항목

- 출력 결과: 크롤링 결과로 받을 데이터 항목

- 샘플 코드:

```py
# Python3 샘플 코드 #
import requests

url = 'http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList'
params ={'serviceKey' : '서비스키', 'YM' : '201201', 'NAT_CD' : '112', 'ED_CD' : 'E' }

response = requests.get(url, params=params)
print(response.content)
```

- 연습 코드

```py
import os, sys, urllib.request, datetime, time, json, requests
import pandas as pd
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET

ServiceKey = "공공데이터포털 한국문화관광연구원 출입국관광통계서비스 일반 인증키 Encoding"

# [CODE 1]
def getRequestUrl(url):
    req = urllib.request.Request(url)
    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            print("[%s] Url Request Success" % datetime.datetime.now())
            return response.read().decode('utf-8')
    except Exception as e:
        print(e)
        print("[%s] Error for URL : %s" % (datetime.datetime.now(), url))
        return None
    
# [CODE 2]
def getTourismStatsItem(yyyymm, national_code, ed_cd):
    service_url ='http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList'
    parameters = '?YM=' + yyyymm
    parameters += '&NAT_CD=' + national_code
    parameters += '&ED_CD' + ed_cd
    parameters += '&serviceKey=' + ServiceKey
    url = service_url + parameters
    params ={'serviceKey' : ServiceKey, 'YM' : yyyymm, 'NAT_CD' : national_code, 'ED_CD' : ed_cd }
    response = requests.get(url, params=params)
    decoded_content = response.content.decode('utf-8')
    root = ET.fromstring(decoded_content)
    result_msg = root.find('./header/resultMsg').text
    natName = root.find('./body/items/item/natKorNm').text.replace(' ', '')
    num = root.find('./body/items/item/num').text
    ed = root.find('./body/items/item/ed').text
    return result_msg, natName, num, ed

# [CODE 3]
def getTourismStatsService(nat_cd, ed_cd, nStartYear, nEndYear):
    jsonResult = []
    result = []
    natName = ''
    dataEND = "{0}{1:0>2}".format(str(nEndYear), str(12))
    isDataEnd = 0
    for year in range(nStartYear, nEndYear+1):
        for month in range(1, 13):
            if(isDataEnd == 1): break
            yyyymm = "{0}{1:0>2}".format(str(year), str(month))
            result_msg, natName, num, ed = getTourismStatsItem(yyyymm, nat_cd, ed_cd) #[CODE 2]
            print('[%s_%s : %s ]' %(natName, yyyymm, num))
            jsonResult.append({'nat_name': natName, 'nat_cd': nat_cd, 'yyyymm': yyyymm, 'visit_cnt': num})
            result.append([natName, nat_cd, yyyymm, num])
    return (jsonResult, result, natName, ed, dataEND)

# [CODE 0]
def main():
    jsonResult = []
    result = []
    print("<< 국내 입국한 외국인의 통계 데이터를 수집합니다. >>")
    nat_cd = input('국가 코드를 입력하세요 (중국: 112 / 일본: 130 / 미국: 275) : ')
    nStartYear = int(input('데이터를 몇 년부터 수집할까요? : '))
    nEndYear = int(input('데이터를 몇 년까지 수집할까요? : '))
    ed_cd = "E" # E : 방한외래관광객, D : 해외 출국
    jsonResult, result, natName, ed, dataEND = getTourismStatsService(nat_cd, ed_cd, nStartYear, nEndYear) # [CODE 3]

    # 파일 저장 : csv 파일
    columns = ["입국자국가", "국가코드", "입국연월", "입국자 수"]
    result_df = pd.DataFrame(result, columns = columns)
    result_df.to_csv('./%s_%s%d%s.csv' % (natName, ed, nStartYear, dataEND), index=False, encoding='cp949')

if __name__ == '__main__':
    main()
              
```

# Reference

- 데이터 과학 기반의 파이썬 빅데이터 분석
    - 공공데이터 API 데이터 양식이 변경된 것 같아서 코드 수정