---
layout: single
title: "Data Analysis"
categories: dataanalysis
tags: [Big Data, Data Analysis, Crawling]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

*데이터 과학 기반의 파이썬 빅데이터 분석*


```python
import json
site = {"http://data.go.kr":"한국 정부에서 제공하는 공공데이터",
"http://kostat.go.kr": "한국 통계청에서 공개하는 데이터",
"http://opendata.hira.or.kr": "한국 보건 의료 빅데이터 개방 시스템",
"http://www.localdata.kr": "한국 지방행정 인허가 데이터",
"https://www.mcst.go.kr": "한국 문화체육관광부 문화 데이터",
"http://data.seoul.go.kr": "서울시 열린데이터 광장",
"https://data.gg.go.kr": "경기도 공공데이터 개방 포털",
"http://data.gov": "미국 정부의 공공데이터",
"http://data.worldbank.org": "세계 은행에서 제공하는 개방 데이터",
"http://open.fda.gov": "미국 식약청의 개방 데이터",
"http://naver.com":"naver",
"https://twitter.com":"twitter",
"https://www.facebook.com":"meta"
}
print(json.dumps(site, indent=4, ensure_ascii=False))
```

    {
        "http://data.go.kr": "한국 정부에서 제공하는 공공데이터",
        "http://kostat.go.kr": "한국 통계청에서 공개하는 데이터",
        "http://opendata.hira.or.kr": "한국 보건 의료 빅데이터 개방 시스템",
        "http://www.localdata.kr": "한국 지방행정 인허가 데이터",
        "https://www.mcst.go.kr": "한국 문화체육관광부 문화 데이터",
        "http://data.seoul.go.kr": "서울시 열린데이터 광장",
        "https://data.gg.go.kr": "경기도 공공데이터 개방 포털",
        "http://data.gov": "미국 정부의 공공데이터",
        "http://data.worldbank.org": "세계 은행에서 제공하는 개방 데이터",
        "http://open.fda.gov": "미국 식약청의 개방 데이터",
        "http://naver.com": "naver",
        "https://twitter.com": "twitter",
        "https://www.facebook.com": "meta"
    }
    


```python

```
