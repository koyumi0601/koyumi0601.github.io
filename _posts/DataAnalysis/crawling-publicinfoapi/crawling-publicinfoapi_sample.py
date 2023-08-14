import requests

url = 'http://openapi.tour.go.kr/openapi/service/EdrcntTourismStatsService/getEdrcntTourismStatsList'
ServiceKey = "" # 공공데이터포털 한국문화관광연구원 출입국관광통계서비스 일반 인증키 Encoding

params ={'serviceKey' : '서비스키', 'YM' : '201201', 'NAT_CD' : '112', 'ED_CD' : 'E' }

response = requests.get(url, params=params)
print(response.content)
