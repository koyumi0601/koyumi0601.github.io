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
    # Refer to API Docs.
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
              
