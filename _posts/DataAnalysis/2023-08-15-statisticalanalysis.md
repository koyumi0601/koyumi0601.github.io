---
layout: single
title: "Statistical Analysis"
categories: dataanalysis
tags: [Big Data, Data Analysis, Crawling]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---



*데이터 과학 기반의 파이썬 빅데이터 분석*

# Part 03 빅데이터 분석 - 기본 프로젝트

## Chapter 07 데이터 과학 기반의 빅데이터분석

#### [기술 통계(descriptive statistics)](https://ko.wikipedia.org/wiki/%EA%B8%B0%EC%88%A0%ED%86%B5%EA%B3%84%ED%95%99)

- 요약 통계(Summary statics)
- 평균(mean), 중앙값(median), 최빈값(mode), 표준편차(standard deviation), 사분위(quartile)

```py
import numpy as np
import pandas as pd

data = list(range(1, 11)) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

np.mean(data)
np.median(data)
np.std(data)
np.max(data)
np.min(data)
np.percentile(data, [0, 25, 50, 75, 100]) # 1.0, 3.25, 5.5, 7.75, 10.0

df = pd.DataFrame(data)
print(df.describe())
```

```
              0
count  10.00000
mean    5.50000
std     3.02765
min     1.00000
25%     3.25000
50%     5.50000
75%     7.75000
max    10.00000
```


- 사분위수 계산 방법
    - 데이터 정렬: 데이터 오름차순 정렬
    - 중앙값 (Q2): 데이터의 중앙값. 데이터가 짝수개이므로 중앙의 두 수의 평균. (5 + 6) / 2 = 5.5
    - 제 1 사분위수 (Q1): 중앙값의 왼쪽 부분 [1, 2, 3, 4, 5]에서 중앙의 양측 값 3, 4를 선형보간하여 25% 지점 값을 구한다. 3.25
    - 제 3 사분위수 (Q3): 중앙값의 오른쪽 부분 [6, 7, 8, 9, 10]에서 중앙의 양측 값 7, 8을 선형보간하여 75% 지점 값을 구한다. 7.75
    - 최소값: 1
    - 최대값: 10



#### [회귀 분석(Regression analysis)](https://ko.wikipedia.org/wiki/%ED%9A%8C%EA%B7%80_%EB%B6%84%EC%84%9D)

- x: independent variable, y: dependent variable
- 단순 회귀(변수 1개) / 다중 회귀(2개 이상)
- **선형 회귀** / 비선형 회귀

$$ y = b_0 + b_1 x_1 + b_2 x_2 + ... + b_n x_n $$

- 머신러닝에서의 선형회귀 활용
  - [https://www.youtube.com/watch?v=ve6gtpZV83E](https://www.youtube.com/watch?v=ve6gtpZV83E)

![2023-08-15_11-16-linearRegression]({{site.url}}/images/$(filename)/2023-08-15_11-16-linearRegression.png)

*모델은 선형방정식으로 가정한다*

*cost는 최소제곱법으로 구한다*

![2023-08-15_11-16-linearRegression2]({{site.url}}/images/$(filename)/2023-08-15_11-16-linearRegression2.png)

*최적 값은, cost 함수에서, w, b에 따른 각각의 기울기(편미분)가 0이 되게 하는 값이다*.

*참고로 영상에서 b의 편미분 계산 수식은 틀렸다*

![2023-08-15_11-16-linearRegression3]({{site.url}}/images/$(filename)/2023-08-15_11-16-linearRegression3.png)

*inital w, initial b를 설정한 후, 반복적으로 cost function을 계산.* 

*예측 값을 업데이트(epoch) - 다음 예측 값은 현재 예측 값에 기울기에 learning rate를 곱하여 더해준다*

![2023-08-15_11-16-linearRegression4]({{site.url}}/images/$(filename)/2023-08-15_11-16-linearRegression4.png)

*참고로, 다른 수학적 방법으로도 풀 수 있다*

*행렬식 형태로 변환하고, 역행렬을 이용하여 풀면, w는 공분산/분산 형태로 정리가 가능하다.* 

*b는 y의 기대값(평균) - x의 기대값(평균) * w로 나타낼 수 있다*

![2023-08-15_11-16-linearRegression5]({{site.url}}/images/$(filename)/2023-08-15_11-16-linearRegression5.png)

*단순 선형회귀 모델의 경우 경사하강법 아닌 다른 수학적 방법으로도 w, b를 풀 수 있지만 실제 데이터는 단순 선형이 아니기 때문에 주로 경사하강법(Gradient descent)을 사용한다.*



#### t-검정


- 모집단의 분산이나 표준편차를 알지 못할 때 모집단을 대표하는 표본으로부터 추정된 분산이나 표준편차를 가지고 검정하는 방법으로 “두 모집단의 평균간의 차이는 없다”라는 귀무가설과 “두 모집단의 평균 간에 차이가 있다”라는 대립가설 중에 하나를 선택할 수 있도록 하는 통계적 검정방법이다.
- t 검정의 특징

![2023-08-15_12-46-ttest3]({{site.url}}/images/$(filename)/2023-08-15_12-46-ttest3.png)


- t 값: t값이란 t 검정에 이용되는 검정통계량으로, 두 집단의 차이의 평균(X)을 표준오차(SE)로 나눈 값 즉, [표준오차]와 [표본평균사이의 차이]의 비율이다.



$$ t= \frac{\overline{X}-\mu}{\frac{S}{\sqrt n}} $$

$$ \overline(X) : 두 집단 차이의 평균 $$

$$ \mu: 모집단 평균 $$

$$ S: 두 집단 차이의 표준편차 $$




- 귀무가설(Null Hypothesis): 

  - 처음부터 버릴 것을 예상하는 가설. 
  - 두 집단 간의 평균 차이는 없을 것이다. 
  - 유의 수준(α)을 0.05라고 가정했을 때, t값이 커져서 (평균차이가 있을 가능성이 커져서) 기각역에 존재하여 유의확률(p값, p-value)이 0.05보다 작으면 평균 차이가 유의미한 것으로 해석되어 귀무가설을 기각한다.그 반대의 경우, 평균 차이가 유의미하지 않으므로 귀무가설을 수용한다.



![2023-08-15_12-46-ttest2]({{site.url}}/images/$(filename)/2023-08-15_12-46-ttest2.png)

- 기각역(Critical Region): 귀무가설이 기각되기 위한 검정통계량(t값)이 위치하는 범위로, 면적=α (유의수준)과 자유도(n-1)에 의해 결정된다. 단측검정(one-tailed test)의 경우 기각역이 한 쪽에 존재하고, 양측검정(two-tailed test)의 경우 기각역이 양쪽에 존재한다.

- 대립가설(Alternative Hypothesis): 
  - 처음부터 채택할 것을 예상하는 가설.
  - 두 집단 간의 평균 차이는 있을 것이다.

- t 검정의 종류

![2023-08-15_12-37-ttest]({{site.url}}/images/$(filename)/2023-08-15_12-37-ttest.png)



##### 대응표본 t 검정 (Paired t test)

- 대응표본 t 검정 순서

  - 가설 설정 > 단순 통계량 계산 > 검정통계량 계산 > 기각역 설정 > 결론

  - 표본

    ```py
                                                             n1          n2          n3          n4          n5
    score before education                           135.000000  136.000000  138.000000  125.000000  150.000000
    score after education                            140.000000  138.000000  142.000000  126.000000  148.000000
    ```

  - 단순통계량 계산

    ```python
                                                             n1          n2          n3          n4          n5
    score before education                           135.000000  136.000000  138.000000  125.000000  150.000000
    score after education                            140.000000  138.000000  142.000000  126.000000  148.000000
    difference                                         5.000000    2.000000    4.000000    1.000000   -2.000000
    average of difference                              2.000000    2.000000    2.000000    2.000000    2.000000
    difference - average difference                    3.000000    0.000000    2.000000   -1.000000   -4.000000
    power of difference - average difference           9.000000    0.000000    4.000000    1.000000   16.000000
    sum of power of difference - average difference   30.000000   30.000000   30.000000   30.000000   30.000000
    sample variance                                    7.500000    7.500000    7.500000    7.500000    7.500000
    sample std                                         2.738613    2.738613    2.738613    2.738613    2.738613
    ```

    - sample variance = 30 / (5-1)

  - 검정통계량 $$ t = \frac{\overline{X}-\mu}{\frac{S}{\sqrt n}} $$ $$ =  \frac{2}{\frac{2.74}{\sqrt 5}} $$ $$ = 1.64 $$

  - 기각역 설정

    ![2023-08-15_12-46-ttest4]({{site.url}}/images/$(filename)/2023-08-15_12-46-ttest4.png)

    ![2023-08-15_12-46-ttest5]({{site.url}}/images/$(filename)/2023-08-15_12-46-ttest5.png)

  - 결론

    ![2023-08-15_13-476]({{site.url}}/images/$(filename)/2023-08-15_13-476.png)



#### [히스토그램](https://ko.wikipedia.org/wiki/%ED%9E%88%EC%8A%A4%ED%86%A0%EA%B7%B8%EB%9E%A8)

- 도수분포표를 그래프로 나타낸 것







#### 실습예제

- 와인 데이터셋, 어바인 대학 머신러닝 저장소 [https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

- 입력변수
  - fixed acidity(고정산)
  - volatile acidity(휘발산)
  - critiric acid(구연산)
  - residual sugar(잔당)
  - chlorides(염화물)
  - free sulfur dioxide(유리 이산화황)
  - total sulfur dioxide(총 이산화황)
  - density(밀도)
  - pH
  - sulphates(황산염)
  - alcohol(알코올)
- 출력변수
  - Quality
- 







# Reference
- 데이터 과학 기반의 파이썬 빅데이터 분석
- Blog [https://m.blog.naver.com/PostView.naver?blogId=kunyoung90&logNo=222082820599&categoryNo=24&proxyReferer=](https://m.blog.naver.com/PostView.naver?blogId=kunyoung90&logNo=222082820599&categoryNo=24&proxyReferer=)
- 사분위수 계산하기 [https://ko.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/interquartile-range-iqr/a/interquartile-range-review](https://ko.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/interquartile-range-iqr/a/interquartile-range-review)
- 사분위수 계산하기 2 [http://tomoyo.ivyro.net/123/wiki.php/%EC%82%AC%EB%B6%84%EC%9C%84%EC%88%98%2Cquartile](http://tomoyo.ivyro.net/123/wiki.php/%EC%82%AC%EB%B6%84%EC%9C%84%EC%88%98%2Cquartile)

- 딥러닝 1-1강. 선형 회귀 [https://www.youtube.com/watch?v=IJRxpLgT7oE&t=33s](https://www.youtube.com/watch?v=IJRxpLgT7oE&t=33s)

- 통계교육블로그, T 검정 정의 [https://m.blog.naver.com/sendmethere/221333164258](https://m.blog.naver.com/sendmethere/221333164258)
- 통계교육블로그, T 검정 예시 [https://m.blog.naver.com/sendmethere/221333267001](https://m.blog.naver.com/sendmethere/221333267001)