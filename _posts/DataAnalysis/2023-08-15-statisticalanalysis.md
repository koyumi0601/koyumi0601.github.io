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


```python
import pandas as pd

wine_df = pd.read_csv('./wine.csv', sep=',', header=0, engine='python')
wine_df.info()
```

```
[5 rows x 13 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6497 entries, 0 to 6496
Data columns (total 13 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   type                  6497 non-null   object 
 1   fixed acidity         6497 non-null   float64
 2   volatile acidity      6497 non-null   float64
 3   citric acid           6497 non-null   float64
 4   residual sugar        6497 non-null   float64
 5   chlorides             6497 non-null   float64
 6   free sulfur dioxide   6497 non-null   float64
 7   total sulfur dioxide  6497 non-null   float64
 8   density               6497 non-null   float64
 9   pH                    6497 non-null   float64
 10  sulphates             6497 non-null   float64
 11  alcohol               6497 non-null   float64
 12  quality               6497 non-null   int64  
dtypes: float64(11), int64(1), object(1)
memory usage: 660.0+ KB
None
```

```python 
wine_df.columns = wine_df.columns.str.replace(' ', '')
wine_df.describe()
```

```
       fixedacidity  volatileacidity   citricacid  residualsugar    chlorides  ...      density           pH    sulphates      alcohol      quality
count   6497.000000      6497.000000  6497.000000    6497.000000  6497.000000  ...  6497.000000  6497.000000  6497.000000  6497.000000  6497.000000
mean       7.215307         0.339666     0.318633       5.443235     0.056034  ...     0.994697     3.218501     0.531268    10.491801     5.818378
std        1.296434         0.164636     0.145318       4.757804     0.035034  ...     0.002999     0.160787     0.148806     1.192712     0.873255
min        3.800000         0.080000     0.000000       0.600000     0.009000  ...     0.987110     2.720000     0.220000     8.000000     3.000000
25%        6.400000         0.230000     0.250000       1.800000     0.038000  ...     0.992340     3.110000     0.430000     9.500000     5.000000
50%        7.000000         0.290000     0.310000       3.000000     0.047000  ...     0.994890     3.210000     0.510000    10.300000     6.000000
75%        7.700000         0.400000     0.390000       8.100000     0.065000  ...     0.996990     3.320000     0.600000    11.300000     6.000000
max       15.900000         1.580000     1.660000      65.800000     0.611000  ...     1.038980     4.010000     2.000000    14.900000     9.000000

[8 rows x 12 columns]
```

```python
sorted(wine_df.quality.unique()) # get unique value of quality
```

```
[3, 4, 5, 6, 7, 8, 9]
```

```python
wine_df.quality.value_counts()
```

```
quality
6    2836
5    2138
7    1079
4     216
8     193
3      30
9       5
Name: count, dtype: int64
```

```python
wine_df.groupby('type')['quality'].describe()
```

```
        count      mean       std  min  25%  50%  75%  max
type                                                      
red    1599.0  5.636023  0.807569  3.0  5.0  6.0  6.0  8.0
white  4898.0  5.877909  0.885639  3.0  5.0  6.0  6.0  9.0
```

```python
wine_df.groupby('type')['quality'].mean()
```

```
type
red      5.636023
white    5.877909
Name: quality, dtype: float64
```

```python
wine_df.groupby('type')['quality'].std()
```

```
type
red      0.807569
white    0.885639
Name: quality, dtype: float64
```

```python
wine_df.groupby('type')['quality'].agg(['mean', 'std'])
```

```
           mean       std
type                     
red    5.636023  0.807569
white  5.877909  0.885639
```


- t-test

```bash
pip install statsmodels
```

```python
from scipy import stats
from statsmodels.formula.api import ols, glm
import pandas as pd

# data
wine_df = pd.read_csv('./wine.csv', sep=',', header=0, engine='python') # load data
wine_df.columns = wine_df.columns.str.replace(' ', '_') # replace space to _ in column name for Rformula
red_wine_quality = wine_df.loc[wine_df['type'] == 'red', 'quality'] # red wine quality
white_wine_quality = wine_df.loc[wine_df['type'] == 'white', 'quality'] # white wine quality

# analysis
stats.ttest_ind(red_wine_quality, white_wine_quality, equal_var = False) # scipy stats, t-test, equal_var=False assuming 2 dataset have different variance
Rformula = 'quality ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol' # dependent variable: quality, independent variables: others
regression_result = ols(Rformula, data=wine_df).fit() # ordinary least squares
print(regression_result.summary())

```

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                quality   R-squared:                       0.292
Model:                            OLS   Adj. R-squared:                  0.291
Method:                 Least Squares   F-statistic:                     243.3
Date:                Tue, 15 Aug 2023   Prob (F-statistic):               0.00
Time:                        16:06:35   Log-Likelihood:                -7215.5
No. Observations:                6497   AIC:                         1.445e+04
Df Residuals:                    6485   BIC:                         1.454e+04
Df Model:                          11                                         
Covariance Type:            nonrobust                                         
========================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
----------------------------------------------------------------------------------------
Intercept               55.7627     11.894      4.688      0.000      32.447      79.079
fixed_acidity            0.0677      0.016      4.346      0.000       0.037       0.098
volatile_acidity        -1.3279      0.077    -17.162      0.000      -1.480      -1.176
citric_acid             -0.1097      0.080     -1.377      0.168      -0.266       0.046
residual_sugar           0.0436      0.005      8.449      0.000       0.033       0.054
chlorides               -0.4837      0.333     -1.454      0.146      -1.136       0.168
free_sulfur_dioxide      0.0060      0.001      7.948      0.000       0.004       0.007
total_sulfur_dioxide    -0.0025      0.000     -8.969      0.000      -0.003      -0.002
density                -54.9669     12.137     -4.529      0.000     -78.760     -31.173
pH                       0.4393      0.090      4.861      0.000       0.262       0.616
sulphates                0.7683      0.076     10.092      0.000       0.619       0.917
alcohol                  0.2670      0.017     15.963      0.000       0.234       0.300
==============================================================================
Omnibus:                      144.075   Durbin-Watson:                   1.646
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              324.712
Skew:                          -0.006   Prob(JB):                     3.09e-71
Kurtosis:                       4.095   Cond. No.                     2.49e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.49e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
```

- Dep. Variable (종속 변수): 응답 또는 종속 변수로 사용된 'quality'입니다.
- R-squared (결정 계수): 모델이 데이터의 변동성을 얼마나 잘 설명하는지 나타내는 지표입니다. 0에서 1 사이의 값으로, 높을수록 모델이 데이터를 잘 설명한다는 의미입니다.
- Adj. R-squared (조정된 결정 계수): 독립 변수의 수를 고려하여 결정 계수를 조정한 값입니다.
- F-statistic (F-통계량): 회귀 모델 전체가 통계적으로 유의한지 검정하는 값입니다.
- Prob (F-statistic) (F-통계량의 확률): F-통계량의 p-value로, 모델 전체의 유의성을 나타냅니다.
- AIC & BIC: 모델의 적합도를 평가하는 정보 기준입니다. 낮은 값이 더 좋은 모델을 나타냅니다.
- Coefficients (계수): 각 독립 변수의 회귀 계수입니다. 이 값들은 해당 변수가 종속 변수에 미치는 영향을 나타냅니다.
- coef: 추정된 회귀 계수입니다.
- std err: 계수의 표준 오차입니다.
- t: t-통계량으로, 계수의 유의성을 검정합니다.
- P>|t|: t-통계량의 p-value로, 계수의 유의성을 나타냅니다.
- [0.025 0.975]: 계수의 95% 신뢰 구간입니다.
- Omnibus & Prob(Omnibus) (전체 검정): 잔차의 정규성을 검정하는 통계량과 p-value입니다.
- Skew (왜도): 잔차의 비대칭도를 나타냅니다.
- Kurtosis (첨도): 잔차의 뾰족함을 나타냅니다.
- Durbin-Watson: 잔차의 자기 상관을 검정하는 통계량입니다.
- Cond. No (조건 번호): 다중공선성(multicollinearity)을 나타내는 지표로, 높은 값은 변수 간의 높은 상관 관계를 나타냅니다.



# Reference
- 데이터 과학 기반의 파이썬 빅데이터 분석
- Blog [https://m.blog.naver.com/PostView.naver?blogId=kunyoung90&logNo=222082820599&categoryNo=24&proxyReferer=](https://m.blog.naver.com/PostView.naver?blogId=kunyoung90&logNo=222082820599&categoryNo=24&proxyReferer=)
- 사분위수 계산하기 [https://ko.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/interquartile-range-iqr/a/interquartile-range-review](https://ko.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/interquartile-range-iqr/a/interquartile-range-review)
- 사분위수 계산하기 2 [http://tomoyo.ivyro.net/123/wiki.php/%EC%82%AC%EB%B6%84%EC%9C%84%EC%88%98%2Cquartile](http://tomoyo.ivyro.net/123/wiki.php/%EC%82%AC%EB%B6%84%EC%9C%84%EC%88%98%2Cquartile)

- 딥러닝 1-1강. 선형 회귀 [https://www.youtube.com/watch?v=IJRxpLgT7oE&t=33s](https://www.youtube.com/watch?v=IJRxpLgT7oE&t=33s)

- 통계교육블로그, T 검정 정의 [https://m.blog.naver.com/sendmethere/221333164258](https://m.blog.naver.com/sendmethere/221333164258)
- 통계교육블로그, T 검정 예시 [https://m.blog.naver.com/sendmethere/221333267001](https://m.blog.naver.com/sendmethere/221333267001)