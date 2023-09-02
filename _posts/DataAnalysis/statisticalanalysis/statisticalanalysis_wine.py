import pandas as pd

# from rawdata, generate wine.csv
# red_df = pd.read_csv('./winequality-red.csv', sep=';', header=0, engine='python')
# white_df = pd.read_csv('./winequality-white.csv', sep=';', header=0, engine='python')
# red_df.insert(0, column='type', value='red')
# white_df.insert(0, column='type', value='white')
# wine_df = pd.concat([red_df, white_df])
# wine_df.to_csv('./wine.csv', index=False)

# laod wine.csv
wine_df = pd.read_csv('./wine.csv', sep=',', header=0, engine='python')
print(wine_df.head())
print(wine_df.info())

# [5 rows x 13 columns]
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 6497 entries, 0 to 6496
# Data columns (total 13 columns):
#  #   Column                Non-Null Count  Dtype  
# ---  ------                --------------  -----  
#  0   type                  6497 non-null   object 
#  1   fixed acidity         6497 non-null   float64
#  2   volatile acidity      6497 non-null   float64
#  3   citric acid           6497 non-null   float64
#  4   residual sugar        6497 non-null   float64
#  5   chlorides             6497 non-null   float64
#  6   free sulfur dioxide   6497 non-null   float64
#  7   total sulfur dioxide  6497 non-null   float64
#  8   density               6497 non-null   float64
#  9   pH                    6497 non-null   float64
#  10  sulphates             6497 non-null   float64
#  11  alcohol               6497 non-null   float64
#  12  quality               6497 non-null   int64  
# dtypes: float64(11), int64(1), object(1)
# memory usage: 660.0+ KB
# None

# calculate descriptive statistics
# describe()
# unique()
# value_counts()
 
wine_df.columns = wine_df.columns.str.replace(' ', '')
print(wine_df.describe())

#        fixedacidity  volatileacidity   citricacid  residualsugar    chlorides  ...      density           pH    sulphates      alcohol      quality
# count   6497.000000      6497.000000  6497.000000    6497.000000  6497.000000  ...  6497.000000  6497.000000  6497.000000  6497.000000  6497.000000
# mean       7.215307         0.339666     0.318633       5.443235     0.056034  ...     0.994697     3.218501     0.531268    10.491801     5.818378
# std        1.296434         0.164636     0.145318       4.757804     0.035034  ...     0.002999     0.160787     0.148806     1.192712     0.873255
# min        3.800000         0.080000     0.000000       0.600000     0.009000  ...     0.987110     2.720000     0.220000     8.000000     3.000000
# 25%        6.400000         0.230000     0.250000       1.800000     0.038000  ...     0.992340     3.110000     0.430000     9.500000     5.000000
# 50%        7.000000         0.290000     0.310000       3.000000     0.047000  ...     0.994890     3.210000     0.510000    10.300000     6.000000
# 75%        7.700000         0.400000     0.390000       8.100000     0.065000  ...     0.996990     3.320000     0.600000    11.300000     6.000000
# max       15.900000         1.580000     1.660000      65.800000     0.611000  ...     1.038980     4.010000     2.000000    14.900000     9.000000

# [8 rows x 12 columns]

print(sorted(wine_df.quality.unique()))

# [3, 4, 5, 6, 7, 8, 9]

print(wine_df.quality.value_counts())

# quality
# 6    2836
# 5    2138
# 7    1079
# 4     216
# 8     193
# 3      30
# 9       5
# Name: count, dtype: int64

print(wine_df.groupby('type')['quality'].describe())

#         count      mean       std  min  25%  50%  75%  max
# type                                                      
# red    1599.0  5.636023  0.807569  3.0  5.0  6.0  6.0  8.0
# white  4898.0  5.877909  0.885639  3.0  5.0  6.0  6.0  9.0

print(wine_df.groupby('type')['quality'].mean())

# type
# red      5.636023
# white    5.877909
# Name: quality, dtype: float64

print(wine_df.groupby('type')['quality'].std())

# type
# red      0.807569
# white    0.885639
# Name: quality, dtype: float64

print(wine_df.groupby('type')['quality'].agg(['mean', 'std']))

#            mean       std
# type                     
# red    5.636023  0.807569
# white  5.877909  0.885639
