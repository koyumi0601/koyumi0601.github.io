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
# print(regression_result.summary())

# expect quality of new sample using regression model
sample1 = wine_df[wine_df.columns.difference(['quality', 'type'])]
sample1 = sample1[0:5][:]
sample1_predict = regression_result.predict(sample1)
print(sample1_predict)

# 0    4.997607
# 1    4.924993
# 2    5.034663
# 3    5.680333
# 4    4.997607

print(wine_df[0:5]['quality'])

# 0    5
# 1    5
# 2    5
# 3    6
# 4    5


## sample 2 to predict quality
data = {"fixed_acidity": [8.5, 8.1], 
        "volatile_acidity":[0.8, 0.5], 
        "citric_acid": [0.3, 0.4], 
        "residual_sugar": [6.1, 5.8], 
        "chlorides": [0.055, 0.044], 
        "free_sulfur_dioxide": [30.0, 31.0], 
        "total_sulfur_dioxide": [98.0, 99], 
        "density": [0.996, 0.91], 
        "pH": [3.25, 3.01], 
        "sulphates": [0.4, 0.35], 
        "alcohol": [9.0, 0.88]}

sample2 = pd.DataFrame(data, columns=sample1.columns)
print(sample2)

sample2_predict = regression_result.predict(sample2)
print(sample2_predict)

# visualize result
# pip install seaborn

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
sns.distplot(red_wine_quality, kde=True, color="red", label = 'red wine')