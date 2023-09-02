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




# visualize result