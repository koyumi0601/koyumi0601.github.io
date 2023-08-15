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

# expect quality of new sample using regression model

# visualize result