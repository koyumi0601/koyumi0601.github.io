import numpy as np
import pandas as pd

# data = list(range(1, 11)) # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# np.mean(data)
# np.median(data)
# np.std(data)
# np.max(data)
# np.min(data)
# print(np.percentile([2, 2, 2, 4, 5, 5, 5, 5, 5, 5], [0, 25, 50, 75, 100])) 
# print(np.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 25, 50, 75, 100])) # [ 1.    3.25  5.5   7.75 10.  ]

# df = pd.DataFrame(data)
# print(df.describe())

## T test example
# data
score = [[135, 136, 138, 125, 150], [140, 138, 142, 126, 148]]
score_col = ['n1', 'n2', 'n3', 'n4', 'n5']
score_row = ['before education', 'after education']
score_df = pd.DataFrame(score, index=score_row, columns=score_col)
print(score_df)

#                    n1   n2   n3   n4   n5
# before education  135  136  138  125  150
# after education   140  138  142  126  148

# Set hypothesis, H0: Null hypothesis. there's no difference from education
# Calculate descriptive statistics
diff = np.array(score[1][:]) - np.array(score[0][:]) # [5, 2, 4, 1, -2]
average_diff = np.mean(diff) # 2.0
diff_minus_average_diff = diff - average_diff # [3, 0, 2, -1, -4]
power_of_diff_minus_average_diff = np.power(diff_minus_average_diff, 2) # [9, 0, 4, 1, 16]
sum_of_power_of_diff_minus_average_diff = sum(power_of_diff_minus_average_diff) # 30
sampleVariance = sum_of_power_of_diff_minus_average_diff / (len(diff) - 1) # 30 / (5-1) = 7.5
sampleStd = np.sqrt(sampleVariance) # sqrt(7.5) = 2.7386...
# print(sampleStd)  

desc_stats_data = score
desc_stats_data.append(list(diff))
desc_stats_data.append([average_diff]*5)
desc_stats_data.append(list(diff_minus_average_diff))
desc_stats_data.append(list(power_of_diff_minus_average_diff))
desc_stats_data.append([sum_of_power_of_diff_minus_average_diff]*5)
desc_stats_data.append([sampleVariance] * 5)
desc_stats_data.append([sampleStd] * 5)

desc_stats_row = ['score before education','score after education', 'difference', 'average of difference', 'difference - average difference', 'power of difference - average difference', 'sum of power of difference - average difference', 'sample variance', 'sample std']
desc_stats_df = pd.DataFrame(desc_stats_data, index=desc_stats_row, columns=score_col)


print(desc_stats_df)
# desc_stats_data.append(list(average_diff))
# print(desc_stats_data)