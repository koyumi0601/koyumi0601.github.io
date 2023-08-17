from datetime import datetime 
import numpy as np
import pandas as pd

def calculate_age(birthdate):
    today = datetime.today()
    birth_year, birth_month, birth_day = map(int, birthdate.split('-'))
    age = today.year - birth_year - ((today.month, today.day) < (birth_month, birth_day))
    return age

def calculate_BasalMetabolicRate(sex, weight, height, age):
    if sex.lower() == 'male':
        BMR = (66.47 + ( 13.75 * weight ) + ( 5 * height) - (6.76 * age))
    elif sex.lower() == 'female':
        BMR = (665.1 + (9.56 * weight ) + ( 1.85 * height ) - (4.68 * age))
    return BMR

def user_info():
    sex = 'Female' # Male, Female
    weight_current = 61.1 # kg
    weight_target = 49.0
    height = 162 # cm
    birthdate = '1987-06-01'
    age = calculate_age(birthdate)
    BMR = calculate_BasalMetabolicRate(sex, weight_current, height, age)
    return sex, weight_current, weight_target, height, birthdate, age, BMR

def plan_diet(weight_current, weight_target):
    weight_loss_limit_per_month = 0.05 # 5%
    diet_period_required_months = int(np.ceil(np.log10(weight_target/weight_current)/np.log10((1-weight_loss_limit_per_month))))
    weight_months = np.zeros(diet_period_required_months+1)
    weight_months[0] = weight_current
    for i in range(diet_period_required_months):
        weight_months[i+1] = weight_months[i]*(1-weight_loss_limit_per_month)
    weight_months = np.round(weight_months, 1)

def calc_calories(weight_months):
    

# def physical_info():
#     # inputs

#     weight_target_loss = weight_current - weight_target # 12.1 kg
#     calorie_per_weight = 7700 # kcal/kg

#     calorie_total_to_target_weight = calorie_per_weight * weight_target_loss # 93170
    
#     weight_months = np.zeros(months_required+1)
#     weight_months[0] = weight_current
#     calorie_loss_months = np.zeros(months_required)
#     calorie_loss_day = np.zeros(months_required)
#     calorie_target_day = np.zeros(months_required)
#     # print(weight_months)
#     for i in range(months_required):
#         weight_months[i+1] = weight_months[i]*(1-weight_loss_limit)
#         calorie_loss_months[i] = 7700 * weight_months[i]*weight_loss_limit
#         calorie_loss_day[i] = calorie_loss_months[i] / 30
#         calorie_target_day[i] = calorie_loss_day[i] - calculate_BasalMetabolicRate(sex, weight_months[i], height, age)



#     print(weight_months)
#     print(calorie_loss_months)
#     print(calorie_loss_day)
#     print(calorie_target_day)
#     print(f"Age: %d " %age)
#     print(f"Basal metabolic rate: %d" %int(np.round(BMR,1)))
#     print(f"diet program period: %d month" %months_required)


def get_food_nutrition():
    # pip install openpyxl
    food_df = pd.read_excel("./KFDA_Nutrition_20230816.xlsx", sheet_name="Sheet0")
    return food_df

def main():

    sex, weight_current, weight_target, height, birthdate, age, BMR = user_info()
    plan_diet(weight_current, weight_target)



if __name__ == "__main__":
    main()