from flask import Flask, render_template
import requests
from bs4 import BeautifulSoup
import random
import json
from datetime import datetime 
import numpy as np
import pandas as pd
import get_seasonalfood_naver
import get_10000recipe_ranking


# Food

def get_KFDA_menu():
    KFDA_menu = pd.read_excel('./KFDA_Nutrition_20230816.xlsx', engine="openpyxl")
    KFDA_menu_name_category = KFDA_menu[['식품명', '식품대분류']]
    filtered_row=[]
    filtered_row.append(KFDA_menu_name_category[(KFDA_menu_name_category['식품대분류']=='구이류') 
                                                | (KFDA_menu_name_category['식품대분류']=='볶음류')
                                                | (KFDA_menu_name_category['식품대분류']=='찌개 및 전골류')
                                                | (KFDA_menu_name_category['식품대분류']=='조림류')
                                                | (KFDA_menu_name_category['식품대분류']=='찌개류')
                                                ])
    filtered_df = pd.DataFrame(filtered_row[0])
    random_menu = filtered_df.sample(n=1)
    random_menu_name = random_menu['식품명'].iloc[0]
    print(random_menu_name)
    return random_menu_name

def get_random_seasonal_ingredient():
    sfood, random_fruit, random_seafood, random_vegetable = get_seasonalfood_naver.get_seasonalfood()
    print(random_fruit)
    print(random_seafood)
    print(random_vegetable)
    return sfood, random_fruit, random_seafood, random_vegetable

app = Flask(__name__)

# Diet

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

def physical_info():
    # inputs
    sex = 'Female' # Male, Female
    weight_current = 61.1 # kg
    weight_target = 49.0
    height = 162 # cm
    birthdate = '1987-06-01'
    age = calculate_age(birthdate)
    BMR = calculate_BasalMetabolicRate(sex, weight_current, height, age)
    weight_target_loss = weight_current - weight_target # 12.1 kg
    calorie_per_weight = 7700 # kcal/kg
    weight_loss_limit = 0.05 # 5%
    calorie_total_to_target_weight = calorie_per_weight * weight_target_loss # 93170
    
    months_required = int(np.ceil(np.log10(weight_target/weight_current)/np.log10((1-weight_loss_limit))))
    weight_months = np.zeros(months_required+1)
    weight_months[0] = weight_current
    calorie_loss_months = np.zeros(months_required)
    calorie_loss_day = np.zeros(months_required)
    calorie_target_day = np.zeros(months_required)
    # print(weight_months)
    for i in range(months_required):
        weight_months[i+1] = weight_months[i]*(1-weight_loss_limit)
        calorie_loss_months[i] = 7700 * weight_months[i]*weight_loss_limit
        calorie_loss_day[i] = calorie_loss_months[i] / 30
        calorie_target_day[i] = calorie_loss_day[i] - calculate_BasalMetabolicRate(sex, weight_months[i], height, age)



    print(weight_months)
    print(calorie_loss_months)
    print(calorie_loss_day)
    print(calorie_target_day)
    print(f"Age: %d " %age)
    print(f"Basal metabolic rate: %d" %int(np.round(BMR,1)))
    print(f"diet program period: %d month" %months_required)


def get_food_nutrition():
    # pip install openpyxl
    food_df = pd.read_excel("./KFDA_Nutrition_20230816.xlsx", sheet_name="Sheet0")
    return food_df

def food_info(sfood):
    url = f"https://www.10000recipe.com/recipe/list.html?q={sfood}"
    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
    else : 
        print("HTTP response error :", response.status_code)
        return
    random_recipe_index = random.randrange(40) 
    food_list = soup.find_all(attrs={'class':'common_sp_link'})
    food_id = food_list[random_recipe_index]['href'].split('/')[-1]
    new_url = f'https://www.10000recipe.com/recipe/{food_id}'
    new_response = requests.get(new_url)
    if new_response.status_code == 200:
        html = new_response.text
        soup = BeautifulSoup(html, 'html.parser')
    else : 
        print("HTTP response error :", response.status_code)
        return
    food_info = soup.find(attrs={'type':'application/ld+json'})
    result = json.loads(food_info.text)
    menu = ''.join(result['name'])
    ingredient = ','.join(result['recipeIngredient'])
    recipe = [result['recipeInstructions'][i]['text'] for i in range(len(result['recipeInstructions']))]
    for i in range(len(recipe)):
        recipe[i] = f'{i+1}. ' + recipe[i]
    res = {
        'seasonal food': sfood, 
        'name': menu,
        'ingredients': ingredient,
        'recipe': recipe
    }
    return res

@app.route('/')
def index():
    
    # today_menu = random_seafood 
    seafood_recipe = food_info(random_seafood)
    vegetable_recipe = food_info(random_vegetable)
    fruit_recipe = food_info(random_fruit)
    diet_recipe = food_info(diet)
    kfda_menu_recipe = food_info(KFDA_menu)
    stew_recipe = food_info(stew)
    beef_recipe = food_info(beef)
    pork_recipe = food_info(pork)
    chiken_recipe = food_info(chiken)
    seafood_routine_recipe = food_info(seafood_routine)
    egg_recipe = food_info(egg)
    tofu_recipe = food_info(tofu)
    side_recipe = food_info(side)

    return render_template('index.html', 
                           stew_recipe=stew_recipe, 
                           beef_recipe=beef_recipe, 
                           pork_recipe=pork_recipe,
                           chiken_recipe=chiken_recipe,
                           seafood_routine_recipe=seafood_routine_recipe,
                           egg_recipe=egg_recipe,
                           tofu_recipe=tofu_recipe,
                           side_recipe=side_recipe,
                           seafood_recipe=seafood_recipe, 
                           vegetable_recipe=vegetable_recipe, 
                           fruit_recipe=fruit_recipe,
                           diet_recipe=diet_recipe,
                           kfda_menu_recipe=kfda_menu_recipe
                           )
   

if __name__ == "__main__":
    physical_info()
    KFDA_menu = get_KFDA_menu()
    sfood, random_fruit, random_seafood, random_vegetable = get_random_seasonal_ingredient()
    stew = get_10000recipe_ranking.get_food_ranking("찌개")
    beef = get_10000recipe_ranking.get_food_ranking("메인:소")
    pork = get_10000recipe_ranking.get_food_ranking("메인:돼지")
    chiken = get_10000recipe_ranking.get_food_ranking("메인:닭")
    seafood_routine = get_10000recipe_ranking.get_food_ranking("메인:해물")
    egg = get_10000recipe_ranking.get_food_ranking("메인:달걀")
    tofu = get_10000recipe_ranking.get_food_ranking("메인:콩")
    side = get_10000recipe_ranking.get_food_ranking("밑반찬")
    diet = get_10000recipe_ranking.get_food_ranking("다이어트")
    app.run(host='0.0.0.0')