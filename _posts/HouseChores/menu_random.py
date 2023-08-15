from flask import Flask, render_template
import requests
from bs4 import BeautifulSoup
import random
import json

app = Flask(__name__)

def random_menu():

    menu=[
    "불고기",
    "갈비찜",
    "제육볶음",
    "감자조림",
    "오징어볶음",
    "닭갈비",
    "불닭볶음면",
    "돼지갈비",
    "갈비구이",
    "닭강정",
    "오징어채볶음",
    "돼지불고기",
    "돼지갈비찜",
    "두루치기",
    "삼겹살볶음",
    "찜닭",
    "닭날개",
    "떡갈비",
    "순대볶음",
    "소불고기",
    "돼지두루치기",
    "소갈비",
    "소불닭",
    "감자볶음",
    "삼겹살김치볶음",
    "소고기볶음",
    "오리불고기",
    "갈치조림",
    "양념갈비",
    "돼지불똥집",
    "삼겹살찜",
    "새우볶음",
    "소세지볶음",
    "목살볶음",
    "닭볶음탕",
    "소불고기찜",
    "소고기구이",
    "닭토마토볶음",
    "감자갈비찜",
    "미니버거",
    "소불닭찜",
    "연어구이",
    "돼지불갈비",
    "낙지볶음",
    "새우튀김",
    "소고기볶음밥",
    "날치알볶음",
    "새우볶음밥",
    "소갈비찜",
    "갈치구이",
    "버터치킨",
    "소고기불고기",
    "소고기야끼니쿠",
    "돼지갈비볶음",
    "소고기갈비",
    "소고기볶음우동",
    "치킨스테이크",
    "소고기야끼우동",
    "닭갈비볶음밥",
    "소고기카레",
    "새우볶음우동",
    "소고기찜",
    "닭갈비찜",
    "소고기크림파스타",
    "매운탕",
    "소고기볶음우동",
    "새우야끼우동",
    "소고기마파두부",
    "소고기피자",
    "새우카레",
    "소고기두부김치",
    "양갈비찜",
    "소고기라멘",
    "소고기감바스",
    "소고기덮밥",
    "새우볶음덮밥",
    "소고기짜장면",
    "소고기불고기덮밥",
    "닭갈비덮밥",
    "소고기커리",
    "소고기짬뽕",
    "소고기오므라이스",
    "매운탕",
    "소고기우동",
    "닭갈비우동",
    "소고기김치볶음밥",
    "갈치조림덮밥",
    "소고기비빔밥",
    "닭고기야끼소바",
    "소고기두루치기볶음밥",
    "새우볶음소바",
    "김치전",
    "부대찌개",
    "김치찌개",
    "떡볶이",
    "쫄면",
    "잔치국수",
    "물냉면",
    "비빔냉면",
    "메밀국수",
    "칼국수",
    "막국수",
    "우렁이해장국",
    "고추장찌개",
    "해물찜",
    "주물럭",
    "장조림",
    "계란찜",
    "아귀찜",
    "닭볶음",
    "뼈해장국",
    "삼계탕",
    "낙지볶음",
    "콩나물불고기",
    "깐풍기",
    "매운소갈비찜",
    "냉이된장찌개",
    "김치말이국수",
    "감자탕",
    "수육",
    "닭고기새송이스테이크",
    "육개장",
    "닭도리탕",
    "양배추롤",
    "양념치킨",
    "불고기샐러드",
    "치킨샐러드",
    "제육덮밥",
    "감자조림",
    "떡국",
    "닭강정",
    "감자볶음",
    "마파두부",
    "누룽지탕",
    "닭도리탕",
    "코다리찜",
    "탕수육",
    "꿔바로우",
    "볶음밥",
    "연어덮밥",
    "야끼우동",
    "꼬치구이",
    "김치볶음",
    "닭갈비볶음",
    "라조기",
    "매콤소갈비",
    "닭찜",
    "닭고기커틀릿",
    "소고기덮밥",
    "새우볶음",
    "소갈비",
    "고추장불고기",
    "돼지불백",
    "감자고로케",
    "닭강정",
    "소고기덮밥",
    "돼지김치볶음",
    "오징어볶음",
    "계란말이",
    "닭갈비",
    "미트볼",
    "돈까스",
    "소고기야채볶음",
    "소고기덮밥",
    "소고기떡볶이",
    "돼지갈비찜",
    "고기전",
    "쭈꾸미볶음",
    "소고기튀김",
    "돼지고기튀김",
    "닭고기튀김",
    "마파두부",
    "된장찌개",
    "돼지고기채소볶음",
    "찜닭",
    "돼지고기김치찌개",
    "감자조림",
    "김치볶음밥",
    "돼지고기고추전",
    "감자전",
    "부추전",
    "무생채",
    "참나물무침",
    "피자",
    "치킨마요",
    "구운생선",
    "바나나우유쉐이크",
    "달걀오믈렛",
    "채소스틱",
    "호박전",
    "알리오올리오스파게티"
    ]
    menu = sorted(list(set(menu)))
    random_menu = random.choice(menu)
    return random_menu


def food_info(name):
    '''
    This function gives you food information for the given input.

    PARAMETERS
        - name(str): name of Korean food in Korean ex) food_info("김치찌개")
    RETURN
        - res(list): list of dict that containing info for some Korean food related to 'name'
            - res['name'](str): name of food
            - res['ingredients'](str): ingredients to make the food
            - res['recipe'](list[str]): contain recipe in order
    '''
    url = f"https://www.10000recipe.com/recipe/list.html?q={name}"
    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
    else : 
        print("HTTP response error :", response.status_code)
        return
    
    food_list = soup.find_all(attrs={'class':'common_sp_link'})
    food_id = food_list[0]['href'].split('/')[-1]
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
    ingredient = ','.join(result['recipeIngredient'])
    recipe = [result['recipeInstructions'][i]['text'] for i in range(len(result['recipeInstructions']))]
    for i in range(len(recipe)):
        recipe[i] = f'{i+1}. ' + recipe[i]
    
    res = {
        'name': name,
        'ingredients': ingredient,
        'recipe': recipe
    }

    return res

@app.route('/')
def index():
    today_menu = random_menu()
    today_recipe = food_info(today_menu)
    return render_template('index.html', menu=today_menu, recipe=today_recipe)
   

if __name__ == "__main__":
    # main()
    app.run(host='0.0.0.0')