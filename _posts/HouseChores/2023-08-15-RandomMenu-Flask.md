---
layout: single
title: "Cooking"
categories: housechores
tags: [menu, housechores, cooking, python, flask]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

*random menu generation using Flask, crawling recipe of random menu from 만개의 레시피*

- install Flask

```bash
pip install Flask
```
- app, html, css 생성

- Flask file tree

```
my_flask_app/
    ├── app.py
    ├── templates/
    │   ├── index.html
    ├── static/
    │   ├── style.css
```

- app.py(menu_random.py)

```python
from flask import Flask, render_template
import requests
from bs4 import BeautifulSoup
import random
import json

app = Flask(__name__)

# pick random menu
# expaned to (1) seasonal food from naver search (2) 10000 recipe ranking (3) KFDA menu
def random_menu():
    menu=[
    "불고기",
    "갈비찜" 
    # Other menus
    ]
    menu = sorted(list(set(menu)))
    random_menu = random.choice(menu)
    return random_menu

# crawl recipe from 10000 recipe
def food_info(name):
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

# app
@app.route('/')
def index():
    today_menu = random_menu()
    today_recipe = food_info(today_menu)
    return render_template('index.html', menu=today_menu, recipe=today_recipe)
   
# main
if __name__ == "__main__":
    app.run(host='0.0.0.0')
```

- index.html

```html
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
    <title>Random Menu</title>
</head>
<body>
    <h1>Today's Random Menu</h1>
    <p><strong>Menu: </strong>{{ menu }}</p>
    <h2>Recipe</h2>
    <p><strong>Ingredients: </strong>{{ recipe['ingredients'] }}</p>
    <h3>Instructions:</h3>
    <ol>
        {% for step in recipe['recipe'] %}
            <ul>{{ step }}</ul>
        {% endfor %}
    </ol>
</body>
</html>
```

![2023-08-15_19-13-html]({{site.url}}/images/$(filename)/2023-08-15_19-13-html.png)

- style.css



```css
body {
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
    margin: 0;
    padding: 0;
    margin-left: 20px;   
    margin-right: 20px;  
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #fff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

h1, h2, h3, p, ul {
    font-family: sans-serif;
    color: #332F2E;
}

```

- webpage for flask: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

![2023-08-15_19-08-randommenu]({{site.url}}/images/$(filename)/2023-08-15_19-08-randommenu.png)

- pc
    - terminal을 열고, 어떤 ip를 사용 중인지 확인한다. 
    ```bash
    ip a
    ```
    - 사용 중인 ip로의 접근을 enable한다
    ```bash
    sudo ufw allow from [some ip] to any port [some port]
    ```
    
- iphone > safari > [ipaddress:port]

