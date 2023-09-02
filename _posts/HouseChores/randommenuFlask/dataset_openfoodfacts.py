# https://openfoodfacts.github.io/openfoodfacts-python/usage/


# API
from openfoodfacts import API, APIVersion, Country, Environment, Flavor

api = API(
    username=None,
    password=None,
    country=Country.world,
    flavor=Flavor.off,
    version=APIVersion.v2,
    environment=Environment.org,
)

results = api.product.text_search("mineral water")
print(results)

# # .csv
# # field larger than field limit
# from openfoodfacts import ProductDataset
# dataset = ProductDataset("csv")

# for product in dataset:
#     print(product["product_name"])