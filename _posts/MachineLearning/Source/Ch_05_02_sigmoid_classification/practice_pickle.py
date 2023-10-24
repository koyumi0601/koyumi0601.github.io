import pickle

# Serialize: to binary
# data = {"name": "John", "age": 30, "city": "New York"}
# with open('data.pkl', 'wb') as file:
#     pickle.dump(data, file)


# Deserialize: from binary
with open('data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
print(loaded_data)  # 출력: {'name': 'John', 'age': 30, 'city': 'New York'}