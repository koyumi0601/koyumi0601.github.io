
# tree
# .
# ├── create_project_structure.sh
# ├── project
# │   ├── build
# │   ├── cache
# │   ├── CMakeLists.txt
# │   ├── config
# │   │   └── settings.json
# │   ├── docs
# │   │   ├── api_reference.md
# │   │   └── user_guide.md
# │   ├── include
# │   │   ├── database
# │   │   │   └── database_manager.h
# │   │   ├── imaging_system
# │   │   │   ├── image_filter.h
# │   │   │   ├── image_loader.h
# │   │   │   └── image_transform.h
# │   │   ├── networking
# │   │   │   └── network_manager.h
# │   │   └── userinterface
# │   │       └── window.h
# │   ├── LICENSE
# │   ├── logs
# │   ├── README.md
# │   ├── requirements.txt
# │   ├── resources
# │   │   ├── data
# │   │   │   └── dataset.csv
# │   │   └── images
# │   │       ├── logo.png
# │   │       └── sampleimage.png
# │   ├── scripts
# │   │   └── python_script1.py
# │   ├── src
# │   │   ├── database
# │   │   │   └── database_manager.cpp
# │   │   ├── imaging_system
# │   │   │   ├── image_filter.cpp
# │   │   │   ├── image_loader.cpp
# │   │   │   └── image_transform.cpp
# │   │   ├── main.cpp
# │   │   ├── networking
# │   │   │   └── network_manager.cpp
# │   │   ├── python_modules
# │   │   │   ├── module1
# │   │   │   │   ├── __init__.py
# │   │   │   │   └── module1_file1.py
# │   │   │   └── module2
# │   │   │       ├── __init__.py
# │   │   │       └── module2_file1.py
# │   │   └── userinterface
# │   │       └── window.cpp
# │   ├── temp
# │   ├── tests
# │   ├── vendor
# │   └── venv
# └── tree.txt

# 27 directories, 30 files


#!/bin/bash

# 프로젝트 루트 디렉토리 생성
mkdir -p project

# 각 디렉토리 및 파일 생성
cd project
touch .gitignore CMakeLists.txt LICENSE README.md requirements.txt
mkdir -p build cache config docs include logs resources scripts src temp tests vendor venv
touch config/settings.json
touch docs/api_reference.md docs/user_guide.md
mkdir -p include/database include/imaging_system include/networking include/userinterface 
touch include/database/database_manager.h
touch include/imaging_system/image_filter.h include/imaging_system/image_loader.h include/imaging_system/image_transform.h
touch include/networking/network_manager.h
touch include/userinterface/window.h
mkdir resources/data resources/images
touch resources/data/dataset.csv
touch resources/images/logo.png resources/images/sampleimage.png
touch scripts/python_script1.py
mkdir src/database src/imaging_system src/networking src/python_modules src/userinterface
touch src/database/database_manager.cpp
touch src/imaging_system/image_filter.cpp src/imaging_system/image_loader.cpp src/imaging_system/image_transform.cpp
touch src/networking/network_manager.cpp
mkdir src/python_modules/module1 src/python_modules/module2
touch src/python_modules/module1/__init__.py src/python_modules/module1/module1_file1.py
touch src/python_modules/module2/__init__.py src/python_modules/module2/module2_file1.py
touch src/userinterface/window.cpp
touch src/main.cpp

echo "프로젝트 디렉토리 구조 생성이 완료되었습니다."
