# install

```bash
#sudo apt-get install sqlite3 # 설치, 터미널에서 동작... #include <sqlite3> 빌드하니 되지 않았음. 
sudo apt-get install libsqlite3-dev # #include 잘 됨.
sqlite3 --version # 설치 확인
```

# getting started

```bash
sqlite3 # sqlite 진입
sqlite3> .eixt # sqlite 콘솔 종료
sqlite3 my_database.db # 현재 디렉토리에 db 생성. 테이블 값이 안들어가면 파일생성이 완료되지 않는다
# 사용자 테이블 생성
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT NOT NULL,
    password TEXT NOT NULL
);
INSERT INTO users (username, email, password) VALUES ('john_doe', 'john@example.com', 'password123');
INSERT INTO users (username, email, password) VALUES ('jane_smith', 'jane@example.com', 'pass123');
# 테이블 열람
SELECT * FROM users;
```

# extension
- Sqlite viewer (별점 5점)
- Sqlite3 editor (별점 5점)
- *.db를 클릭하면 vscode 위에 선택 가능하게 나옴.

# .cpp에서 사용하려면, CMakeLists 변경

```cmake
find_library(SQLITE3_LIBRARY sqlite3)
target_link_libraries(MyProject PRIVATE ${OpenCV_LIBS} tesseract stdc++fs pthread ${SQLITE3_LIBRARY})
```


# Info
- 공식 페이지 [https://www.sqlite.org/index.html](https://www.sqlite.org/index.html)
- linux, window 둘 다 호환