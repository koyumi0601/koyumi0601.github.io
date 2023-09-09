// 생성자
// 일반 생성자 (Constructor): 이 생성자는 객체를 처음 만들 때 호출됩니다. 주로 초기화를 위해 사용됩니다. 일반 생성자를 사용하여 객체를 초기화할 때 전달되는 값은 새로운 객체의 속성을 설정하는 데 사용됩니다.
// 복사 생성자 (Copy Constructor): 복사 생성자는 이미 존재하는 객체를 기반으로 새로운 객체를 만들 때 호출됩니다. 일반적으로 다른 객체로부터 현재 객체를 복사하는 데 사용됩니다. 복사 생성자는 "깊은 복사"를 수행하여 새로운 객체가 이전 객체와 동일한 내용을 가지도록 합니다.


// 생성자의 종류



// 1. 기본 생성자 (Default Constructor): 
// 매개변수를 갖지 않는 생성자로, 클래스 인스턴스를 만들 때 아무런 인수를 전달하지 않고 생성
class MyClass {
public:
    MyClass() {
        // 생성자 코드
    }
};


// 2. 매개변수가 있는 생성자: 
// 클래스 인스턴스를 생성할 때 초기값을 설정하기 위해 매개변수를 사용하는 생성자
class MyClass {
public:
    MyClass(int value) {
        // 생성자 코드
    }
};


// 3. 생성자 이니셜라이저 목록 (Constructor Initialization List): 
// 멤버 변수를 초기화하기 위해 생성자 이니셜라이저 목록을 사용할 수 있음
class MyClass {
public:
    MyClass(int a, int b) : x(a), y(b) {
        // 생성자 코드
    }
private:
    int x, y;
};

// 4. 복사 생성자 (Copy Constructor): 
// 다른 인스턴스에서 생성자를 호출하여 객체를 복사(깊은 복사)하는 데 사용
// 메모리 주소는 다르고 값이 같음.
class MyClass {
public:
    MyClass(const MyClass& other) {
        // 복사 생성자 코드
    }
};


