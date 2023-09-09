
#include <iostream>
using namespace std; 

// class ListCell; // 원문
class List; // forward declaration

class ListCell {
// ListCell: 
// - Linked list의 노드. 
// - 데이터 contents, 다음 노드 포인터 Next. 
// - 생성자, 복사 생성자

    friend List; // List의 private 접근 허용
    public:
        // 클래스의 생성자
        ListCell(char c, ListCell *p = NULL) : contents(c), Next(p) {}; 
        // 매개변수 char c, ListCell *p 포인터, 초기값 NULL
        // 초기화 목록 contents(c), Next(p) - 멤버변수 contents, Next를 c, p로 초기화한다.

        // 클래스의 기본 생성자 (default contructor)
        ListCell() : contents('\0'), Next(NULL) {};
        // 매개 변수 없음
        // 초기화 목록 '\0' 문자열 끝을 나타내는 Null 문자, 빈 문자열로 초기화하라는 뜻.
        // NULL 포인터를 NULL로 초기화하라는 뜻 

        // 클래스의 복사 생성자 (copy constructor)
        ListCell(const ListCell& lc)
        {
            contents = lc.contents;
            Next = lc.Next;
        }
        // lc: 복사하려는 다른 ListCell 객체 참조
        // 현재의 contents, Next가 lc.contents, lc.Next를 참조해서 복사를 수행한다.

    // 클래스 내의 할당연산자(=) 정의
    ListCell& operator= (const ListCell& lc)
    // 인자로 다른 ListCell을 받음. 읽기 전용.
    { 
        contents = lc.contents; // 실제 할당 연산
        Next = lc.Next;
        return *this;
        // 현재 객체인 ListCell 객체의 참조를 반환. > 이렇게하면, 아래와 같이 할당 연산자를 연속적으로 사용할 수 있다. 연쇄 할당
        // ListCell cell1; ListCell cell2; ListCell; cell1 = cell2 = cell3; 
    }

        // 클래스의 멤버 함수들
        char GetContents() { return contents; } // contents 반환
        ListCell* GetNext() { return Next; } // Next 반환
        
        // 클래스의 소멸자
        ~ListCell() {};

    // 클래스의 멤버 변수들
    protected:
        char contents;
        ListCell* Next;
};

class List {
// List:
// Linked List의 제어담당

    public:
        // 클래스의 생성자
        List() { First = NULL; }
        // 매개 변수 없음.
        // 멤버 변수 First를 Null로 초기화
        // 초기화 목록(: )은 사용하지 않음
        
        // 클래스의 복사생성자의 prototype. 실제 코드는 클래스 외부에 기술되어 있다.
        List (const List&);
        // 매개 변수 List&, 읽기 전용

        // 클래스의 할당연산자(=) 정의
        List& operator= (List& l);
        // 객체의 멤버 변수 등을 복사하여 값을 복사
        // 매개 변수 List& l, 우변의 List 객체를 받아들임
        // List 객체를 할당 연산의 대상인 *this 객체에 복사하는 역할
        // *this 객체의 참조를 반환하여 연속적인 할당 연산을 가능하게 함. 연쇄할당

        // 멤버 함수 Add
        // 목적: 새로운 문자 char c를 리스트에 추가 함.
        // 주요역할: First 멤버 변수를 업데이트하여 새로운 요소를 리스트의 맨 앞에 추가.
        void add(char c)
        {
            First = new ListCell(c, First);
        }

        // 클래스의 멤버 함수 remove의 prototype. 실제 코드는 클래스 외부에 기술되어 있다. 
        void remove(char);
        // 매개변수 char. 어떤 매개변수를 쓰는 지 이름도 같이 명시하는 것이 더 좋다.
        // 목적: 리스트에서 해당 문자열을 제거
        // 리스트의 첫 번째 노드부터 시작해서 주어진 문자열과 동일한 내용을 가진 노드를 찾고 해당 노드를 제거하는 작업을 수행

        // 클래스의 멤버 함수 IsIn의 prototype. 실제 코드는 클래스 외부에 기술되어 있다.
        int IsIn(char);
        // 매개변수 char. 어떤 매개변수를 쓰는 지 이름도 같이 명시하는 것이 더 좋다.
        // 목적: 리스트 안에 주어진 문자열이 존재하는 지 확인
        // 리스트의 첫 번째 노드부터 시작해서 주어진 문자열과 동일한 내용을 가진 노드를 찾고, 찾으면 1 아니면 0을 반환

        // 클래스의 소멸자
        ~List();

    // 클래스의 멤버 변수
    protected:
        ListCell* First;
};

// 클래스의 복사생성자를 외부에서 정의
List::List(const List& l)
{
    if (l.First == NULL)
        First = NULL;
    else
    {
        First = new ListCell(l.First->contents, NULL);
        ListCell *p1 = l.First->Next;
        ListCell *p2 = First;
        while (p1 != NULL)
        {
            p2->Next = new ListCell(p1->contents, NULL); 
            p2 = p2->Next;
            p1 = p1->Next;
        }
    }
}

// 클래스의 할당연산자(=)를 외부에서 정의
List& List::operator = (List& l)
{
    if (l.First == NULL) 
    { 
        return *this;
    }
    First = new ListCell(l.First->contents, NULL);
    ListCell *p1 = l.First->Next;
    ListCell *p2 = First;
    while (p1 != NULL)
    {
        p2->Next = new ListCell(p1->contents, NULL);
        p2 = p2->Next;
        p1 = p1->Next;
    }
    return *this;
}

// 클래스의 소멸자를 클래스 외부에서 정의함.
List::~List()
{
    ListCell *p = First, *next;
    while (p != NULL)
    {
        next = p->Next;
        delete p;
        p = next;
    }
}

// 클래스의 멤버 함수 InIn을 클래스 외부에서 정의함
int List::IsIn(char c)
{
    ListCell* p = First;
    while (p != NULL)
    {
        if (p->contents == c) break;
        p = p->Next;
    }
    return p != NULL;
}

// 클래스의 멤버 함수 remove을 클래스 외부에서 정의함
void List::remove(char c)
{
    ListCell *p1 = First, *p2 = NULL;
    while (p1->contents !=c)
    {
        p2 = p1;
        p1 = p1->Next;
    }
    p2->Next = p1->Next;
    delete p1;
}


// Main
int main()
{
    List l1, l2;
    l1.add('a');
    l1.add('b');
    l1.add('c');
    l2 = l1;
    l1.remove('b');
    cout << l1.IsIn('b') << l2.IsIn('c') << endl;
    return 0;
}