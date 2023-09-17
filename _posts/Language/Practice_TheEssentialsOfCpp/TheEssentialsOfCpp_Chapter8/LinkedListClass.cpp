#include <iostream>
using namespace std; // <iostream> 헤더를 포함하고 using 지시문 추가

class List; // List 클래스를 미리 선언

class ListCell {
    friend class List; // List 클래스를 friend로 지정
public:
    ListCell(char c, ListCell *p = NULL) : contents(c), Next(p) {}
    ListCell() : contents('\0'), Next(NULL) {}
    ListCell(const ListCell& lc) {
        contents = lc.contents;
        Next = lc.Next;
    }

    ListCell& operator = (const ListCell& lc) {
        contents = lc.contents;
        Next = lc.Next;
        return *this;
    }

    char GetContents() { return contents; } // 오타 수정
    ListCell* GetNext() { return Next; }

    ~ListCell() {}
protected:
    char contents;
    ListCell* Next;
};

class List {
public:
    List() { First = NULL; }
    List(const List&);
    List& operator = (const List&);

    void add(char c) {
        First = new ListCell(c, First);
    }

    void remove(char);
    int IsIn(char);

    ~List();
protected:
    ListCell* First;
};

List::List(const List& l) {
    if (l.First == NULL)
        First = NULL;
    else {
        First = new ListCell(l.First->contents, NULL);
        ListCell *p1 = l.First->Next;
        ListCell *p2 = First;
        while (p1 != NULL) {
            p2->Next = new ListCell(p1->contents, NULL);
            p2 = p2->Next;
            p1 = p1->Next;
        }
    }
}

List& List::operator = (const List& l) {
    if (l.First == NULL)
        return *this;

    First = new ListCell(l.First->contents, NULL);
    ListCell *p1 = l.First->Next;
    ListCell *p2 = First;
    while (p1 != NULL) {
        p2->Next = new ListCell(p1->contents, NULL);
        p2 = p2->Next;
        p1 = p1->Next;
    }
    return *this;
}

List::~List() {
    ListCell *p = First, *next;
    while (p != NULL) {
        next = p->Next;
        delete p;
        p = next;
    }
}

int List::IsIn(char c) {
    ListCell* p = First;
    while (p != NULL) {
        if (p->contents == c) break;
        p = p->Next;
    }
    return p != NULL;
}

void List::remove(char c) {
    ListCell *p1 = First, *p2 = NULL;
    while (p1 != NULL) {
        if (p1->contents == c) {
            if (p2 == NULL) {
                First = p1->Next;
            } else {
                p2->Next = p1->Next;
            }
            delete p1;
            break;
        }
        p2 = p1;
        p1 = p1->Next;
    }
}

int main() {
    List l1, l2; // List 객체 생성 및 초기화
    l1.add('a'); // 노드 추가, add 메서드 사용
    l1.add('b');
    l1.add('c');
    l2 = l1;
    l1.remove('b');
    cout << l1.IsIn('b') << l2.IsIn('c') << endl;
    return 0;
}