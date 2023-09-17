#include <iostream>
using namespace std;

class ListCell {
public:
    ListCell(char c, ListCell* p = nullptr) : contents(c), Next(p) {}
    char GetContents() { return contents; }
    ListCell* GetNext() { return Next; } // Next 반환 함수 추가
    void SetNext(ListCell* p) { Next = p; }
protected:
    char contents;
    ListCell* Next;
};

class ListIter;

class List {
    friend class ListIter;
public:
    List() : First(nullptr) {}
    void add(char c) {
        ListCell* cell = new ListCell(c, First);
        First = cell;
    }
protected:
    ListCell* First;
};

class ListIter {
public:
    ListIter() : ptr(nullptr) {}
    ListIter(List& l) : ptr(l.First) {}
    void reset(List& l) { ptr = l.First; }
    char operator()() {
        if (ptr != nullptr)
            return ptr->GetContents();
        else
            return '\0';
    }
    void operator++() {
        if (ptr != nullptr) ptr = ptr->GetNext();
    }
    void operator=(char c) {
        // ListCell 클래스에서 contents를 변경할 수 없으므로, 여기에서 변경 불가
        // 적절한 방법으로 수정해야 함
    }
    int operator!() { return ptr != nullptr; }
protected:
    ListCell* ptr;
};

int main() {
    List l;
    l.add('a');
    l.add('b');
    l.add('c');

    ListIter i(l);
    while (!i) {
        cout << i();
        ++i;
    }
    
    return 0;
}