
#include <iostream>
using namespace std;

struct ListCell; // forward declaration
struct ListCell {
    char contents;
    ListCell *next;
};

class LinkedList {
public:
    LinkedList() : head(nullptr) {}

    // 리스트에 새로운 요소 추가
    void add(char value) {
        ListCell* newCell = new ListCell;
        newCell->contents = value;
        newCell->next = nullptr;

        if (!head) {
            head = newCell;
        } else {
            ListCell* current = head;
            while (current->next) {
                current = current->next;
            }
            current->next = newCell;
        }
    }

    // 리스트 내용 출력
    void print() {
        ListCell* current = head;
        while (current) {
            std::cout << current->contents << " ";
            current = current->next;
        }
        std::cout << std::endl;
    }

    // 리스트 해제
    void clear() {
        while (head) {
            ListCell* temp = head;
            head = head->next;
            delete temp;
        }
    }

private:
    ListCell* head;
};

int main() {
    LinkedList myList;
    myList.add('A');
    myList.add('B');
    myList.add('C');
    myList.print(); // 출력: A B C
    myList.clear(); // 메모리 해제
    return 0;
}