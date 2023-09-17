---
layout: single
title: "Data Structure, Linked List"
categories: language
tags: [language, programming, cpp]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---



*Introduction to Linked List*





### Single Linked List: Navigation is forward only

#### 구성

- Node = Data + Link
- Data contains the actual data
- Link contains the address of next node of the list

##### Example 

![2023-09-09_14-45-linedlist]({{site.url}}/images/$(filename)/2023-09-09_14-45-linedlist.png)

- Data: 값, Link: 다음 노드의 주소

- 첫 번째 노드에 도달하기 위해서는 포인터(head)가 있어야 한다. 첫 번째 노드의 주소(1000)를 가지고 있다.

- 메모리에 순차적으로 저장될 필요가 없다

### Doubly Linked List: Forward and backward navigation is possible
### Circular Linked List: Last element is linked to the first element


- [Neso Academy, Introduction to Linked List](https://www.youtube.com/watch?v=R9PTBwOzceo)


#### 장점

- 동적 크기: Linked List는 크기가 동적으로 조절 가능합니다. 요소를 추가하거나 제거할 때마다 메모리를 동적으로 할당하거나 해제할 수 있어 정확한 크기를 미리 지정할 필요가 없습니다.
<br>
- 삽입 및 삭제 효율성: Linked List는 **중간에 요소를 추가하거나 삭제하는 작업이 배열보다 효율적**입니다. 배열은 요소를 삽입하거나 삭제할 때 요소를 이동해야 하지만, Linked List는 삽입 또는 삭제할 노드의 앞과 뒤 노드만 조정하면 됩니다.
<br>
- 메모리 관리: Linked List는 동적 메모리 할당을 사용하므로 메모리 사용량을 최적화할 수 있습니다. 필요한 만큼의 메모리만 할당하고 해제할 수 있습니다.
<br>
- 데이터 구조의 유연성: Linked List는 다양한 형태의 데이터 구조를 구현하는 데 사용될 수 있습니다. 단순 연결 리스트, 이중 연결 리스트, 원형 연결 리스트 등 다양한 변형이 있어 다양한 문제에 적용할 수 있습니다.
<br>
- 스레드 안전성: Linked List는 스레드 안전성을 제공하는 동시성 제어 메커니즘과 결합할 수 있어 다중 스레드 환경에서 사용될 수 있습니다.
<br>
- 메모리 단편화 방지: Linked List는 동적 메모리 할당을 사용하며 요소를 추가하거나 제거할 때 메모리 단편화를 최소화하는 데 도움이 됩니다.
<br>

##### 예시, 스레드 안전성

```cpp
#include <iostream>
#include <thread>
#include <mutex>

// 노드 구조체 정의
struct Node {
    int data;
    Node* next;
    Node(int value) : data(value), next(nullptr) {}
};

class ThreadSafeLinkedList {
public:
    ThreadSafeLinkedList() : head(nullptr) {}

    // 노드를 추가하는 함수 (스레드 안전)
    void addNode(int value) {
        std::lock_guard<std::mutex> lock(mutex); // 뮤텍스 락 획득

        Node* newNode = new Node(value);
        newNode->next = head;
        head = newNode;
    }

    // 리스트를 출력하는 함수 (스레드 안전)
    void printList() {
        std::lock_guard<std::mutex> lock(mutex); // 뮤텍스 락 획득

        Node* current = head;
        while (current != nullptr) {
            std::cout << current->data << " ";
            current = current->next;
        }
        std::cout << std::endl;
    }

    ~ThreadSafeLinkedList() {
        Node* current = head;
        while (current != nullptr) {
            Node* next = current->next;
            delete current;
            current = next;
        }
    }

private:
    Node* head;
    std::mutex mutex; // 뮤텍스를 사용하여 스레드 안전성 보장
};

int main() {
    ThreadSafeLinkedList myList;

    // 스레드 1: 노드 추가
    std::thread thread1([&myList]() {
        for (int i = 1; i <= 5; ++i) {
            myList.addNode(i);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // 스레드 2: 리스트 출력
    std::thread thread2([&myList]() {
        for (int i = 0; i < 3; ++i) {
            myList.printList();
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });

    thread1.join();
    thread2.join();

    return 0;
}
```


##### 예시, 메모리 단편화 최소화

```cpp
#include <iostream>

// 노드 구조체 정의
struct Node {
    int data;
    Node* next;
    Node(int value) : data(value), next(nullptr) {}
};

class MemoryEfficientLinkedList {
public:
    MemoryEfficientLinkedList() : head(nullptr) {}

    // 노드를 추가하는 함수 (메모리 단편화 최소화)
    void addNode(int value) {
        Node* newNode = new Node(value);
        if (head == nullptr) {
            head = newNode;
        } else {
            Node* current = head;
            while (current->next != nullptr) {
                current = current->next;
            }
            current->next = newNode;
        }
    }

    // 리스트를 출력하는 함수
    void printList() {
        Node* current = head;
        while (current != nullptr) {
            std::cout << current->data << " ";
            current = current->next;
        }
        std::cout << std::endl;
    }

    ~MemoryEfficientLinkedList() {
        Node* current = head;
        while (current != nullptr) {
            Node* next = current->next;
            delete current;
            current = next;
        }
    }

private:
    Node* head;
};

int main() {
    MemoryEfficientLinkedList myList;

    // 노드 추가
    for (int i = 1; i <= 5; ++i) {
        myList.addNode(i);
    }

    // 리스트 출력
    myList.printList();

    return 0;
}
```