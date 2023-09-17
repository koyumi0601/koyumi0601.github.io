#include <iostream>
using namespace std;

// void read_array(int *a, int& size) // 읽기 쉬운 표현
void read_array(int a[], int& size) // 배열 전달임이 좀 더 명확
{
    cout << "How many data points?";
    cin >> size;
    cout << "Enter data: \n ";
    for (int i = 0; i < size; ++i)
        cin >> a[i];
}

int sum(const int *a, const int size) // 읽기 전용
{
    int i, temp = 0;
    for (int i = 0; i < size; ++i)
        temp = temp + a[i];
    return temp;
}

int main()
{
    int data[100], n; // data는 크기 100의 array, integer. n은 integer
    read_array(data, n); // data and n now have values
    cout << "Sum is: " << sum(data, n);
    return 0;
}