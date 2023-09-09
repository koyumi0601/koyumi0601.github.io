// Pass by Value

// #include <iostream>

// void swap(int a, int b)
// {
//     int temp = a;
//     a = b;
//     b = temp;
// }

// int main()
// {
//     int value = 2;
//     int list[5] = {1, 3, 5, 7, 9};

//     int temp1 = value;
//     int temp2 = list[0];

//     swap(temp1, temp2);

//     value = temp1;
//     list[0] = temp2;

//     std::cout << value << "   " << list[0] << std::endl;

//     temp1 = list[0];
//     temp2 = list[1];

//     swap(list[0], list[1]);

//     list[0] = temp1;
//     list[1] = temp2;

//     std::cout << list[0] << "   " << list[1] << std::endl;

//     temp1 = value;
//     temp2 = list[value];

//     swap(value, list[value]);

//     value = temp1;
//     list[value] = temp2;

//     std::cout << value << "   " << list[value] << std::endl;

//     return 0;
// }


// Pass by Value Result

#include <iostream>

void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

int main()
{
    int value = 2;
    int list[5] = {1, 3, 5, 7, 9};

    int temp1 = value;
    int temp2 = list[0];

    swap(temp1, temp2);

    value = temp1;
    list[0] = temp2;

    std::cout << value << "   " << list[0] << std::endl;

    temp1 = list[0];
    temp2 = list[1];

    swap(list[0], list[1]);

    list[0] = temp1;
    list[1] = temp2;

    std::cout << list[0] << "   " << list[1] << std::endl;

    temp1 = value;
    temp2 = list[value];

    swap(value, list[value]);

    value = temp1;
    list[value] = temp2;

    std::cout << value << "   " << list[value] << std::endl;

    return 0;
}