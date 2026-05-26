#include <iostream>

int main(){

    int arr[3][5] = {
        {1,2,3,4,5},
        {6,7,8,9,10},
        {11,12,13,14,15}
    };

    // ip存放的是第一个数组的地址，即 arr[0] 的地址，而*ip存放的是第一个数组arr[0]的地址对应的内容，
    // 即 arr[0][0] 的地址，**ip存放的是 arr[0][0] 的值
    int (*ip)[5] = arr; // ip 是一个指向包含 5 个整数的数组的指针
    std::cout << "add: " << ip << " " << ip + 1 << std::endl; //ip 存放了 arr中第一个数组的地址，ip + 1 存放了 arr[1] 的地址
    std::cout << "add: " << *ip << " " << *ip + 1 << std::endl; // *ip 存放了 arr[0] 的地址，*ip + 1 存放了 arr[0][1] 的地址
    std::cout << "add: " << **ip << std::endl; // **ip 存放了 arr[0][0] 的值
    // add: 0x7fff7b2040b0 0x7fff7b2040c4 步长是 4 * 5 = 20 字节
    // add: 0x7fff7b2040b0 0x7fff7b2040b4 步长是 4 字节
    // add: 1

    for (int i=0; i < 3; i++){
        for (int j=0;j < 5; j++){
            std::cout << *(*ip + j) << " ";
        }
        std::cout << std::endl;
        ip++;
    }

    std::cout << "int* 长度:" << sizeof(int*) << std::endl; // 8

    int arr1[5] = {1,2,3,4,5};
    int arr2[5] = {6,7,8,9,10};
    int a = 10;
    int arr3[5] = {111,112,113,114,115};

    int* arr_2d[3] = {arr1, arr2, arr3}; 
    int** ip2 = arr_2d;// 二级指针
    std::cout << "一维数组地址：" << arr1 << " " << arr2 << " " << arr3 << std::endl; // 一维数组地址：0x7fff7b2040b0 0x7fff7b2040c4 0x7fff7b2040d8
    std::cout << "二级指针地址：" ;
    for (int i=0; i < 3; i++){
        std::cout << *(ip2 + i) << " " ;
    }
    std::cout << std::endl;
    std::cout << "ip2步长: " << ip2 << " " << ip2 + 1 << std::endl; // ip2 存放了 arr_2d 中第一个元素的地址，即 arr1 的地址，ip2 + 1 存放了 arr_2d 中第二个元素的地址，即 arr2 的地址

    for (int i=0; i < 3; i++){
        for (int j=0; j< 5; j++){
            std::cout << *(arr_2d[i] + j)  << " "; // 
            std::cout << arr_2d[i] + j << std::endl;
        }
        std::cout << std::endl;
        for (int j=0; j < 5; j++){
            std::cout << *(*ip2 + j) << " ";
        }
        // for (int j=0; j < 5; j++){
        //     std::cout << *ip2[j] << " ";
        // }
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "ip2: " << ip2 << std::endl;
        ip2++;
    }

    return 0;
}