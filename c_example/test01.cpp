#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

int main() {
    const size_t SIZE = 100'000;  // 10万个元素
    const int THRESHOLD = 128;
    
    // 准备两个数组：一个随机，一个排序
    std::vector<int> random_data(SIZE);
    std::vector<int> sorted_data(SIZE);
    
    // 填充随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);  // 0-255的随机数
    
    for (size_t i = 0; i < SIZE; i++) {
        random_data[i] = dis(gen);
        sorted_data[i] = random_data[i];
    }
    
    
    // 需要累加结果，防止编译器优化掉整个循环
    long long sum_random = 0;
    long long sum_sorted = 0;
    
    // 测试随机数组
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < SIZE; i++) {
        if (random_data[i] >= THRESHOLD) {
            sum_random += random_data[i];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto random_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // 排序其中一个
    std::sort(sorted_data.begin(), sorted_data.end());
    
    // 测试排序数组
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < SIZE; i++) {
        if (sorted_data[i] >= THRESHOLD) {
            sum_sorted += sorted_data[i];
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto sorted_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // 输出结果
    std::cout << "随机数组  耗时: " << random_time << " 微秒, 累加值: " << sum_random << std::endl;
    std::cout << "排序数组  耗时: " << sorted_time << " 微秒, 累加值: " << sum_sorted << std::endl;
    std::cout << "速度提升: " << (double)random_time / sorted_time << " 倍" << std::endl;
    
    return 0;
}