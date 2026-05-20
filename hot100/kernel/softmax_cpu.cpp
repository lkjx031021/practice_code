#include <stdio.h>
#include <iostream>
#include <cmath>

void softmax_cpu(float* out, float* input, int N, int C){
  for (int i = 0; i < N; i++){
    const float* inp = input + i * C; // 每行的起始地址
    float* outp = out + i * C;        // 输出行的起始

    float max_val = -INFINITY; // 初始化为负无穷
    for (int j = 0; j < C; j++){
      if (inp[j] > max_val){
        max_val = inp[j]; // 找到当前行的最大值
      }
    }
    std::cout << "-----------" << std::endl;

    float sum = .0f; // 计算分母的和
    for (int j = 0; j < C; j++){
      outp[j] = expf(inp[j] - max_val); // 计算指数值，减去max_val防止溢出
      std::cout << outp[j] << " "; // 输出每个元素的指数值
      sum += outp[j]; // 累加分母的和
    }
    std::cout << std::endl;

    float norm = 1.0f / sum;
    for (int j = 0; j < C; j++){
      outp[j] *= norm; // 归一化,得到softmax输出
    }

  }
}


int main() {
  int N = 3;
  int C = 4;

  size_t size = N * C;
  float* input = (float*)malloc(size * sizeof(float));
  float* output = (float*)malloc(size * sizeof(float));

  for (int i = 0; i < size; i++){
    *(input + i) = float(i);
  }
  for (int i = 0; i < C * 2; i++){
    std::cout << input[i] << " ";
  }
  std::cout << std::endl;

  softmax_cpu(output, input, N, C);
  for (int i = 0; i < C * 2; i++){
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;
  free(input);
  free(output);
  return 0;
}