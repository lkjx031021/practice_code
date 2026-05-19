
#include <stdio.h>  // 标准输入输出库，用来使用 printf函数
#include <string.h> // 字符串处理库，用来使用 strlen函数

/**
 * 函数功能：计算无重复字符的最长子串长度
参数说明：
   char *s : 这是一个指针，指向我们要处理的字符串第一个字符的地址。
             在C语言中，字符串本质上就是一个字符数组，以 '\0' 结尾。
返回值：
   int     : 返回一个整数，表示最长子串的长度。
*/
int lengthOfLongestSubstring(char *s) {
    
    // 【步骤1：边界检查】
    // 如果传入的指针是空的(NULL)，或者字符串第一个字符就是结束符('\0')，说明是空串
    if (s == NULL || *s == '\0') {
        return 0; // 直接返回长度0
    }

    // 【步骤2：初始化哈希表（查找表）】
    // int charIndex[128]; 
    // 这里定义了一个包含128个整数的数组。
    // 为什么是128？因为标准ASCII码表有128个字符（0-127）。
    // 我们可以用字符的ASCII值作为数组的下标（索引）。
    // 比如 'a' 的ASCII是97，我们就用 charIndex[97] 来存 'a' 的信息。
    int charIndex[128]; 

    // 初始化数组，把所有位置都设为 -1。
    // -1 是一个特殊的标记，表示“这个字符还没出现过”。
    for (int i = 0; i < 128; i++) {
        charIndex[i] = -1;
    }

    int maxLength = 0;   // 用于记录目前找到的“最长”长度，初始为0
    int left = 0;        // 滑动窗口的“左边界”下标，初始从0开始
    int len = strlen(s); // strlen是库函数，计算字符串 s 的实际长度

    // 【步骤3：开始遍历字符串（移动右边界）】
    // right 是滑动窗口的“右边界”，从0一直走到字符串末尾
    for (int right = 0; right < len; right++) {
        
        // 获取当前右边界指向的字符
        // (unsigned char) 是为了防止某些编译器把 char 当作负数处理导致数组越界
        // 简单理解：把字符转换成它对应的数字编号（ASCII码）
        unsigned char c = (unsigned char)s[right];

        // 【核心逻辑判断】
        // charIndex[c] 存的是字符 c **上一次**出现的位置下标。
        // 如果 charIndex[c] >= left，说明：
        // 1. 这个字符之前出现过（因为如果是-1就不可能>=left，left最小是0）
        // 2. 而且它出现的位置在当前窗口内（left右边或就是left）
        // 这意味着：我们在当前窗口里遇到了重复字符！
        if (charIndex[c] >= left) {
            // 既然重复了，左边界 left 必须跳过那个旧的重复字符。
            // 新的左边界 = 旧字符位置 + 1
            left = charIndex[c] + 1;
        }

        // 【更新哈希表】
        // 不管有没有重复，我们都要更新字符 c 的最新位置为当前的 right
        // 这样下次再遇到 c，就知道它最近一次是在哪出现的了
        charIndex[c] = right;

        // 【计算当前窗口长度并更新最大值】
        // 窗口范围是 [left, right]，长度公式是：右边界 - 左边界 + 1
        int currentLen = right - left + 1;
        
        // 如果当前窗口比之前记录的最大值还大，就更新最大值
        if (currentLen > maxLength) {
            maxLength = currentLen;
        }
    }

    // 【步骤4：返回结果】
    return maxLength;
}

// 主函数，程序入口
int main() {
    // 定义几个测试字符串
    // 注意：在C语言中，双引号括起来的是字符串常量
    char *test1 = "abcabcbb";
    char *test2 = "bbbbb";
    char *test3 = "pwwkew";

    // 调用函数并打印结果
    // %s 表示打印字符串，%d 表示打印整数
    printf("字符串: \"%s\" -> 最长无重复子串长度: %d\n", test1, lengthOfLongestSubstring(test1));
    printf("字符串: \"%s\" -> 最长无重复子串长度: %d\n", test2, lengthOfLongestSubstring(test2));
    printf("字符串: \"%s\" -> 最长无重复子串长度: %d\n", test3, lengthOfLongestSubstring(test3));

    return 0; // 程序正常结束
}