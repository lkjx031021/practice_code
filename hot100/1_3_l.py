def lengthOfLongestSubstring(s: str) -> int:
    # 记录字符最后出现下标，ASCII共128位
    last_pos = [-1] * 128
    left = 0
    max_len = 0

    for right, char in enumerate(s):
        idx = ord(char)
        # 字符在窗口内重复，左边界右移
        if last_pos[idx] >= left:
            left = last_pos[idx] + 1
        # 更新当前字符最新位置
        last_pos[idx] = right
        # 更新最大长度
        max_len = max(max_len, right - left + 1)
    return max_len

    

a = 'abcbdebb'
print(lengthOfLongestSubstring(a))

def lll(s):
    last_pos = [-1] * 128 # 记录字符上一次出现的位置，ASCII共128位
    max_len = 0
    left = 0

    for right, char in enumerate(s):
        idx = ord(char)

        # 如果当前字符没出现过则值是-1，或者上一次出现的位置在当前窗口左边界的左侧，则不需要更新左边界
        if last_pos[idx] >= left:
            left = last_pos[idx] + 1

        last_pos[idx] = right
        max_len = max(max_len, right - left + 1)
    return max_len

print(lll(a))