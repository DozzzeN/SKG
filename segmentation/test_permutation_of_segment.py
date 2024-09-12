from functools import reduce
import math

# 如果有n1个a1，n2个a2，...，nk个ak，要计算所有这些字符的置换种类数，可以用多重集排列的公式来解决
def count_permutations(counts):
    total = sum(counts)
    denominator = reduce(lambda x, y: x * y, (math.factorial(count) for count in counts))
    return math.factorial(total) // denominator


# # 示例
# n1 = 1  # 数组中 'a1' 的个数
# n2 = 2  # 数组中 'a2' 的个数
# n3 = 1  # 数组中 'a3' 的个数
# result = count_permutations([n1, n2, n3])
# print(f"所有置换的种类数: {result}")