from itertools import permutations

# 测试是否密钥里面有连续的索引，例如1243里面有连续的索引12
# 进而统计有连续索引的密钥个数和全排列的密钥个数
# 发现全排列中超过一半密钥都是有连续索引的
for M in range(10):
    permutation = list(set(permutations(range(M))))
    successive_keys = []
    for i in range(len(permutation)):
        for j in range(1, M):
            if permutation[i][j] - permutation[i][j - 1] == 1:
                successive_keys.append(permutation[i])
                break
    non_successive_keys = list(set(permutation) - set(successive_keys))
    print("M: ", M, "num of non-successive keys:", len(non_successive_keys), "num of all keys:", len(permutation))
