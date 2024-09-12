# 根据样本数据估计概率分布
import numpy as np


def frequency(samples):
    samples = np.array(samples)
    total_samples = len(samples)

    # 使用字典来记录每个数值出现的次数
    frequency_count = {}
    for sample in samples:
        if sample in frequency_count:
            frequency_count[sample] += 1
        else:
            frequency_count[sample] = 1

    # 计算每个数值的频率，即概率分布
    frequency = []
    for sample in frequency_count:
        frequency.append(frequency_count[sample] / total_samples)

    return frequency


def minEntropy(probabilities):
    return -np.log2(np.max(probabilities) + 1e-12)

bit_len = 8
keys = []
for i in range(100 * 2 ** bit_len):
    keys.append(np.random.randint(0, 2 ** bit_len))

distribution = frequency(keys)
print("minEntropy", minEntropy(distribution) / bit_len, "bit_len", bit_len, "keyLen", len(keys))