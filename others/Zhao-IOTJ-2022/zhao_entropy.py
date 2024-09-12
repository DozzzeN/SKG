import numpy as np
from scipy.io import loadmat


# 根据样本数据估计概率分布
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


# RSS security strength
# mi1 0.16313411259824803
# si1 0.20425408124939995
# mo1 0.18468915916178602
# so1 0.23201999747297797

# CSI security strength
# mi1 0.21761972452223396
# si1 0.17766255946326023
# mo1 0.21415983087087764
# so1 0.17459009111937704


keys = []

names = [
    'key_csi_mi.mat',
    'key_csi_mo.mat',
    'key_csi_si.mat',
    'key_csi_so.mat',
    'key_rss_mi.mat',
    'key_rss_mo.mat',
    'key_rss_si.mat',
    'key_rss_so.mat'
]
for name in names:
    a_list = loadmat(name)['a_list_key']
    a_list = np.array(a_list).T
    for i in range(len(a_list)):
        keys.append("".join(map(str, a_list[i])))

    bit_len = 16
    distribution = frequency(keys)
    print(name, "minEntropy", minEntropy(distribution) / bit_len, "bit_len", bit_len, "keyLen", len(keys))
