from collections import Counter

import numpy as np
import scipy.stats

from segmentation.methods import normal2uniform, smooth, find_sub_opt_segment_method_sliding_threshold, \
    find_sub_opt_segment_method_sliding

np.random.seed(100000)
CSIa1Orig = np.random.normal(0, 1, 10000)

segLen = 4
segNum = 4

# 1W数据
# 4 8 16 32 64 128 256 512
# withIndexValue = True
# ({4: 570, 5: 56})
# ({8: 202, 7: 90, 9: 22})
# ({16: 84, 15: 47, 17: 22, 14: 2, 18: 1, 13: 1})
# ({30: 28, 31: 25, 32: 11, 29: 7, 33: 5, 28: 2})
# ({61: 11, 62: 9, 60: 7, 64: 5, 63: 3, 59: 2, 57: 1, 65: 1})
# ({120: 4, 122: 4, 121: 3, 126: 2, 123: 2, 119: 1, 124: 1, 118: 1, 125: 1})
# ({244: 3, 243: 2, 246: 2, 245: 1, 242: 1})
# ({488: 1, 493: 1, 490: 1, 486: 1})

# 100W数据
# 4 8 16 32 64 128 256 512
# withIndexValue = True
# ({4: 57144, 5: 5357})
# ({8: 19077, 7: 8723, 9: 3361, 10: 90})
# ({16: 8015, 15: 4091, 17: 2873, 14: 448, 18: 189, 13: 5, 19: 5})
# ({31: 2925, 30: 2429, 32: 1486, 29: 620, 33: 286, 28: 42, 34: 23, 27: 2})
# ({61: 1056, 62: 1041, 60: 676, 63: 549, 59: 260, 64: 213, 58: 52, 65: 48, 66: 6, 57: 4, 67: 1})
# ({122: 375, 123: 373, 124: 323, 121: 276, 125: 189, 120: 151, 126: 96, 119: 78, 127: 47, 128: 21, 118: 11, 129: 7, 116: 3, 117: 3})
# ({245: 131, 247: 125, 244: 125, 246: 110, 243: 109, 242: 84, 248: 78, 249: 59, 241: 45, 240: 28, 250: 26, 251: 22, 239: 13, 238: 8, 252: 4, 236: 3, 254: 2, 237: 2, 255: 1, 235: 1})
# ({492: 51, 488: 50, 489: 49, 493: 42, 491: 41, 490: 41, 487: 41, 486: 40, 494: 25, 485: 19, 495: 18, 496: 16, 484: 14, 483: 11, 497: 9, 499: 7, 482: 4, 501: 2, 481: 2, 480: 2, 500: 1, 479: 1, 478: 1, 498: 1})

for segNum in range(2, 10):
    segNum = int(2 ** segNum)
    keyLen = segNum * segLen

    withIndexValue = True

    # 是否排序
    # withoutSorts = [True, False]
    withoutSorts = [False]
    # 是否添加噪声
    # addNoises = ["pca", "mul", "add", ""]
    addNoises = ["mul"]

    for addNoise in addNoises:
        for withoutSort in withoutSorts:

            # 为了使用cox-box处理，将滤波放在总循环前面，结果与old版本的有略微的不一致
            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')

            CSIa1Orig = (CSIa1Orig - np.min(CSIa1Orig)) / (np.max(CSIa1Orig) - np.min(CSIa1Orig))
            CSIa1Orig = scipy.stats.boxcox(np.abs(CSIa1Orig) + 1e-4)[0]
            CSIa1Orig = normal2uniform(CSIa1Orig) * 2
            CSIa1Orig = np.array(CSIa1Orig)

            dataLen = len(CSIa1Orig)
            print("dataLen", dataLen)

            print("segLen", segLen)
            print("keyLen", keyLen / segLen)

            dataLenLoop = dataLen
            keyLenLoop = keyLen

            segment_lengths = []
            staInd = -keyLenLoop
            while staInd < dataLenLoop:
                staInd += keyLenLoop
                keyLen = segNum * segLen
                endInd = staInd + keyLen
                if endInd >= len(CSIa1Orig):
                    break

                tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]

                # 目的是把加噪音+无排序的结果降下来
                if addNoise == "mul":
                    np.random.seed(10000)
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                else:
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)

                if withIndexValue:
                    # 将测量值和其索引结合成二维数组
                    tmpCSIa1Index = np.array(tmpCSIa1).argsort().argsort()

                    # 将index和value放缩到同样的区间内
                    tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    tmpCSIa1Index = (tmpCSIa1Index - np.min(tmpCSIa1Index)) / (np.max(tmpCSIa1Index) - np.min(tmpCSIa1Index))
                    tmpCSIa1Ind = np.array(list(zip(tmpCSIa1, tmpCSIa1Index)))

                else:
                    if withoutSort:
                        tmpCSIa1Ind = np.array(tmpCSIa1)
                    else:
                        tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()

                # 滑动窗口分段
                if withIndexValue:
                    segment_method_ori = find_sub_opt_segment_method_sliding(
                        tmpCSIa1Ind, tmpCSIa1Ind, 3, 5)
                else:
                    if withoutSort is True:
                        segment_method_ori = find_sub_opt_segment_method_sliding(
                            tmpCSIa1Ind, tmpCSIa1Ind, 3, 5)
                        # segment_method_ori = find_sub_opt_segment_method_sliding_threshold(
                        #     tmpCSIa1Ind, tmpCSIa1Ind, 3, 5, 0.005)
                    else:
                        segment_method_ori = find_sub_opt_segment_method_sliding_threshold(
                            tmpCSIa1Ind, tmpCSIa1Ind, 3, 5, 60)

                num_segments = len(segment_method_ori)

                segment_lengths.append(num_segments)

            print("segment lengths", Counter(segment_lengths))
            print("\n")
