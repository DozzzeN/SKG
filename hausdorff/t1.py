import random


def genRandomStep(len, lowBound, highBound):
    length = 0
    randomStep = []
    # 少于三则无法分，因为至少要划分出一个三角形
    while len - length >= lowBound:
        step = random.randint(lowBound, highBound)
        randomStep.append(step)
        length += step
    # # 判断最后一个step连同剩余的值之和不超过bound
    # times = 0
    # while randomStep[-1] + len - length > highBound or randomStep[-1] + len - length < lowBound:
    #     randomStep[-1] = random.randint(lowBound, highBound)
    #     times += 1
    #     if times >= 5:
    #         break
    # randomStep[-1] += len - length

    return randomStep


print(genRandomStep(21, 3, 5))
