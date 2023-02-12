import numpy as np


def internalReproduce(list, dimension):
    res = []
    orders = []
    for i in range(dimension):
        isRepeated = True
        tmp = []
        while isRepeated is True:
            tmp = np.random.permutation(len(list))
            for j in range(len(orders)):
                for k in range(len(list)):
                    if orders[j][k] == tmp[k]:
                        break
                else:
                    continue
                break
            else:
                isRepeated = False
        orders.append(tmp)
    return orders


l = []
for i in range(128):
    l.append(i)
print(internalReproduce(l, 10))
