import itertools
from collections import Counter
from scipy.special import comb, perm

n = 2
# n = Counter({0: 1, 1: 1})
# ('xy', 0) ('yx', 0)
# n = 2 Counter({1: 3, 0: 1})  {1:2, 0:0}
# ('xyxy', 1) ('xyyx', 1) ('yxxy', 0) ('yxyx', 1)
# n = 3 Counter({2: 4, 1: 4})  {2:3, 1:1}
# ('xyxyxy', 2) ('xyxyyx', 2) ('xyyxxy', 1) ('xyyxyx', 2)
# ('yxxyxy', 1) ('yxxyyx', 1) ('yxyxxy', 1) ('yxyxyx', 2)
# n = 4 Counter({2: 10, 3: 5, 1: 1})  {3:4, 2:4}
# xyxyxyxy:3\quad xyxyxyyx:3\quad xyxyyxxy:2\quad xyxyyxyx:3\\
# xyyxxyxy:2\quad xyyxxyyx:2\quad xyyxyxxy:2\quad xyyxyxyx:3\\
# yxxyxyxy:2\quad yxxyxyyx:2\quad yxxyyxxy:1\quad yxxyyxyx:2\\
# yxyxxyxy:2\quad yxyxxyyx:2\quad yxyxyxxy:2\quad yxyxyxyx:3\\


# ('xyxyxyxy', 3) ('xyxyxyyx', 3) ('xyxyyxxy', 2) ('xyxyyxyx', 3)
# ('xyyxxyxy', 2) ('xyyxxyyx', 2) ('xyyxyxxy', 2) ('xyyxyxyx', 3)
# ('yxxyxyxy', 2) ('yxxyxyyx', 2) ('yxxyyxxy', 1) ('yxxyyxyx', 2)
# ('yxyxxyxy', 2) ('yxyxxyyx', 2) ('yxyxyxxy', 2) ('yxyxyxyx', 3)
# n = 5 Counter({3: 20, 4: 6, 2: 6})
# ('xyxyxyxyxy', 4) ('xyxyxyxyyx', 4) ('xyxyxyyxxy', 3) ('xyxyxyyxyx', 4)
# ('xyxyyxxyxy', 3) ('xyxyyxxyyx', 3) ('xyxyyxyxxy', 3) ('xyxyyxyxyx', 4)
# ('xyyxxyxyxy', 3) ('xyyxxyxyyx', 3) ('xyyxxyyxxy', 2) ('xyyxxyyxyx', 3)
# ('xyyxyxxyxy', 3) ('xyyxyxxyyx', 3) ('xyyxyxyxxy', 3) ('xyyxyxyxyx', 4)
# ('yxxyxyxyxy', 3) ('yxxyxyxyyx', 3) ('yxxyxyyxxy', 2) ('yxxyxyyxyx', 3)
# ('yxxyyxxyxy', 2) ('yxxyyxxyyx', 2) ('yxxyyxyxxy', 2) ('yxxyyxyxyx', 3)
# ('yxyxxyxyxy', 3) ('yxyxxyxyyx', 3) ('yxyxxyyxxy', 2) ('yxyxxyyxyx', 3)
# ('yxyxyxxyxy', 3) ('yxyxyxxyyx', 3) ('yxyxyxyxxy', 3) ('yxyxyxyxyx', 4)
bools = list(itertools.product(["xy", "yx"], repeat=n))

print(len(bools))
# print(bools)
results = []
for i in range(len(bools)):
    s = ""
    for j in range(len(bools[i])):
        s += bools[i][j][0:1]
        s += bools[i][j][1:]
    results.append(s)
# print(results)
counts = []
for i in range(len(results)):
    c = 0
    for j in range(len(results[i])):
        if results[i][j:j + 3] == "xyx":
            c += 1
        elif results[i][j:j + 4] == "xyyx":
            c += 1
    counts.append(c)
    # counts.append(results[i].count('xyx') + results[i].count('xyyx'))
print(Counter(counts))
r = list(zip(results, counts))
print(r)
for i in range(0, len(r), 4):
    print(r[i], r[i + 1], r[i + 2], r[i + 3])
